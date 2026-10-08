"""
Storage for experiment status, as a tree of small files under `status/`.

Each fact is one file with exactly one writer, written atomically (temp file + rename).
No locking is involved anywhere, which is the whole point: the ensemble members advance
as separate jobs on separate nodes, and any lock-based store (sqlite, lockfiles) is
unreliable on the network filesystems these experiments live on.

This works because no field is ever read-modify-written by two processes at once:

- The per-member files are written in parallel, but each job only ever writes its own
  file, so writers never collide.
- Everything else (current cycle, cycle markers, optional operations) is written by
  serial commands (`filter`, `analysis`, `cycle`) or interactively by the user.

Layout::

    status/
      experiment.json                      {"version": 1, "current_cycle": 5}
      cycles/cycle_000/
        members/member_00.json             advancement + runtime statistics
        filter_complete                    marker files, existence means done
        analysis_complete
        cycle_complete
        ops/apply_perturbations            optional operation markers
      segments/cycle_003.json              plan of the segment starting at cycle 3

Everything is plain JSON/text, so a stuck experiment can be inspected and repaired with
`ls`, an editor and `rm`.
"""

import datetime as dt
import json
import os
import re
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wrf_ensembly import utils
from wrf_ensembly.console import logger
from wrf_ensembly.segments import SegmentPlan

from .dataclasses import RuntimeStatistics
from .paths import ExperimentPaths

STATE_VERSION = 1

CYCLE_DIR_RE = re.compile(r"^cycle_(\d+)$")
MEMBER_FILE_RE = re.compile(r"^member_(\d+)\.json$")
SEGMENT_FILE_RE = re.compile(r"^cycle_(\d+)\.json$")


@dataclass
class MemberRecord:
    """The stored status of one member for one cycle"""

    cycle: int
    i: int
    advanced: bool
    runtime: RuntimeStatistics | None = None
    host: str | None = None
    job_id: str | None = None


class ExperimentState:
    """
    Reads and writes the experiment status files.

    Readers are deliberately forgiving: a missing, empty or unparseable file is treated
    as "not done" with a warning, never as an error. One bad file should not be able to
    wedge an experiment.
    """

    def __init__(self, paths: ExperimentPaths, n_members: int):
        self.paths = paths
        self.n_members = n_members

    # Reading/writing raw files

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        """
        Read a JSON object from a file, returning None if it is missing or unreadable.

        The contents are deliberately not validated beyond being an object: these files
        can be hand-edited, so every caller checks the fields it needs rather than
        trusting the shape.
        """

        try:
            content = path.read_text()
        except FileNotFoundError:
            return None
        except OSError as e:
            logger.warning(f"Could not read status file {path}: {e}")
            return None

        if not content.strip():
            logger.warning(f"Status file {path} is empty, ignoring")
            return None

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Status file {path} is not valid JSON ({e}), ignoring")
            return None

        # Valid JSON that is not an object (a list, a bare number, ...) would break
        # every caller further down
        if not isinstance(data, dict):
            logger.warning(
                f"Status file {path} contains {type(data).__name__}, expected an object, ignoring"
            )
            return None

        return data

    def _write_json(self, path: Path, data: dict[str, Any]):
        utils.atomic_write_text(path, json.dumps(data, indent=2) + "\n")

    # Experiment-level state

    def initialize(self):
        """Create the status directory tree and experiment.json, if not already there"""

        self.paths.status_cycles.mkdir(parents=True, exist_ok=True)
        if self._read_json(self.paths.status_experiment_file) is None:
            self._write_json(
                self.paths.status_experiment_file,
                {"version": STATE_VERSION, "current_cycle": 0},
            )

    def get_current_cycle(self) -> int:
        """Which cycle the experiment is currently on"""

        data = self._read_json(self.paths.status_experiment_file)
        if data is None:
            return 0

        cycle = data.get("current_cycle", 0)
        if not isinstance(cycle, int) or cycle < 0:
            logger.warning(
                f"Invalid current_cycle {cycle!r} in {self.paths.status_experiment_file}, using 0"
            )
            return 0
        return cycle

    def set_current_cycle(self, cycle: int):
        """
        Move the experiment to a given cycle. Only ever called by serial commands
        (`cycle`, `link-experiment`, `status set-experiment`).
        """

        data = self._read_json(self.paths.status_experiment_file) or {
            "version": STATE_VERSION
        }
        data["current_cycle"] = cycle
        self._write_json(self.paths.status_experiment_file, data)

    # Member status

    def get_member(self, cycle: int, i: int) -> MemberRecord | None:
        """Read one member's status for a cycle, or None if it has not been written"""

        data = self._read_json(self.paths.member_status_path(cycle, i))
        if data is None:
            return None

        runtime = None
        try:
            if data.get("start") is not None and data.get("end") is not None:
                runtime = RuntimeStatistics(
                    cycle=cycle,
                    start=dt.datetime.fromisoformat(data["start"]),
                    end=dt.datetime.fromisoformat(data["end"]),
                    duration_s=int(data["duration_s"]),
                    simulated_s=(
                        int(data["simulated_s"])
                        if data.get("simulated_s") is not None
                        else None
                    ),
                )
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(
                f"Could not read runtime statistics for cycle {cycle} member {i}: {e}"
            )

        return MemberRecord(
            cycle=cycle,
            i=i,
            advanced=bool(data.get("advanced", False)),
            runtime=runtime,
            host=data.get("host"),
            job_id=data.get("job_id"),
        )

    def set_member_advanced(
        self,
        cycle: int,
        i: int,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        duration_s: int | None = None,
        simulated_s: int | None = None,
    ):
        """
        Record that a member finished running the model for a cycle, along with how long
        it took (and how much time it simulated, which is more than the cycle for a
        segment). This is the only write that happens in parallel across nodes; each job
        writes only its own file.

        The runtime statistics are optional so that `status reconcile` can mark a member
        as advanced based on its output files, where the timings are not recoverable.
        """

        data = {
            "advanced": True,
            "start": start.isoformat() if start is not None else None,
            "end": end.isoformat() if end is not None else None,
            "duration_s": duration_s,
            "simulated_s": simulated_s,
            "host": socket.gethostname(),
            "job_id": os.environ.get("SLURM_JOB_ID"),
        }
        self._write_json(self.paths.member_status_path(cycle, i), data)

    def get_advanced_members(self, cycle: int) -> set[int]:
        """Indices of all members that have advanced for a given cycle"""

        members_dir = self.paths.cycle_members_path(cycle)
        try:
            entries = os.listdir(members_dir)
        except FileNotFoundError:
            return set()
        except OSError as e:
            logger.warning(f"Could not list {members_dir}: {e}")
            return set()

        advanced = set()
        for entry in entries:
            match = MEMBER_FILE_RE.match(entry)
            if match is None:
                continue

            i = int(match.group(1))
            if i >= self.n_members:
                # Left over from a run with a larger ensemble, ignore
                continue

            record = self.get_member(cycle, i)
            if record is not None and record.advanced:
                advanced.add(i)

        return advanced

    def count_advanced(
        self, cycle: int, expect: int | None = None, timeout: float = 0.0
    ) -> int:
        """
        Count how many members have advanced for a cycle.

        On NFS a directory listing can be served from the attribute cache for up to
        `acdirmax` (~30s by default), so a member that has just finished may not be
        visible yet. When `expect` is given, keep re-listing until the count is reached
        or `timeout` seconds pass. Lustre and GPFS have coherent metadata, so this
        returns on the first try there.
        """

        n = len(self.get_advanced_members(cycle))
        if expect is None or n >= expect or timeout <= 0:
            return n

        logger.info(
            f"Only {n}/{expect} members visible for cycle {cycle}, waiting up to {timeout:.0f}s "
            "in case the filesystem is serving a stale directory listing"
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(min(5.0, max(0.0, deadline - time.monotonic())))
            n = len(self.get_advanced_members(cycle))
            if n >= expect:
                logger.info(f"All {expect} members visible for cycle {cycle}")
                return n

        return n

    def clear_member(self, cycle: int, i: int):
        """Forget that a member advanced for a cycle"""

        self.paths.member_status_path(cycle, i).unlink(missing_ok=True)

    def clear_cycle_members(self, cycle: int):
        """Forget the advancement status of every member for a cycle"""

        for i in range(self.n_members):
            self.clear_member(cycle, i)

    # Runtime statistics, stored inside the member files

    def _iter_cycle_dirs(self):
        """Yield (cycle_index, path) for every cycle directory that exists"""

        try:
            entries = sorted(os.listdir(self.paths.status_cycles))
        except FileNotFoundError:
            return
        except OSError as e:
            logger.warning(f"Could not list {self.paths.status_cycles}: {e}")
            return

        for entry in entries:
            match = CYCLE_DIR_RE.match(entry)
            if match is not None:
                yield int(match.group(1)), self.paths.status_cycles / entry

    def get_all_runtime_statistics(self) -> list[tuple[RuntimeStatistics, int]]:
        """
        Every recorded model run as (statistics, member index), across all cycles.

        This reads one file per (cycle, member), so it is only used by the `status`
        commands - never on the startup path of a command that runs the model.
        """

        stats = []
        for cycle, cycle_dir in self._iter_cycle_dirs():
            try:
                entries = os.listdir(cycle_dir / "members")
            except FileNotFoundError:
                continue
            except OSError as e:
                logger.warning(f"Could not list {cycle_dir / 'members'}: {e}")
                continue

            for entry in sorted(entries):
                match = MEMBER_FILE_RE.match(entry)
                if match is None:
                    continue

                i = int(match.group(1))
                record = self.get_member(cycle, i)
                if record is not None and record.runtime is not None:
                    stats.append((record.runtime, i))

        stats.sort(key=lambda s: (s[0].cycle, s[1]))
        return stats

    def clear_runtime_statistics(self):
        """
        Drop the timing information from every member file, keeping the advancement
        status intact.
        """

        for cycle, _ in self._iter_cycle_dirs():
            for i in range(self.n_members):
                path = self.paths.member_status_path(cycle, i)
                data = self._read_json(path)
                if data is None:
                    continue

                data["start"] = None
                data["end"] = None
                data["duration_s"] = None
                data["simulated_s"] = None
                self._write_json(path, data)

    # Cycle markers (filter_complete, analysis_complete, cycle_complete)

    def is_marker_set(self, cycle: int, name: str) -> bool:
        return self.paths.cycle_marker_path(cycle, name).exists()

    def set_marker(self, cycle: int, name: str):
        utils.atomic_write_text(
            self.paths.cycle_marker_path(cycle, name),
            f"{dt.datetime.now(dt.timezone.utc).isoformat()} {socket.gethostname()}\n",
        )

    def clear_marker(self, cycle: int, name: str):
        self.paths.cycle_marker_path(cycle, name).unlink(missing_ok=True)

    def clear_cycle_markers(self, cycle: int, names: list[str]):
        for name in names:
            self.clear_marker(cycle, name)

    # Optional operations (perturbation generation/application)

    def is_optional_operation_complete(self, cycle: int, name: str) -> bool:
        return self.paths.cycle_op_path(cycle, name).exists()

    def mark_optional_operation_complete(self, cycle: int, name: str):
        utils.atomic_write_text(
            self.paths.cycle_op_path(cycle, name),
            f"{dt.datetime.now(dt.timezone.utc).isoformat()} {socket.gethostname()}\n",
        )

    # Segment plans, written by serial commands only (`cycle`, `ensemble setup`,
    # `plan-segment`, `run-experiment`)

    def get_segment_plan(self, first_cycle: int) -> SegmentPlan | None:
        """The plan of the segment starting at a cycle, or None if there is none"""

        path = self.paths.segment_plan_path(first_cycle)
        data = self._read_json(path)
        if data is None:
            return None
        try:
            return SegmentPlan.from_dict(data)
        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Could not read segment plan {path} ({e}), ignoring")
            return None

    def get_segment_plans(self) -> list[SegmentPlan]:
        """Every stored segment plan, by first cycle"""

        try:
            entries = sorted(os.listdir(self.paths.status_segments))
        except FileNotFoundError:
            return []
        except OSError as e:
            logger.warning(f"Could not list {self.paths.status_segments}: {e}")
            return []

        plans = []
        for entry in entries:
            match = SEGMENT_FILE_RE.match(entry)
            if match is None:
                continue
            plan = self.get_segment_plan(int(match.group(1)))
            if plan is not None:
                plans.append(plan)
        return plans

    def set_segment_plan(self, plan: SegmentPlan):
        self._write_json(self.paths.segment_plan_path(plan.first), plan.to_dict())

    def clear_segment_plan(self, first_cycle: int):
        self.paths.segment_plan_path(first_cycle).unlink(missing_ok=True)

    # Whole-experiment reset

    def reset(self):
        """Reset the experiment to cycle 0, forgetting all per-cycle status and plans"""

        if self.paths.status_cycles.exists():
            utils.rm_tree(self.paths.status_cycles)
        if self.paths.status_segments.exists():
            utils.rm_tree(self.paths.status_segments)
        self.paths.status_cycles.mkdir(parents=True, exist_ok=True)
        self.set_current_cycle(0)
