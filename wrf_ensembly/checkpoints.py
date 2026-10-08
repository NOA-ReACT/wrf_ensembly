"""
Checkpoints: the WRF restart files members write during a segment (see `segments.py`),
so a job that dies can continue from the newest one instead of the segment's start.

A restart file WRF was writing when the job died can look fine (it opens, its `Times`
is right) and still be missing data, so a checkpoint is only used once it is confirmed:
WRF logs `Timing for Writing restart` after writing one, and since restart files come at
fixed intervals from the start of the run, the k-th such line confirms the one at
`start + k * interval`. A `CheckpointWatcher` records the confirmed ones while wrf.exe
runs, and deletes all but the newest few, so a segment needs about as much space as a
single cycle, whatever its length.
"""

import datetime as dt
import json
import threading
from pathlib import Path

import netCDF4

from wrf_ensembly import update_bc, utils, wrf
from wrf_ensembly.console import logger

CONFIRMED_FILE = "checkpoints.json"
"""Confirmed checkpoint times, in the checkpoint directory"""

RESTART_WRITTEN = "Timing for Writing restart for domain"
"""What WRF logs (in rsl.out.0000) after it has written a restart file"""


def list_checkpoints(directory: Path) -> list[tuple[dt.datetime, Path]]:
    """The restart files in a directory with their time (from the name), oldest first"""

    found = []
    for f in directory.glob("wrfrst_d01_*"):
        try:
            found.append((wrf.wrfout_time(f.name), f))
        except ValueError:
            continue  # Not a restart file of WRF's naming, e.g. a temporary file
    return sorted(found)


def is_complete(path: Path, time: dt.datetime) -> bool:
    """
    Whether a restart file opens and its `Times` says it is the state at `time`. Only a
    sanity check, see `confirmed_times` for whether WRF finished writing it.
    """

    try:
        with netCDF4.Dataset(path, "r") as ds:  # type: ignore
            times = update_bc.parse_wrf_times(ds["Times"])
    except (OSError, IndexError, KeyError, RuntimeError, ValueError) as e:
        logger.warning(f"Can't read checkpoint {path}: {e}")
        return False
    return times == [time]


def confirmed_times(directory: Path) -> set[dt.datetime]:
    """The checkpoint times in a directory that WRF finished writing"""

    try:
        data = json.loads((directory / CONFIRMED_FILE).read_text())
        return {dt.datetime.fromisoformat(t) for t in data["confirmed"]}
    except FileNotFoundError:
        return set()
    except (OSError, ValueError, KeyError, TypeError) as e:
        logger.warning(f"Can't read {directory / CONFIRMED_FILE}: {e}")
        return set()


def confirm(directory: Path, times: set[dt.datetime]) -> None:
    """Adds checkpoint times to the confirmed ones of a directory"""

    known = confirmed_times(directory)
    if times <= known:
        return
    utils.atomic_write_text(
        directory / CONFIRMED_FILE,
        json.dumps({"confirmed": sorted(t.isoformat() for t in known | times)}) + "\n",
    )


def written_by_run(
    rsl_text: str, run_start: dt.datetime, interval: dt.timedelta
) -> set[dt.datetime]:
    """The restart times a run starting at `run_start` has written, from its log"""

    n = rsl_text.count(RESTART_WRITTEN)
    return {run_start + k * interval for k in range(1, n + 1)}


def prune(directory: Path, keep: int) -> list[Path]:
    """
    Deletes all but the newest `keep` restart files in a directory. Returns the deleted
    files.
    """

    checkpoints = list_checkpoints(directory)
    deleted = []
    for _, f in checkpoints[: max(len(checkpoints) - keep, 0)]:
        try:
            f.unlink()
        except FileNotFoundError:
            continue
        deleted.append(f)
    return deleted


class CheckpointWatcher:
    """
    Watches a run's checkpoints in a background thread while wrf.exe writes them:
    confirms the ones WRF finished writing (from its log), and deletes all but the
    newest `keep`, unless `keep` is None. Use as a context manager around the run:

        with CheckpointWatcher(directory, rsl_path, run_start, interval, keep=2):
            run wrf.exe

    Keep at least 2: the newest one may be half-written when the job is killed.
    """

    def __init__(
        self,
        directory: Path,
        rsl_path: Path,
        run_start: dt.datetime,
        interval: dt.timedelta,
        keep: int | None,
        poll_s: float = 30.0,
    ):
        if keep is not None and keep < 2:
            raise ValueError("Keep at least 2 checkpoints")
        self.directory = directory
        self.rsl_path = rsl_path
        self.run_start = run_start
        self.interval = interval
        self.keep = keep
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.poll_s):
            self.poll()

    def poll(self):
        # Never let a hiccup of the filesystem take the run down with it
        try:
            if self.rsl_path.exists():
                written = written_by_run(
                    self.rsl_path.read_text(errors="replace"),
                    self.run_start,
                    self.interval,
                )
                confirm(self.directory, written)
            if self.keep is not None:
                for f in prune(self.directory, self.keep):
                    logger.info(f"Removed old checkpoint {f.name}")
        except OSError as e:
            logger.warning(f"Could not check the checkpoints in {self.directory}: {e}")

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self._thread.join()
        self.poll()
