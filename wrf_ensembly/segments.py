"""
Segments: running several cycles as one WRF run, in the `restart` cycling mode.

A cycle stays the bookkeeping unit (status, outputs, postprocessing all work per cycle),
while a segment is the execution unit: the members run once from the start of its first
cycle to the end of its last one. Segments end at the cycles where the members have to
stop (see `find_stops`), or earlier if the run would not fit in the walltime limit.

Everything in this module is pure, so the planning can be tested without an experiment.
The `Experiment` methods gather the inputs and store the plans.
"""

import datetime as dt
import math
from dataclasses import asdict, dataclass
from typing import Any

from wrf_ensembly.cycling import CycleInformation

STOP_OBSERVATIONS = "observations"
STOP_KEEP_RESTART = "keep_restart_files_for_cycles"
STOP_RUN_UNTIL = "run_until"
STOP_LAST_CYCLE = "last_cycle"
STOP_WALLTIME = "walltime"


@dataclass
class SegmentPlan:
    """
    Which cycles one segment covers and how it runs. Written once, when the segment is
    about to start, and not changed after a member has started it.
    """

    first: int
    """First cycle of the segment, where the experiment pointer is while it runs"""

    last: int
    """Last cycle (inclusive), where the members stop"""

    checkpoint_interval_min: int
    """WRF's `restart_interval` for the run, divides the segment's length"""

    estimated_walltime_s: float
    """Expected member runtime, without the safety factor"""

    rate_s_per_sim_hour: float
    """Wall clock seconds per simulated hour the estimate is based on"""

    rate_source: str
    """Where the rate came from (`expected_walltime_per_sim_hour` or `statistics`)"""

    stop_reason: str
    """Why the segment ends at `last` (one of the `STOP_*` constants)"""

    run_until: int | None = None
    """The `--run-until` cycle that was in effect when planning, if any"""

    created: str | None = None
    """When the plan was made (ISO timestamp)"""

    @property
    def cycles(self) -> range:
        return range(self.first, self.last + 1)

    def __contains__(self, cycle_i: int) -> bool:
        return self.first <= cycle_i <= self.last

    def __str__(self) -> str:
        if self.first == self.last:
            return f"cycle {self.first}"
        return f"cycles {self.first}-{self.last}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentPlan":
        return cls(
            first=int(data["first"]),
            last=int(data["last"]),
            checkpoint_interval_min=int(data["checkpoint_interval_min"]),
            estimated_walltime_s=float(data["estimated_walltime_s"]),
            rate_s_per_sim_hour=float(data["rate_s_per_sim_hour"]),
            rate_source=str(data["rate_source"]),
            stop_reason=str(data["stop_reason"]),
            run_until=None if data.get("run_until") is None else int(data["run_until"]),
            created=data.get("created"),
        )


def find_stops(
    n_cycles: int,
    observation_cycles: set[int],
    keep_restart_cycles: set[int],
    run_until: int | None = None,
) -> dict[int, str]:
    """
    The cycles a segment must end at, with the reason. A cycle with observations needs
    the filter at its end, a cycle in `keep_restart_files_for_cycles` a restart file at
    its end that another experiment can start from, and `--run-until` asks to stop.
    The last cycle is always a stop.
    """

    stops: dict[int, str] = {}
    # In order of precedence for the reason, later ones overwrite
    if run_until is not None and 0 <= run_until < n_cycles:
        stops[run_until] = STOP_RUN_UNTIL
    for i in keep_restart_cycles:
        if 0 <= i < n_cycles:
            stops[i] = STOP_KEEP_RESTART
    for i in observation_cycles:
        if 0 <= i < n_cycles:
            stops[i] = STOP_OBSERVATIONS
    stops.setdefault(n_cycles - 1, STOP_LAST_CYCLE)
    return stops


def simulated_hours(cycles: list[CycleInformation], first: int, last: int) -> float:
    """Simulated hours from the start of `first` to the end of `last`"""

    return (cycles[last].end - cycles[first].start).total_seconds() / 3600


def checkpoint_interval(
    segment_min: int, max_interval_min: int, analysis_interval_min: int
) -> int:
    """
    The `restart_interval` (minutes) for a segment of `segment_min` minutes: the longest
    interval that is at most `max_interval_min` and divides the segment, since WRF counts
    the interval from the start of the run and the segment's end needs a restart file.

    Intervals that are a multiple of the analysis interval are preferred, so checkpoints
    fall on cycle boundaries. Only a shorter last cycle (clamped to the experiment end)
    makes that impossible; then any divisor of at least an hour is used, or else no
    checkpoints before the end.
    """

    if segment_min <= max_interval_min:
        return segment_min

    divisors = [d for d in range(1, max_interval_min + 1) if segment_min % d == 0]
    on_cycle_boundaries = [d for d in divisors if d % analysis_interval_min == 0]
    if on_cycle_boundaries:
        return on_cycle_boundaries[-1]
    at_least_an_hour = [d for d in divisors if d >= 60]
    if at_least_an_hour:
        return at_least_an_hour[-1]
    return segment_min


def plan_segment(
    cycles: list[CycleInformation],
    first: int,
    stops: dict[int, str],
    rate_s_per_sim_hour: float,
    rate_source: str,
    max_walltime_s: float,
    safety_factor: float,
    checkpoint_interval_hours: float,
    run_until: int | None = None,
    now: dt.datetime | None = None,
) -> SegmentPlan:
    """
    Plans the segment starting at cycle `first`: it extends cycle by cycle until it
    reaches a stop, or until one more cycle would make the estimated runtime (times the
    safety factor) longer than `max_walltime_s`. A segment always has at least one cycle,
    even if that one doesn't fit, so the experiment can't get stuck.
    """

    if not 0 <= first < len(cycles):
        raise ValueError(f"Cycle {first} does not exist")

    def estimate(last: int) -> float:
        return simulated_hours(cycles, first, last) * rate_s_per_sim_hour

    stops = stops | {len(cycles) - 1: stops.get(len(cycles) - 1, STOP_LAST_CYCLE)}
    last = first
    reason = stops.get(first)
    while reason is None:
        if estimate(last + 1) * safety_factor > max_walltime_s:
            reason = STOP_WALLTIME
            break
        last += 1
        reason = stops.get(last)

    segment_min = round(simulated_hours(cycles, first, last) * 60)
    analysis_interval_min = round(
        (cycles[first].end - cycles[first].start).total_seconds() / 60
    )
    interval = checkpoint_interval(
        segment_min, math.floor(checkpoint_interval_hours * 60), analysis_interval_min
    )

    now = now or dt.datetime.now(dt.timezone.utc)
    return SegmentPlan(
        first=first,
        last=last,
        checkpoint_interval_min=interval,
        estimated_walltime_s=estimate(last),
        rate_s_per_sim_hour=rate_s_per_sim_hour,
        rate_source=rate_source,
        stop_reason=reason,
        run_until=run_until,
        created=now.isoformat(),
    )
