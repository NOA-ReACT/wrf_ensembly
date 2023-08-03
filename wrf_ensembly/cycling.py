from dataclasses import dataclass
from datetime import datetime, timedelta

from wrf_ensembly import config


@dataclass
class CycleInformation:
    start: datetime
    end: datetime
    cycle_offset: timedelta
    index: int

    def __str__(self):
        return f"Cycle #{self.index}: {self.start} -> {self.end}, Offset: {self.cycle_offset.seconds // 60 // 60}h"


def get_cycle_information(cfg: config.Config) -> list[CycleInformation]:
    """
    Get a list of cycle information objects for the given configuration.

    Args:
        cfg: The experiment configuration
    """

    experiment_start = cfg.time_control.start
    experiment_end = cfg.time_control.end
    analysis_interval = cfg.time_control.analysis_interval

    cycle_count = int(
        (experiment_end - experiment_start).total_seconds() // 60 // analysis_interval
    )

    cycles = []
    for i in range(cycle_count):
        cycle_start = experiment_start + timedelta(minutes=i * analysis_interval)
        cycle_end = cycle_start + timedelta(minutes=analysis_interval)
        cycle = CycleInformation(
            start=cycle_start,
            end=cycle_end,
            cycle_offset=cycle_start - experiment_start,
            index=i,
        )
        cycles.append(cycle)
    cycles = sorted(cycles, key=lambda c: c.index)

    return cycles
