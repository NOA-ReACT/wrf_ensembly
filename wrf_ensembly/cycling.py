from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from wrf_ensembly import config


@dataclass
class CycleInformation:
    start: datetime
    end: datetime
    cycle_offset: timedelta
    index: int
    output_interval: Optional[int]

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

    t = experiment_start
    i = 0
    cycles = []
    while t < experiment_end:
        # If a custom duration is specified, don't use the analysis interval for this cycle
        duration = analysis_interval
        output_interval = None
        if i in cfg.time_control.cycles:
            cycle_cfg = cfg.time_control.cycles[i]
            if cycle_cfg.duration is not None:
                duration = cycle_cfg.duration
            if cycle_cfg.output_interval is not None:
                output_interval = cycle_cfg.output_interval

        cycle_start = t
        cycle_end = t + timedelta(minutes=duration)
        # Clamp to end of experiment
        if cycle_end > experiment_end:
            cycle_end = experiment_end

        cycle = CycleInformation(
            start=cycle_start,
            end=cycle_end,
            cycle_offset=cycle_start - experiment_start,
            index=i,
            output_interval=output_interval,
        )

        cycles.append(cycle)
        t += timedelta(minutes=duration)
        i += 1

    cycles = sorted(cycles, key=lambda c: c.index)

    return cycles
