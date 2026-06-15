from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from wrf_ensembly import config


@dataclass
class CycleInformation:
    start: datetime
    end: datetime
    cycle_offset: timedelta
    index: int
    output_interval: int | None
    forecast_end: datetime | None = None  # type: ignore[assignment]
    """
    End time of the forward (member) run, which may extend past `end` by
    `time_control.forecast_extension` minutes (clamped to the experiment end).
    Used only for the forward run and boundary conditions; the assimilation
    boundary remains `end`. Defaults to `end` when no extension is configured.
    """

    def __post_init__(self):
        if self.forecast_end is None:
            self.forecast_end = self.end

    def __str__(self) -> str:
        return f"Cycle #{self.index}: {self.start} -> {self.end}, Offset: {self.cycle_offset.seconds // 60 // 60}h"


def cycles_to_dataframe(cycles: list[CycleInformation]) -> pd.DataFrame:
    """Converts a list of CycleInformation to a DataFrame with cycle_index, start_time, end_time."""
    return pd.DataFrame(
        [
            {
                "cycle_index": c.index,
                "start_time": pd.Timestamp(c.start).tz_convert("UTC"),
                "end_time": pd.Timestamp(c.end).tz_convert("UTC"),
            }
            for c in cycles
        ]
    )


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
        forecast_extension = cfg.time_control.forecast_extension
        if i in cfg.time_control.cycles:
            cycle_cfg = cfg.time_control.cycles[i]
            if cycle_cfg.duration is not None:
                duration = cycle_cfg.duration
            if cycle_cfg.output_interval is not None:
                output_interval = cycle_cfg.output_interval
            if cycle_cfg.forecast_extension is not None:
                forecast_extension = cycle_cfg.forecast_extension

        cycle_start = t
        cycle_end = t + timedelta(minutes=duration)
        # Clamp to end of experiment
        if cycle_end > experiment_end:
            cycle_end = experiment_end

        # Extend the forward run past the cycle end for independent forecasts. Clamp to
        # the experiment end, since no boundary data exists beyond it.
        forecast_end = cycle_end + timedelta(minutes=forecast_extension)
        if forecast_end > experiment_end:
            forecast_end = experiment_end

        cycle = CycleInformation(
            start=cycle_start,
            end=cycle_end,
            cycle_offset=cycle_start - experiment_start,
            index=i,
            output_interval=output_interval,
            forecast_end=forecast_end,
        )

        cycles.append(cycle)
        t += timedelta(minutes=duration)
        i += 1

    cycles = sorted(cycles, key=lambda c: c.index)

    return cycles
