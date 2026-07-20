"""Reconstruct which cycle's forecast verified each observation, and at what lead.

The observations database records the interpolated model value but not its
provenance, so the owning cycle is derived here rather than read back. See
`assign_forecast_lead` for the semantics being replayed.
"""

import numpy as np
import pandas as pd

from wrf_ensembly.cycling import CycleInformation


def _to_naive_utc(times) -> pd.Series:
    """
    Normalise timestamps to timezone-naive UTC.

    Observation times reach us in three flavours: timezone-aware UTC from the
    DuckDB readers, timezone-naive from `read_obs_seq_nc`, and plain `datetime`
    objects from the cycle configuration. Comparing across flavours raises, so
    everything is coerced here before any interval test.
    """

    times = pd.to_datetime(pd.Series(times))
    if times.dt.tz is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    return times


def assign_forecast_lead(
    times,
    cycles: list[CycleInformation],
    prefer_extended_forecast: bool = True,
) -> pd.DataFrame:
    """
    Map observation times to the forecast frame that would have verified them.

    A cycle's forecast covers ``[start, forecast_end]``, where `forecast_end`
    extends past `end` by `time_control.forecast_extension` minutes. Without an
    extension the cycles tile time contiguously and each observation falls in
    exactly one window, so the lead is unambiguous. With an extension the windows
    overlap and two cycles offer a frame at the same instant; this function
    replays the tie-break in `ModelInterpolation._open_forecast_resolved`:

    - `prefer_extended_forecast` (the default): keep the **largest** lead, i.e.
      the earlier cycle's independent forecast that never saw the analysis at the
      cycle end. This is the correct background for O-B verification.
    - Otherwise: keep the **smallest** lead, the later cycle's analysis-driven
      forecast.

    Note that cycle boundaries are inclusive on both ends, so an observation
    exactly on a boundary falls in two windows even with no extension. The same
    tie-break resolves it, selecting the earlier cycle by default.

    This is a *reconstruction*, not a record of what the interpolation actually
    did. If the forecast files on disk changed since `validation
    interpolate-model` ran, the two can disagree. Persisting the lead alongside
    `model_forecast` would replace the body of this function.

    Args:
        times: Observation timestamps, any timezone flavour.
        cycles: The experiment's cycles, i.e. `experiment.cycles`.
        prefer_extended_forecast: `config.validation.prefer_extended_forecast`.

    Returns:
        A frame aligned to `times` with columns `cycle_index` (Int64, NA for times
        outside every forecast window), `lead` (timedelta64) and `lead_hours`
        (float).
    """

    obs_times = _to_naive_utc(times)
    n = len(obs_times)

    best_cycle = np.full(n, -1, dtype=np.int64)
    best_lead = np.full(n, np.nan, dtype="float64")  # seconds

    for cycle in cycles:
        start = pd.Timestamp(cycle.start)
        end = pd.Timestamp(cycle.forecast_end)
        if start.tz is not None:
            start = start.tz_convert("UTC").tz_localize(None)
        if end.tz is not None:
            end = end.tz_convert("UTC").tz_localize(None)

        in_window = (obs_times >= start) & (obs_times <= end)
        if not in_window.any():
            continue

        lead_s = (obs_times - start).dt.total_seconds().to_numpy()
        mask = in_window.to_numpy()

        unset = mask & ~np.isfinite(best_lead)
        if prefer_extended_forecast:
            better = mask & np.isfinite(best_lead) & (lead_s > best_lead)
        else:
            better = mask & np.isfinite(best_lead) & (lead_s < best_lead)

        take = unset | better
        best_cycle[take] = cycle.index
        best_lead[take] = lead_s[take]

    cycle_index = pd.array(best_cycle, dtype="Int64")
    cycle_index[best_cycle < 0] = pd.NA

    lead = pd.to_timedelta(pd.Series(best_lead, index=obs_times.index), unit="s")

    return pd.DataFrame(
        {
            "cycle_index": cycle_index,
            "lead": lead,
            "lead_hours": best_lead / 3600.0,
        },
        index=obs_times.index,
    )


def bin_by_lead(lead_hours, bin_hours: float) -> pd.Series:
    """
    Bin lead times, labelling each by its left edge in hours.

    Args:
        lead_hours: Lead times in hours.
        bin_hours: Bin width in hours. Must be positive.

    Returns:
        The left edge of each observation's bin, as a float Series. NaN leads stay
        NaN.
    """

    if bin_hours <= 0:
        raise ValueError(f"bin_hours must be positive, got {bin_hours}")

    lead_hours = pd.Series(lead_hours, dtype="float64")
    return np.floor(lead_hours / bin_hours) * bin_hours
