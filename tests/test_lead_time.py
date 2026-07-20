import datetime as dt

import numpy as np
import pandas as pd
import pytest

from wrf_ensembly.cycling import CycleInformation
from wrf_ensembly.validation.lead_time import assign_forecast_lead, bin_by_lead

START = dt.datetime(2024, 8, 10, 0, 0, tzinfo=dt.timezone.utc)


def _cycles(n: int, duration_h: int = 6, extension_h: int = 0):
    """n back-to-back cycles, each optionally over-running into the next."""
    cycles = []
    for i in range(n):
        start = START + dt.timedelta(hours=i * duration_h)
        end = start + dt.timedelta(hours=duration_h)
        cycles.append(
            CycleInformation(
                start=start,
                end=end,
                cycle_offset=start - START,
                index=i,
                output_interval=60,
                forecast_end=end + dt.timedelta(hours=extension_h),
            )
        )
    return cycles


def test_no_extension_leads_are_exact():
    """Contiguous cycles: each time belongs to one cycle at an exact lead."""
    cycles = _cycles(3)
    times = pd.Series(
        [
            START + dt.timedelta(hours=1),  # cycle 0, lead 1h
            START + dt.timedelta(hours=7),  # cycle 1, lead 1h
            START + dt.timedelta(hours=14),  # cycle 2, lead 2h
        ]
    )

    result = assign_forecast_lead(times, cycles)

    assert list(result["cycle_index"]) == [0, 1, 2]
    assert list(result["lead_hours"]) == [1.0, 1.0, 2.0]


def test_times_outside_every_window_are_na():
    cycles = _cycles(2)
    times = pd.Series(
        [
            START - dt.timedelta(hours=1),  # before the experiment
            START + dt.timedelta(hours=3),  # inside cycle 0
            START + dt.timedelta(hours=99),  # after the experiment
        ]
    )

    result = assign_forecast_lead(times, cycles)

    assert pd.isna(result["cycle_index"].iloc[0])
    assert result["cycle_index"].iloc[1] == 0
    assert pd.isna(result["cycle_index"].iloc[2])
    assert np.isnan(result["lead_hours"].iloc[0])
    assert np.isnan(result["lead_hours"].iloc[2])


def test_extension_prefers_longer_lead_by_default():
    """Overlapping windows: the earlier cycle's extended forecast wins."""
    cycles = _cycles(3, duration_h=6, extension_h=3)
    # 07:00 sits in cycle 0's extension (lead 7h) and cycle 1 proper (lead 1h)
    times = pd.Series([START + dt.timedelta(hours=7)])

    result = assign_forecast_lead(times, cycles, prefer_extended_forecast=True)

    assert result["cycle_index"].iloc[0] == 0
    assert result["lead_hours"].iloc[0] == 7.0


def test_extension_can_prefer_shorter_lead():
    """The analysis-driven frame from the later cycle, when asked for."""
    cycles = _cycles(3, duration_h=6, extension_h=3)
    times = pd.Series([START + dt.timedelta(hours=7)])

    result = assign_forecast_lead(times, cycles, prefer_extended_forecast=False)

    assert result["cycle_index"].iloc[0] == 1
    assert result["lead_hours"].iloc[0] == 1.0


def test_cycle_boundary_resolves_by_preference():
    """Boundaries are inclusive on both ends, so the tie-break decides."""
    cycles = _cycles(2)
    times = pd.Series([START + dt.timedelta(hours=6)])  # cycle 0 end == cycle 1 start

    extended = assign_forecast_lead(times, cycles, prefer_extended_forecast=True)
    assert extended["cycle_index"].iloc[0] == 0
    assert extended["lead_hours"].iloc[0] == 6.0

    driven = assign_forecast_lead(times, cycles, prefer_extended_forecast=False)
    assert driven["cycle_index"].iloc[0] == 1
    assert driven["lead_hours"].iloc[0] == 0.0


def test_naive_and_aware_inputs_agree():
    """The DuckDB readers give aware timestamps, read_obs_seq_nc gives naive ones."""
    cycles = _cycles(3)
    aware = pd.Series(pd.to_datetime([START + dt.timedelta(hours=7)], utc=True))
    naive = pd.Series(pd.to_datetime(["2024-08-10 07:00:00"]))

    from_aware = assign_forecast_lead(aware, cycles)
    from_naive = assign_forecast_lead(naive, cycles)

    assert from_aware["cycle_index"].iloc[0] == from_naive["cycle_index"].iloc[0]
    assert from_aware["lead_hours"].iloc[0] == from_naive["lead_hours"].iloc[0] == 1.0


def test_result_is_aligned_to_input_index():
    """A non-trivial index must survive, so callers can join the result back."""
    cycles = _cycles(2)
    times = pd.Series(
        [START + dt.timedelta(hours=1), START + dt.timedelta(hours=7)],
        index=[100, 200],
    )

    result = assign_forecast_lead(times, cycles)

    assert list(result.index) == [100, 200]
    assert list(result["cycle_index"]) == [0, 1]


def test_lead_column_is_a_timedelta():
    cycles = _cycles(2)
    times = pd.Series([START + dt.timedelta(hours=2, minutes=30)])

    result = assign_forecast_lead(times, cycles)

    assert result["lead"].iloc[0] == pd.Timedelta(hours=2, minutes=30)


def test_bin_by_lead():
    leads = pd.Series([0.0, 0.9, 1.0, 5.9, 6.0, np.nan])

    binned = bin_by_lead(leads, bin_hours=3.0)

    assert list(binned[:5]) == [0.0, 0.0, 0.0, 3.0, 6.0]
    assert np.isnan(binned.iloc[5])


def test_bin_by_lead_rejects_nonpositive_width():
    with pytest.raises(ValueError):
        bin_by_lead(pd.Series([1.0]), bin_hours=0.0)
