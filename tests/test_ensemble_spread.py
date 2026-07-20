import datetime as dt
from types import SimpleNamespace

import pandas as pd

from wrf_ensembly.cycling import CycleInformation
from wrf_ensembly.validation.ensemble_spread import EnsembleSpreadAnalysis

START = dt.datetime(2024, 8, 10, 0, 0, tzinfo=dt.timezone.utc)


def _cycles(n: int, duration_h: int = 6, extension_h: int = 0):
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


def _resolve(cycles, series, prefer_extended: bool) -> pd.DataFrame:
    """Call `_resolve_overlaps` with just enough experiment state stubbed in."""
    stub = SimpleNamespace(
        exp=SimpleNamespace(
            cycles=cycles,
            cfg=SimpleNamespace(
                validation=SimpleNamespace(prefer_extended_forecast=prefer_extended)
            ),
        )
    )
    return EnsembleSpreadAnalysis._resolve_overlaps(stub, series)


def _overlapping_series(cycles):
    """Two cycles both producing a frame at 06:00, with distinguishable values."""
    t_overlap = pd.Timestamp("2024-08-10 06:00:00")
    return pd.DataFrame(
        {
            "AOD_mean": [0.1, 0.2, 0.9],
            "AOD_sd": [0.01, 0.02, 0.09],
            "cycle_index": [0, 0, 1],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-08-10 05:00:00"),
                t_overlap,  # from cycle 0's extension
                t_overlap,  # from cycle 1 proper
            ],
            name="time",
        ),
    )


def test_no_overlap_passes_through_sorted():
    cycles = _cycles(2)
    series = pd.DataFrame(
        {"AOD_mean": [0.2, 0.1], "AOD_sd": [0.02, 0.01], "cycle_index": [1, 0]},
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-08-10 07:00:00"),
                pd.Timestamp("2024-08-10 01:00:00"),
            ],
            name="time",
        ),
    )

    result = _resolve(cycles, series, prefer_extended=True)

    assert list(result.index) == sorted(result.index)
    assert "cycle_index" not in result.columns
    assert len(result) == 2


def test_overlap_prefers_extended_forecast():
    """The earlier cycle's longer-lead frame wins by default."""
    cycles = _cycles(2, duration_h=6, extension_h=3)

    result = _resolve(cycles, _overlapping_series(cycles), prefer_extended=True)

    assert len(result) == 2
    # 0.2 is cycle 0's frame at the overlapping timestamp
    assert result.loc[pd.Timestamp("2024-08-10 06:00:00"), "AOD_mean"] == 0.2


def test_overlap_can_prefer_analysis_driven():
    """The later cycle's shorter-lead frame wins when asked for."""
    cycles = _cycles(2, duration_h=6, extension_h=3)

    result = _resolve(cycles, _overlapping_series(cycles), prefer_extended=False)

    assert len(result) == 2
    # 0.9 is cycle 1's frame at the overlapping timestamp
    assert result.loc[pd.Timestamp("2024-08-10 06:00:00"), "AOD_mean"] == 0.9


def test_overlap_resolution_drops_no_timestamps():
    """Every distinct timestamp survives, whichever cycle is preferred."""
    cycles = _cycles(2, duration_h=6, extension_h=3)
    series = _overlapping_series(cycles)

    for prefer in (True, False):
        result = _resolve(cycles, series, prefer_extended=prefer)
        assert set(result.index) == set(series.index)
