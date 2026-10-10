from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd

from wrf_ensembly.config import ObservationsConfig
from wrf_ensembly.cycling import CycleInformation
from wrf_ensembly.experiment.observations import ExperimentObservations
from wrf_ensembly.observations import io


def cycle(index: int, start_hour: int) -> CycleInformation:
    start = datetime(2026, 3, 26, start_hour, tzinfo=timezone.utc)
    end = start + timedelta(hours=3)
    return CycleInformation(
        start=start, end=end, cycle_offset=timedelta(0), index=index,
        output_interval=60, forecast_end=end,
    )


def test_counts_the_window_past_the_cycle_end_and_only_good_qc(tmp_path):
    times = ["12:00"] * 3 + ["12:30"]
    qc = [0, -1, -1, 0]  # two thinning hold-outs
    df = pd.DataFrame(
        {
            "instrument": "MTG_REACT",
            "quantity": "DOD_355nm",
            "time": pd.to_datetime([f"2026-03-26T{t}" for t in times], utc=True),
            "longitude": 0.0, "latitude": 30.0, "x": 0.0, "y": 0.0, "z": 0.0,
            "z_type": "columnar",
            "value": 0.2, "value_uncertainty": 0.1,
            "qc_flag": qc,
            "orig_coords": [
                {"indices": [i], "shape": [4], "names": ["pixel"]} for i in range(4)
            ],
            "orig_filename": "slot.parquet",
            "metadata": [{"model_id": "test"} for _ in qc],
        }
    )
    path = tmp_path / "slot.parquet"
    io.write_obs(df, path)
    cfg = SimpleNamespace(
        observations=ObservationsConfig(),
        assimilation=SimpleNamespace(half_window_length_minutes=60),
    )
    obs = ExperimentObservations(
        cfg, [], SimpleNamespace(obs_db=tmp_path / "observations.duckdb")
    )
    obs.add_observation_file(path)

    summary = obs.get_cycle_summary([cycle(0, 9), cycle(1, 12)]).set_index("cycle_index")

    # Cycle 0 (09-12 UTC, window 11-13 UTC): the 12:30 obs is after its end but in
    # its window, and the hold-outs are not assimilated
    assert summary.loc[0, "total"] == 3
    assert summary.loc[0, "to_assimilate"] == 2
    # Cycle 1 (12-15 UTC, window 14-16 UTC)
    assert summary.loc[1, "total"] == 4
    assert summary.loc[1, "to_assimilate"] == 0
