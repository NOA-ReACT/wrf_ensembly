"""Tests for the MTG-REACT dust optical depth converter."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wrf_ensembly.observations import io as obs_io
from wrf_ensembly.observations import utils as obs_utils
from wrf_ensembly.observations.converters.mtg_react import (
    GRID_SHAPE,
    PREDICTION_COLUMN,
    QC_COLUMNS,
    convert_mtg_react,
    model_id_from_path,
)

MODEL_HASH = "ead14bc77dc0c39fc2229588b6922b7227fa6a98a8c76d3511901ae49b0dc5ce"


@pytest.fixture
def react_file(tmp_path: Path) -> Path:
    """A miniature prediction file, laid out like the real ones (float32 coords, one
    timestamp, an unused input feature) inside a hash-suffixed model directory."""

    model_dir = tmp_path / f"model_XGBoost_EARTHCARE_DOD_a355_{MODEL_HASH}"
    model_dir.mkdir()

    df = pd.DataFrame(
        {
            "time": pd.Timestamp("2026-03-20 12:00"),
            "latitude": np.array([4.0, 20.05, 42.0, 30.0], dtype=np.float32),
            "longitude": np.array([-18.0, 16.95, 60.0, 10.0], dtype=np.float32),
            "FCI_ir_105": np.float32(280.0),
            PREDICTION_COLUMN: np.array([0.1, 0.25, 0.5, np.nan], dtype=np.float32),
            "MTCM_quality_illumination": np.float32(3.0),
            "MTCM_quality_mtg_parameters": np.float32(1.0),
            "MTCM_quality_nwp_parameters": np.array([1, 2, np.nan, 1], dtype=np.float32),
            "MTCM_quality_overall_processing": np.float32(1.0),
        }
    )
    path = model_dir / "DOD_a355_prediction_2026-03-20_1200.parquet"
    df.to_parquet(path)
    return path


def test_convert(react_file: Path):
    df = convert_mtg_react(react_file, uncertainty=0.2)
    obs_io.validate_schema(df)

    # The NaN prediction is dropped
    assert len(df) == 3
    assert (df["instrument"] == "MTG_REACT").all()
    assert (df["quantity"] == "DOD_355nm").all()
    assert (df["time"] == pd.Timestamp("2026-03-20 12:00", tz="UTC")).all()
    assert (df["value_uncertainty"] == 0.2).all()
    np.testing.assert_allclose(df["value"], [0.1, 0.25, 0.5], rtol=1e-6)

    # Grid corners and an interior point land on the expected cells
    indices = [tuple(c["indices"]) for c in df["orig_coords"]]
    assert indices == [(0, 0), (321, 699), (GRID_SHAPE[0] - 1, GRID_SHAPE[1] - 1)]

    meta = df["metadata"].iloc[0]
    assert meta["model_id"] == MODEL_HASH[:8]
    assert set(QC_COLUMNS) <= meta.keys()
    assert meta["MTCM_quality_illumination"] == 3
    assert df["metadata"].iloc[2]["MTCM_quality_nwp_parameters"] is None


def test_round_trip(react_file: Path, tmp_path: Path):
    out = tmp_path / "out.parquet"
    obs_io.write_obs(convert_mtg_react(react_file), out)

    ds = obs_utils.reconstruct_array(obs_io.read_obs(out), trim_all_nan_slices=False)
    assert ds["value"].shape == GRID_SHAPE
    assert int(ds["value"].notnull().sum()) == 3


def test_outside_grid(react_file: Path):
    df = pd.read_parquet(react_file)
    df.loc[0, "latitude"] = 50.0
    df.to_parquet(react_file)

    with pytest.raises(ValueError, match="outside the expected grid"):
        convert_mtg_react(react_file)


def test_model_id_without_hash_dir(tmp_path: Path):
    assert model_id_from_path(tmp_path / "file.parquet") is None
