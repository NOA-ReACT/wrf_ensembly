"""Tests for the observation file schema validation."""

import numpy as np
import pandas as pd
import pytest

from wrf_ensembly.observations import io as obs_io


def make_df(n: int = 3, **overrides) -> pd.DataFrame:
    """A minimal valid observation DataFrame."""

    df = pd.DataFrame(
        {
            "instrument": "TEST",
            "quantity": "AOD_550nm",
            "time": pd.Timestamp("2022-08-01", tz="UTC"),
            "longitude": np.linspace(20.0, 21.0, n),
            "latitude": np.linspace(30.0, 31.0, n),
            "z": 0.0,
            "z_type": "columnar",
            "value": np.linspace(0.1, 0.5, n),
            "value_uncertainty": 0.05,
            "qc_flag": 0,
            "orig_filename": "test.nc",
            "metadata": pd.NA,
        }
    )
    df["orig_coords"] = [
        {"indices": (0, i), "shape": (1, n), "names": ("y", "x")} for i in range(n)
    ]
    df = df[obs_io.REQUIRED_COLUMNS]

    for column, value in overrides.items():
        df[column] = value
    return df


def test_accepts_a_valid_frame():
    obs_io.validate_schema(make_df())


def test_accepts_numpy_typed_orig_coords():
    """AERONET builds orig_coords out of numpy arrays rather than tuples."""

    df = make_df(2)
    df["orig_coords"] = [
        {
            "indices": np.array((i,), dtype=int),
            "shape": np.array((2,), dtype=int),
            "names": np.array(("row",), dtype=object),
        }
        for i in range(2)
    ]
    obs_io.validate_schema(df)


def test_rejects_missing_columns():
    df = make_df().drop(columns=["value_uncertainty", "z_type"])

    with pytest.raises(ValueError, match="Missing columns"):
        obs_io.validate_schema(df)


def test_rejects_bad_z_type_and_names_the_row():
    df = make_df(4)
    df.loc[2, "z_type"] = "stratospheric"

    with pytest.raises(ValueError, match="Invalid z_type encountered at row 2"):
        obs_io.validate_schema(df)


def test_reports_the_original_index_label():
    """The frame need not be RangeIndexed; the error should name the real label."""

    df = make_df(3)
    df.index = ["a", "b", "c"]
    df.loc["b", "z_type"] = "nope"

    with pytest.raises(ValueError, match="at row b"):
        obs_io.validate_schema(df)


def test_rejects_non_dict_orig_coords():
    df = make_df(3)
    df["orig_coords"] = [{"indices": (0,), "shape": (3,), "names": ("x",)}, None, None]

    with pytest.raises(ValueError, match="orig_coords must be a dictionary at row 1"):
        obs_io.validate_schema(df)


def test_rejects_missing_orig_coords_keys():
    df = make_df(2)
    df["orig_coords"] = [
        {"indices": (0,), "shape": (2,), "names": ("x",)},
        {"indices": (1,), "shape": (2,)},
    ]

    with pytest.raises(
        ValueError, match="must contain 'indices', 'shape', and 'names'"
    ):
        obs_io.validate_schema(df)


def test_rejects_mismatched_orig_coords_lengths():
    df = make_df(2)
    df["orig_coords"] = [
        {"indices": (0,), "shape": (2,), "names": ("x",)},
        {"indices": (1, 0), "shape": (2,), "names": ("x",)},
    ]

    with pytest.raises(ValueError, match="must have the same length at row 1"):
        obs_io.validate_schema(df)


@pytest.mark.parametrize(
    ("key", "bad", "message"),
    [
        ("indices", (0.5,), "'indices' must be integers"),
        ("shape", ("2",), "'shape' must be integers"),
        ("names", (7,), "'names' must be strings"),
    ],
)
def test_rejects_wrongly_typed_orig_coords_entries(key, bad, message):
    df = make_df(2)
    coords = [{"indices": (i,), "shape": (2,), "names": ("x",)} for i in range(2)]
    coords[1][key] = bad
    df["orig_coords"] = coords

    with pytest.raises(ValueError, match=message):
        obs_io.validate_schema(df)


def test_write_obs_validates_before_writing(tmp_path):
    df = make_df(3)
    df.loc[1, "z_type"] = "bogus"
    out = tmp_path / "obs.parquet"

    with pytest.raises(ValueError, match="Invalid z_type"):
        obs_io.write_obs(df, out)
    assert not out.exists()


def test_roundtrip(tmp_path):
    df = make_df(5)
    out = tmp_path / "obs.parquet"

    obs_io.write_obs(df, out)
    back = obs_io.read_obs(out)

    assert len(back) == len(df)
    assert list(back.columns) == obs_io.REQUIRED_COLUMNS
