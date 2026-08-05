"""Tests for the GRASP synergy (OLCI + TROPOMI) observation converter."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from wrf_ensembly.observations import io as obs_io
from wrf_ensembly.observations import utils as obs_utils
from wrf_ensembly.observations.converters.grasp_synergy import (
    UNCERTAINTY_INTERCEPT,
    UNCERTAINTY_SLOPE,
    convert_grasp_synergy,
)
from wrf_ensembly.observations.definitions import (
    INSTRUMENT_REGISTRY,
    QUANTITY_REGISTRY,
)

# The full band list as it appears in the real files, mixed formatting included.
BANDS = [
    "340",
    "367",
    "380",
    "412.5",
    "416",
    "440",
    "442.5",
    "490.0",
    "494",
    "510.0",
    "560.0",
    "665.0",
    "670",
    "747",
    "753.0",
    "772",
    "865.0",
    "1020.0",
    "2313",
]

TIME_UNITS = "days since 2022-08-01 05:04:54.961128960"

Y_SIZE = 4
X_SIZE = 5

EXPECTED_QUANTITIES = {
    "AOD_440nm",
    "AOD_500nm",
    "AOD_550nm",
    "AOD_665nm",
    "AOD_870nm",
    "AOD_Fine_440nm",
    "AOD_Fine_500nm",
    "AOD_Fine_550nm",
    "AOD_Fine_665nm",
    "AOD_Fine_870nm",
    "AOD_Coarse_440nm",
    "AOD_Coarse_500nm",
    "AOD_Coarse_550nm",
    "AOD_Coarse_665nm",
    "AOD_Coarse_870nm",
}

# Pixels deliberately made unusable, in (y, x) form: a NaN retrieval and a negative one.
NAN_PIXEL = (0, 0)
NEGATIVE_PIXEL = (1, 2)


@pytest.fixture
def synergy_file(tmp_path: Path) -> Path:
    """A miniature file with the same variables/dims as a real synergy granule."""

    rng = np.random.default_rng(0)
    shape = (1, Y_SIZE, X_SIZE, len(BANDS))

    def aod(scale: float) -> np.ndarray:
        data = rng.uniform(0.05, 1.5, size=shape).astype("float32") * scale
        data[0, NAN_PIXEL[0], NAN_PIXEL[1], :] = np.nan
        data[0, NEGATIVE_PIXEL[0], NEGATIVE_PIXEL[1], :] = -0.5
        return data

    lat, lon = np.meshgrid(
        np.linspace(30.0, 33.0, Y_SIZE), np.linspace(20.0, 24.0, X_SIZE), indexing="ij"
    )

    # First two rows land, rest ocean
    land_percentage = np.zeros((1, Y_SIZE, X_SIZE), dtype="int8")
    land_percentage[0, :2, :] = 100

    qa_normal = np.ones((1, Y_SIZE, X_SIZE), dtype="float32")
    qa_normal[0, 0, 1] = np.nan  # missing QA must survive as a JSON null

    ds = xr.Dataset(
        {
            "aerosol_optical_depth_total": (("t", "y", "x", "band"), aod(1.0)),
            "aerosol_fine_mode_optical_depth": (("t", "y", "x", "band"), aod(0.6)),
            "aerosol_coarse_mode_optical_depth": (("t", "y", "x", "band"), aod(0.4)),
            "land_percentage": (("t", "y", "x"), land_percentage),
            "qa_flag_normal": (("t", "y", "x"), qa_normal),
            "qa_flag_extended": (
                ("t", "y", "x"),
                np.full((1, Y_SIZE, X_SIZE), 2.0, dtype="float32"),
            ),
            "latitude": (("y", "x"), lat.astype("float32")),
            "longitude": (("y", "x"), lon.astype("float32")),
        },
        coords={
            "t": ("t", [0.0]),
            "band": ("band", BANDS),
            "y": ("y", np.arange(Y_SIZE)),
            "x": ("x", np.arange(X_SIZE)),
        },
    )
    ds["t"].attrs = {"units": TIME_UNITS, "calendar": "proleptic_gregorian"}

    path = tmp_path / "GRASP_SYN_OLCIA+OLCIB+TROPOMI__20220801T050454.nc"
    ds.to_netcdf(path)
    return path


def test_maps_only_the_close_bands(synergy_file: Path):
    df = convert_grasp_synergy(synergy_file)

    assert df is not None
    assert set(df["quantity"]) == EXPECTED_QUANTITIES
    assert set(df["instrument"]) == {"GRASP_SYNERGY"}


def test_registries_know_the_emitted_names(synergy_file: Path):
    """Plotting does unguarded registry lookups, so anything we emit must be registered."""

    df = convert_grasp_synergy(synergy_file)

    assert df is not None
    assert set(df["instrument"]) <= set(INSTRUMENT_REGISTRY)
    assert set(df["quantity"]) <= set(QUANTITY_REGISTRY)


def test_time_comes_from_the_t_coordinate(synergy_file: Path):
    df = convert_grasp_synergy(synergy_file)

    assert df is not None
    assert df["time"].dt.tz is not None
    assert df["time"].nunique() == 1
    assert df["time"].iloc[0] == pd.Timestamp("2022-08-01 05:04:55", tz="UTC")


def test_land_flag_follows_land_percentage(synergy_file: Path):
    df = convert_grasp_synergy(synergy_file)

    assert df is not None
    rows = df[df["quantity"] == "AOD_550nm"]
    y = rows["orig_coords"].apply(lambda c: c["indices"][0])
    is_over_land = rows["metadata"].apply(lambda m: m["is_over_land"])

    assert set(is_over_land[y < 2]) == {1}
    assert set(is_over_land[y >= 2]) == {0}


def test_missing_qa_becomes_null(synergy_file: Path):
    df = convert_grasp_synergy(synergy_file)

    assert df is not None
    rows = df[df["quantity"] == "AOD_550nm"]
    at_pixel = rows["orig_coords"].apply(lambda c: tuple(c["indices"]) == (0, 1))

    assert rows[at_pixel]["metadata"].iloc[0]["qa_flag_normal"] is None
    assert set(rows[~at_pixel]["metadata"].apply(lambda m: m["qa_flag_normal"])) == {
        1.0
    }
    assert set(rows["metadata"].apply(lambda m: m["qa_flag_extended"])) == {2.0}


def test_invalid_pixels_are_dropped_by_default(synergy_file: Path):
    df = convert_grasp_synergy(synergy_file)

    assert df is not None
    assert (df["qc_flag"] == 0).all()
    assert df["value"].notna().all()
    assert (df["value"] >= 0).all()

    indices = set(df["orig_coords"].apply(lambda c: tuple(c["indices"])))
    assert NAN_PIXEL not in indices
    assert NEGATIVE_PIXEL not in indices

    n_good = Y_SIZE * X_SIZE - 2
    assert len(df) == n_good * len(EXPECTED_QUANTITIES)


def test_keep_invalid_pixels_emits_the_full_grid(synergy_file: Path):
    df = convert_grasp_synergy(synergy_file, keep_invalid_pixels=True)

    assert df is not None
    assert len(df) == Y_SIZE * X_SIZE * len(EXPECTED_QUANTITIES)

    flagged = df[df["qc_flag"] == 1]
    assert set(flagged["orig_coords"].apply(lambda c: tuple(c["indices"]))) == {
        NAN_PIXEL,
        NEGATIVE_PIXEL,
    }


def test_uncertainty_model(synergy_file: Path):
    df = convert_grasp_synergy(synergy_file)

    assert df is not None
    expected = UNCERTAINTY_INTERCEPT + UNCERTAINTY_SLOPE * df["value"]
    assert np.allclose(df["value_uncertainty"], expected)


def test_disable_options(synergy_file: Path):
    df = convert_grasp_synergy(
        synergy_file,
        disabled_bands=("440",),
        disable_fine_mode=True,
        disable_coarse_mode=True,
    )

    assert df is not None
    assert set(df["quantity"]) == {
        "AOD_500nm",
        "AOD_550nm",
        "AOD_665nm",
        "AOD_870nm",
    }


def test_roundtrips_through_the_schema(synergy_file: Path, tmp_path: Path):
    df = convert_grasp_synergy(synergy_file)

    assert df is not None
    assert list(df.columns) == obs_io.REQUIRED_COLUMNS

    out = tmp_path / "obs.parquet"
    obs_io.write_obs(df, out)
    obs_io.validate_schema(obs_io.read_obs(out))


def test_sparse_output_reconstructs_onto_the_full_grid(synergy_file: Path):
    """Dropping the empty pixels must not break reconstruction back to (y, x)."""

    df = convert_grasp_synergy(synergy_file)

    assert df is not None
    rows = df[df["quantity"] == "AOD_550nm"]
    ds = obs_utils.reconstruct_array(rows, trim_all_nan_slices=False)

    assert ds["value"].shape == (Y_SIZE, X_SIZE)
    assert np.isnan(ds["value"].values[NAN_PIXEL])
    assert np.isnan(ds["value"].values[NEGATIVE_PIXEL])

    # Every converted observation landed back on its original pixel
    for _, row in rows.iterrows():
        y, x = row["orig_coords"]["indices"]
        assert ds["value"].values[y, x] == pytest.approx(row["value"])
