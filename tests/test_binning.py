import numpy as np
import pandas as pd
import pytest

from wrf_ensembly.config import ThinningConfig
from wrf_ensembly.observations.io import QC_VALIDATION_HOLDOUT
from wrf_ensembly.superobs import grid_bin, spatial_thin


def gridded_obs(ny: int, nx: int, filename: str = "slot_1200.parquet") -> pd.DataFrame:
    """A regular (y, x) grid of good-QC observations from one file, like MTG-REACT."""
    iy, ix = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    iy, ix = iy.ravel(), ix.ravel()
    return pd.DataFrame(
        {
            "instrument": "MTG_REACT",
            "quantity": "DOD_355nm",
            "time": pd.Timestamp("2026-04-08T12:00", tz="UTC"),
            "longitude": -18.0 + 0.05 * ix,
            "latitude": 4.0 + 0.05 * iy,
            "x": 5000.0 * ix,
            "y": 5000.0 * iy,
            "z": 0.0,
            "z_type": "columnar",
            "value": 0.1 + 0.001 * (iy + ix),
            "value_uncertainty": 0.1,
            "qc_flag": 0,
            "orig_coords": [
                {"indices": [a, b], "shape": [ny, nx], "names": ["y", "x"]}
                for a, b in zip(iy, ix)
            ],
            "orig_filename": filename,
            "metadata": [{"model_id": "test"} for _ in iy],
        }
    )


def kept_indices(df: pd.DataFrame) -> set[tuple[int, int]]:
    good = df[df["qc_flag"] == 0]
    return {tuple(int(i) for i in oc["indices"]) for oc in good["orig_coords"]}


def test_keeps_a_regular_lattice_of_superobs():
    superobs = grid_bin(gridded_obs(20, 30), {"y": 5, "x": 5}, {}, False)
    assert len(superobs) == 4 * 6

    thinned = spatial_thin(superobs, {"y_bin": 2, "x_bin": 3})

    assert len(thinned) == len(superobs)
    assert kept_indices(thinned) == {(0, 0), (0, 3), (2, 0), (2, 3)}
    assert (thinned["qc_flag"] == QC_VALIDATION_HOLDOUT).sum() == 24 - 4


def test_offsets_shift_the_lattice():
    thinned = spatial_thin(gridded_obs(6, 6), {"y": 3, "x": 3}, {"x": 1})
    assert kept_indices(thinned) == {(0, 1), (0, 4), (3, 1), (3, 4)}


def test_bad_qc_is_left_alone():
    obs = gridded_obs(4, 4)
    obs.loc[obs.index[1], "qc_flag"] = 1  # (0, 1), not on the lattice

    thinned = spatial_thin(obs, {"y": 2, "x": 2})

    assert thinned.loc[obs.index[1], "qc_flag"] == 1
    assert kept_indices(thinned) == {(0, 0), (0, 2), (2, 0), (2, 2)}


def test_lattice_restarts_in_every_file():
    obs = pd.concat([gridded_obs(2, 2, "a.parquet"), gridded_obs(2, 2, "b.parquet")])
    thinned = spatial_thin(obs, {"y": 2, "x": 2})
    good = thinned[thinned["qc_flag"] == 0]
    assert sorted(good["orig_filename"]) == ["a.parquet", "b.parquet"]


def test_does_not_modify_the_input():
    obs = gridded_obs(4, 4)
    spatial_thin(obs, {"y": 2})
    assert (obs["qc_flag"] == 0).all()


def test_unknown_dimension_names_the_available_ones():
    with pytest.raises(ValueError, match=r"'y_bin'.*\['y', 'x'\]"):
        spatial_thin(gridded_obs(2, 2), {"y_bin": 2})


def test_config_validation():
    ThinningConfig(hoz_strides={"y_bin": 4, "x_bin": 4}, hoz_offsets={"x_bin": 3})
    with pytest.raises(ValueError, match="hoz_strides.y_bin"):
        ThinningConfig(hoz_strides={"y_bin": 0})
    with pytest.raises(ValueError, match="no matching hoz_strides"):
        ThinningConfig(hoz_offsets={"x_bin": 1})
    with pytest.raises(ValueError, match=r"in \[0, 4\)"):
        ThinningConfig(hoz_strides={"x_bin": 4}, hoz_offsets={"x_bin": 4})
    with pytest.raises(ValueError, match="keep_every_n"):
        ThinningConfig(keep_every_n=0)


def test_add_thins_superobs(tmp_path):
    """Superobbing builds new rows, thinning must still find their pair (regression)"""
    from types import SimpleNamespace

    import duckdb

    from wrf_ensembly.config import ObservationsConfig, SuperObsConfig
    from wrf_ensembly.experiment.observations import ExperimentObservations
    from wrf_ensembly.observations import io

    path = tmp_path / "slot_1200.parquet"
    io.write_obs(gridded_obs(20, 30), path)
    cfg = SimpleNamespace(
        observations=ObservationsConfig(
            superobs={"MTG_REACT.DOD_355nm": SuperObsConfig({"y": 5, "x": 5}, {}, False)},
            thinning={"MTG_REACT.DOD_355nm": ThinningConfig(hoz_strides={"y_bin": 2, "x_bin": 3})},
        )
    )
    paths = SimpleNamespace(obs_db=tmp_path / "observations.duckdb")

    assert ExperimentObservations(cfg, [], paths).add_observation_file(path) == 24

    con = duckdb.connect(str(paths.obs_db), read_only=True)
    kept = con.execute(
        "SELECT orig_coords.indices[1], orig_coords.indices[2] FROM observations "
        "WHERE qc_flag = 0 ORDER BY 1, 2"
    ).fetchall()
    assert kept == [(0, 0), (0, 3), (2, 0), (2, 3)]


def test_superobs_ignore_observations_without_a_value():
    """Trimming sets out-of-domain pixels to NaN; they must not shape the superob"""
    obs = gridded_obs(5, 10)  # two 5x5 bins
    first_bin = obs["orig_coords"].map(lambda oc: oc["indices"][1] < 5)
    outside = first_bin & obs["orig_coords"].map(lambda oc: oc["indices"][0] >= 2)
    obs.loc[outside, ["value", "value_uncertainty"]] = np.nan
    obs.loc[~first_bin, ["value", "value_uncertainty"]] = np.nan  # second bin all NaN

    superobs = grid_bin(obs, {"y": 5, "x": 5}, {}, False)

    assert len(superobs) == 1  # the all-NaN bin makes no superob
    so = superobs.iloc[0]
    valid = obs[first_bin & ~outside]
    assert so["metadata"]["superob"]["n_contributing"] == len(valid) == 10
    assert so["latitude"] == pytest.approx(valid["latitude"].mean())
    assert so["value"] == pytest.approx(valid["value"].mean())
    assert np.isfinite(so["value_uncertainty"])


def test_single_valid_observation_keeps_an_error():
    obs = gridded_obs(5, 5)
    obs.loc[obs.index[1:], ["value", "value_uncertainty"]] = np.nan
    so = grid_bin(obs, {"y": 5, "x": 5}, {}, False).iloc[0]
    assert so["metadata"]["superob"]["n_contributing"] == 1
    assert np.isfinite(so["value_uncertainty"])


@pytest.mark.parametrize("n_valid, qc", [(14, 1), (15, 0), (25, 0)])
def test_valid_fraction_flags_partly_filled_bins(n_valid, qc):
    obs = gridded_obs(5, 5)
    obs.loc[obs.index[n_valid:], ["value", "value_uncertainty"]] = np.nan
    so = grid_bin(obs, {"y": 5, "x": 5}, {}, False, valid_fraction=0.6).iloc[0]
    assert so["qc_flag"] == qc
    assert so["metadata"]["superob"]["n_contributing"] == n_valid


def test_valid_fraction_config_validation():
    from wrf_ensembly.config import SuperObsConfig

    assert SuperObsConfig({"y": 5}, {}, valid_fraction=0.6).valid_fraction == 0.6
    with pytest.raises(ValueError, match="valid_fraction"):
        SuperObsConfig({"y": 5}, {}, valid_fraction=1.5)
