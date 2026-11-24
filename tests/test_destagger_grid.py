"""Only the model grid is destaggered in postprocessing, not every `_stag` dimension"""

import numpy as np
import xarray as xr

from wrf_ensembly.processors import destagger_grid


def dataset() -> xr.Dataset:
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {
            "U": (("t", "z", "y", "x_stag"), rng.random((2, 3, 4, 6))),
            "W": (("t", "z_stag", "y", "x"), rng.random((2, 4, 4, 5))),
            "T": (("t", "z", "y", "x"), rng.random((2, 3, 4, 5))),
            "SMOIS": (("t", "soil_layers_stag", "y", "x"), rng.random((2, 4, 4, 5))),
            "SED_FLUX": (
                ("t", "bio_emissions_dimension_stag", "y", "x"),
                rng.random((2, 41, 4, 5)),
            ),
        }
    )


def test_grid_dimensions_are_destaggered():
    ds = dataset()
    out = destagger_grid(ds)

    assert out["U"].dims == ("t", "z", "y", "x")
    assert out["W"].dims == ("t", "z", "y", "x")
    np.testing.assert_allclose(
        out["U"].values, 0.5 * (ds["U"].values[..., :-1] + ds["U"].values[..., 1:])
    )
    np.testing.assert_array_equal(out["T"].values, ds["T"].values)


def test_species_and_soil_dimensions_are_left_alone():
    ds = dataset()
    out = destagger_grid(ds)

    for name in ("SMOIS", "SED_FLUX"):
        assert out[name].dims == ds[name].dims
        np.testing.assert_array_equal(out[name].values, ds[name].values)


def test_dataset_without_staggered_grid_dimensions():
    ds = dataset()[["T", "SED_FLUX"]]
    out = destagger_grid(ds)

    assert out["SED_FLUX"].sizes["bio_emissions_dimension_stag"] == 41
