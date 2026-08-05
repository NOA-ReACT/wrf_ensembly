"""Tests for the observation plotting helpers."""

import numpy as np
import xarray as xr

from wrf_ensembly.observations.plotting import _fill_grid_gaps


def _grid(y_size: int = 5, x_size: int = 6) -> xr.DataArray:
    lat, _ = np.meshgrid(
        np.linspace(30.0, 34.0, y_size), np.linspace(20.0, 25.0, x_size), indexing="ij"
    )
    return xr.DataArray(lat, dims=("y", "x"))


def test_fills_interior_gaps_exactly():
    """Geolocation is linear in the grid indices, so the fill should be exact."""

    full = _grid()
    holed = full.copy()
    holed[2, 3] = np.nan
    holed[1, 1] = np.nan

    filled = _fill_grid_gaps(holed)

    assert np.isfinite(filled).all()
    assert np.allclose(filled, full.to_numpy())


def test_fills_edges_by_extrapolation():
    full = _grid()
    holed = full.copy()
    holed[0, :] = np.nan
    holed[:, -1] = np.nan

    filled = _fill_grid_gaps(holed)

    assert np.isfinite(filled).all()
    assert np.allclose(filled, full.to_numpy())


def test_sparse_grid_is_fully_filled():
    """The case that matters: a mostly-empty swath, as GRASP granules are."""

    rng = np.random.default_rng(0)
    full = _grid(20, 30)
    holed = full.copy()
    holed.values = np.where(rng.random(full.shape) > 0.85, full.to_numpy(), np.nan)

    filled = _fill_grid_gaps(holed)

    assert np.isfinite(filled).all()
    observed = np.isfinite(holed.to_numpy())
    assert np.allclose(filled[observed], full.to_numpy()[observed])
