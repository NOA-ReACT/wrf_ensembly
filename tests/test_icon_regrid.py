"""Tests for the ICON to lat/lon interpolation (`wrf_ensembly.icon.regrid`)."""

import numpy as np
import pytest
import xarray as xr

from wrf_ensembly.icon.regrid import (
    METHODS,
    IconGrid,
    LatLonGrid,
    Regridder,
    locate_cells,
)


def grid_from_triangles(vertices: np.ndarray, triangles: np.ndarray) -> IconGrid:
    """IconGrid from unit vectors (nv, 3) and vertex indices (ncells, 3)."""
    corners = vertices[triangles]
    centres = corners.mean(axis=1)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    return IconGrid(
        uuid="test",
        clon=np.arctan2(centres[:, 1], centres[:, 0]),
        clat=np.arcsin(centres[:, 2]),
        vlon=np.arctan2(corners[:, :, 1], corners[:, :, 0]),
        vlat=np.arcsin(np.clip(corners[:, :, 2], -1, 1)),
    )


def limited_area_grid(n: int = 40, lon0=20.0, lat0=35.0, width=8.0) -> IconGrid:
    """A patch of triangles: a regular mesh of points with every square cut in two."""
    lon = np.radians(lon0 + np.linspace(-width / 2, width / 2, n))
    lat = np.radians(lat0 + np.linspace(-width / 2, width / 2, n))
    lon2d, lat2d = np.meshgrid(lon, lat)
    vertices = np.stack(
        [
            np.cos(lat2d) * np.cos(lon2d),
            np.cos(lat2d) * np.sin(lon2d),
            np.sin(lat2d),
        ],
        axis=-1,
    ).reshape(-1, 3)
    index = np.arange(n * n).reshape(n, n)
    a, b = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
    c, d = index[1:, :-1].ravel(), index[1:, 1:].ravel()
    triangles = np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])
    return grid_from_triangles(vertices, triangles)


def global_grid(bisections: int = 4) -> IconGrid:
    """An icosahedron with every triangle split in four `bisections` times."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=float,
    )  # fmt: skip
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    # Turn it so that two vertices sit on the poles, as in ICON
    north = vertices[5]
    axis = np.cross(north, [0, 0, 1.0])
    s, c = np.linalg.norm(axis), north[2]
    axis /= s
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    rotation = np.eye(3) + s * k + (1 - c) * (k @ k)
    vertices = vertices @ rotation.T
    triangles = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ]
    )  # fmt: skip
    for _ in range(bisections):
        vertex_list = list(vertices)
        midpoints: dict[tuple[int, int], int] = {}

        def midpoint(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key not in midpoints:
                m = vertex_list[i] + vertex_list[j]
                vertex_list.append(m / np.linalg.norm(m))
                midpoints[key] = len(vertex_list) - 1
            return midpoints[key]

        new = []
        for i, j, k_ in triangles:
            ij, jk, ki = midpoint(i, j), midpoint(j, k_), midpoint(k_, i)
            new += [[i, ij, ki], [j, jk, ij], [k_, ki, jk], [ij, jk, ki]]
        vertices, triangles = np.array(vertex_list), np.array(new)
    return grid_from_triangles(vertices, triangles)


def smooth_field(lon, lat):
    return np.sin(2 * lon) * np.cos(lat) + 0.5 * np.sin(lat)


def test_locate_cells_finds_the_containing_cell():
    grid = limited_area_grid(20)
    # Cell centres are inside their own cells
    found = locate_cells(grid, grid.clon, grid.clat)
    np.testing.assert_array_equal(found, np.arange(grid.ncells))
    # Far away points are in no cell
    outside = locate_cells(grid, np.radians([100.0, -50.0]), np.radians([0.0, 80.0]))
    assert (outside == -1).all()


@pytest.mark.parametrize("method", METHODS)
def test_constant_field_stays_constant(method):
    grid = limited_area_grid()
    regridder = Regridder.build(grid, method=method)
    out = regridder(np.full(grid.ncells, 3.5))
    assert regridder.valid.mean() > 0.9
    np.testing.assert_allclose(out[np.isfinite(out)], 3.5, rtol=1e-12)
    assert np.isfinite(out).sum() == regridder.valid.sum()


@pytest.mark.parametrize(
    "method,tolerance", [("nearest", 2e-2), ("linear", 2e-3), ("conservative", 1e-2)]
)
def test_smooth_field_is_reproduced(method, tolerance):
    grid = limited_area_grid(60)
    target = LatLonGrid.regular(17, 23, 32, 38, 0.1)
    regridder = Regridder.build(grid, target, method)
    out = regridder(smooth_field(grid.clon, grid.clat))
    lon, lat = target.points()
    truth = smooth_field(lon, lat).reshape(target.shape)
    assert regridder.valid.all()
    assert np.abs(out - truth).max() < tolerance


def test_outside_of_the_domain_is_filled():
    grid = limited_area_grid(20, width=4.0)
    target = LatLonGrid.regular(10, 30, 25, 45, 0.5)
    for method in METHODS:
        regridder = Regridder.build(grid, target, method)
        out = regridder(np.ones(grid.ncells), fill_value=-999.0)
        lon2d, lat2d = np.meshgrid(target.lon, target.lat)
        far = (np.abs(lon2d - 20) > 3) | (np.abs(lat2d - 35) > 3)
        near = (np.abs(lon2d - 20) < 1.5) & (np.abs(lat2d - 35) < 1.5)
        assert (out[far] == -999.0).all(), method
        np.testing.assert_allclose(out[near], 1.0, err_msg=method)


def test_conservative_keeps_the_integral():
    grid = limited_area_grid(60)
    # Coarser than the model grid, and fully inside it
    target = LatLonGrid.regular(17, 23, 32, 38, 0.5)
    regridder = Regridder.build(grid, target, "conservative")
    field = smooth_field(grid.clon, grid.clat) + 2.0

    out = regridder(field)
    lon_edges, lat_edges = np.radians(target.lon_edges), np.radians(target.lat_edges)
    target_area = np.diff(np.sin(lat_edges))[:, None] * np.diff(lon_edges)[None, :]
    regridded_integral = (out * target_area).sum()

    # Integral of the piecewise constant model field over the target box: sample it
    # on a very fine mesh
    fine = LatLonGrid.regular(17, 23, 32, 38, 0.01)
    sampled = Regridder.build(grid, fine, "nearest")(field)
    fine_area = (
        np.diff(np.sin(np.radians(fine.lat_edges)))[:, None]
        * np.diff(np.radians(fine.lon_edges))[None, :]
    )
    assert regridded_integral == pytest.approx((sampled * fine_area).sum(), rel=2e-4)


@pytest.mark.parametrize("method", METHODS)
def test_global_grid(method):
    grid = global_grid(4)
    assert grid.is_global()
    target = LatLonGrid.covering(grid, 2.0)
    assert target.shape == (90, 180)
    regridder = Regridder.build(grid, target, method)
    # The whole globe is covered, poles and dateline included
    assert regridder.valid.all()
    np.testing.assert_allclose(regridder(np.ones(grid.ncells)), 1.0, rtol=1e-9)

    out = regridder(smooth_field(grid.clon, grid.clat))
    lon, lat = target.points()
    truth = smooth_field(lon, lat).reshape(target.shape)
    assert np.abs(out - truth).max() < (0.05 if method == "linear" else 0.12)


def test_global_conservative_weights_add_up_to_the_sphere():
    grid = global_grid(3)
    target = LatLonGrid.covering(grid, 5.0)
    regridder = Regridder.build(grid, target, "conservative")
    lon_edges, lat_edges = np.radians(target.lon_edges), np.radians(target.lat_edges)
    target_area = (
        np.diff(np.sin(lat_edges))[:, None] * np.diff(lon_edges)[None, :]
    ).ravel()
    # Area that each model cell contributes, summed over all target cells
    per_cell = np.asarray((regridder.weights.T @ target_area)).ravel()
    assert per_cell.sum() == pytest.approx(4 * np.pi, rel=1e-3)
    # Every cell is used, and none is counted twice
    assert per_cell.min() > 0
    assert per_cell.max() < 4 * per_cell.mean()


def test_leading_dimensions_dtype_and_threads():
    grid = limited_area_grid(20)
    regridder = Regridder.build(grid, method="linear")
    data = np.random.default_rng(0).random((2, 3, grid.ncells)).astype(np.float32)
    out = regridder(data)
    assert out.shape == (2, 3, *regridder.target.shape)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out[1, 2], regridder(data[1, 2]))
    np.testing.assert_array_equal(out, regridder(data, threads=4))
    assert regridder(np.arange(grid.ncells)).dtype == np.float64
    with pytest.raises(ValueError):
        regridder(np.zeros(grid.ncells + 1))


def test_save_and_load(tmp_path):
    grid = limited_area_grid(20)
    regridder = Regridder.build(grid, method="conservative")
    regridder.save(tmp_path / "weights.npz")
    loaded = Regridder.load(tmp_path / "weights.npz")
    assert loaded.method == "conservative"
    assert loaded.target_matches(regridder.target)
    field = smooth_field(grid.clon, grid.clat)
    np.testing.assert_array_equal(loaded(field), regridder(field))


def test_regrid_dataset():
    grid = limited_area_grid(20)
    regridder = Regridder.build(grid, method="linear")
    ncells = grid.ncells
    ds = xr.Dataset(
        {
            "temp": (
                ("time", "height", "ncells"),
                np.ones((2, 3, ncells), dtype=np.float32),
                {"units": "K", "CDI_grid_type": "unstructured"},
            ),
            "pres_sfc": (("time", "ncells"), np.full((2, ncells), 1e5)),
            "clon_bnds": (("ncells", "vertices"), np.zeros((ncells, 3))),
            "scalar": ((), 1.0),
        },
        coords={
            "clon": ("ncells", grid.clon),
            "clat": ("ncells", grid.clat),
            "time": [0, 1],
        },
        attrs={"title": "ICON simulation"},
    )
    for dataset in (ds, ds.chunk({"time": 1})):
        out = regridder.regrid_dataset(dataset).compute()
        assert out["temp"].dims == ("time", "height", "lat", "lon")
        assert out["temp"].dtype == np.float32
        assert out["temp"].attrs == {"units": "K"}
        assert out["pres_sfc"].dims == ("time", "lat", "lon")
        assert "clon" not in out.variables and "clon_bnds" not in out.variables
        assert float(out["scalar"]) == 1.0
        assert out.attrs["title"] == "ICON simulation"
        assert out["lat"].attrs["units"] == "degrees_north"
        valid = regridder.valid.reshape(regridder.target.shape)
        np.testing.assert_allclose(out["temp"].values[0, 0][valid], 1.0, rtol=1e-6)
        assert np.isnan(out["temp"].values[0, 0][~valid]).all()
