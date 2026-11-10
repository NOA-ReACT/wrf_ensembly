"""
Horizontal interpolation from the ICON triangular grid to a regular lat/lon grid.

ICON stores cell fields as one value per triangle (dimension `ncells`), which most
tools cannot plot or compare with anything. Here we compute, once per pair of
(ICON grid, target grid), a sparse matrix of interpolation weights. Applying it to a
field is then one sparse matrix-vector product per level and time step, which is much
faster than computing the weights again for every file.

Three methods:

- `nearest`: each target point takes the value of the triangle that contains it.
- `linear`: barycentric interpolation between the three nearest cell centres
  (Delaunay triangulation of the cell centres). Smooth, the default.
- `conservative`: area-weighted mean of the triangles that overlap each target cell
  (first order). Use it for fluxes/precipitation, or when the target grid is coarser
  than the model grid.

Target points outside the model domain (limited-area grids) are set to the fill value.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import netCDF4
import numpy as np
import scipy.sparse as sp
import xarray as xr
from scipy.spatial import Delaunay, cKDTree

Method = Literal["nearest", "linear", "conservative"]
METHODS: tuple[str, ...] = ("nearest", "linear", "conservative")

# Bump when the weights of an existing cache file would no longer be what the code
# computes, so that old caches are ignored.
CACHE_VERSION = 1


def _xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Unit vectors for lon/lat in radians, shape (..., 3)."""
    coslat = np.cos(lat)
    return np.stack([coslat * np.cos(lon), coslat * np.sin(lon), np.sin(lat)], axis=-1)


def _wrap(dlon: np.ndarray) -> np.ndarray:
    """Wrap a longitude difference (radians) into [-pi, pi)."""
    return (dlon + np.pi) % (2 * np.pi) - np.pi


@dataclass
class IconGrid:
    """The parts of an ICON grid file that the interpolation needs (all in radians)."""

    uuid: str
    clon: np.ndarray  # (ncells,)
    clat: np.ndarray  # (ncells,)
    vlon: np.ndarray  # (ncells, 3) longitude of the three vertices of each cell
    vlat: np.ndarray  # (ncells, 3)

    @classmethod
    def from_file(cls, path: Path | str) -> "IconGrid":
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            clon = np.asarray(ds["clon"][:], dtype=np.float64)
            clat = np.asarray(ds["clat"][:], dtype=np.float64)
            if "clon_vertices" in ds.variables:
                vlon = np.asarray(ds["clon_vertices"][:], dtype=np.float64)
                vlat = np.asarray(ds["clat_vertices"][:], dtype=np.float64)
            else:
                # (3, ncells), 1-based
                voc = np.asarray(ds["vertex_of_cell"][:]).T - 1
                vlon = np.asarray(ds["vlon"][:], dtype=np.float64)[voc]
                vlat = np.asarray(ds["vlat"][:], dtype=np.float64)[voc]
            uuid = str(getattr(ds, "uuidOfHGrid", ""))
        return cls(uuid=uuid, clon=clon, clat=clat, vlon=vlon, vlat=vlat)

    @property
    def ncells(self) -> int:
        return self.clon.size

    def cell_areas(self) -> np.ndarray:
        """Area of each cell on the unit sphere (flat triangle between the vertices)."""
        v = _xyz(self.vlon, self.vlat)
        return 0.5 * np.linalg.norm(
            np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=-1
        )

    def resolution_deg(self) -> float:
        """Typical distance between neighbouring cell centres, in degrees."""
        return float(np.degrees(np.sqrt(self.cell_areas().mean())))

    def is_global(self) -> bool:
        centre = _xyz(self.clon, self.clat).mean(axis=0)
        return bool(np.linalg.norm(centre) < 0.3)


@dataclass
class LatLonGrid:
    """A regular lat/lon grid given by its cell centres in degrees, both ascending."""

    lon: np.ndarray
    lat: np.ndarray

    def __post_init__(self):
        self.lon = np.asarray(self.lon, dtype=np.float64)
        self.lat = np.asarray(self.lat, dtype=np.float64)
        if self.lon.size < 2 or self.lat.size < 2:
            raise ValueError("The target grid needs at least 2 points per direction")
        if np.any(np.diff(self.lon) <= 0) or np.any(np.diff(self.lat) <= 0):
            raise ValueError("Target grid lon and lat must be strictly ascending")

    @classmethod
    def regular(
        cls,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        dlon: float,
        dlat: float | None = None,
    ) -> "LatLonGrid":
        """Grid whose cells tile the box [lon_min, lon_max] x [lat_min, lat_max]."""
        dlat = dlon if dlat is None else dlat
        nlon = max(2, int(round((lon_max - lon_min) / dlon)))
        nlat = max(2, int(round((lat_max - lat_min) / dlat)))
        return cls(
            lon=lon_min + dlon * (np.arange(nlon) + 0.5),
            lat=lat_min + dlat * (np.arange(nlat) + 0.5),
        )

    @classmethod
    def covering(cls, grid: IconGrid, resolution: float | None = None) -> "LatLonGrid":
        """
        The smallest lat/lon box that contains the whole ICON grid.

        Args:
            resolution: Grid spacing in degrees. Defaults to the spacing of the ICON
                grid, rounded to two significant digits.
        """
        if resolution is None:
            res = grid.resolution_deg()
            magnitude = 10 ** (np.floor(np.log10(res)) - 1)
            resolution = float(np.round(res / magnitude) * magnitude)

        if grid.is_global():
            return cls.regular(-180, 180, -90, 90, resolution)

        # Longitudes relative to the domain centre, so that a domain across the
        # dateline does not turn into a box around the whole globe
        centre = _xyz(grid.clon, grid.clat).mean(axis=0)
        lon0 = np.arctan2(centre[1], centre[0])
        rel = np.degrees(_wrap(grid.vlon - lon0))
        lon0 = np.degrees(lon0)
        lat = np.degrees(grid.vlat)

        def snap(value: float, fn) -> float:
            return float(fn(value / resolution) * resolution)

        return cls.regular(
            snap(lon0 + rel.min(), np.floor),
            snap(lon0 + rel.max(), np.ceil),
            max(-90.0, snap(lat.min(), np.floor)),
            min(90.0, snap(lat.max(), np.ceil)),
            resolution,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.lat.size, self.lon.size)

    @property
    def size(self) -> int:
        return self.lat.size * self.lon.size

    @staticmethod
    def _edges(centres: np.ndarray) -> np.ndarray:
        mid = 0.5 * (centres[1:] + centres[:-1])
        first = centres[0] - (mid[0] - centres[0])
        last = centres[-1] + (centres[-1] - mid[-1])
        return np.concatenate([[first], mid, [last]])

    @property
    def lon_edges(self) -> np.ndarray:
        return self._edges(self.lon)

    @property
    def lat_edges(self) -> np.ndarray:
        return np.clip(self._edges(self.lat), -90.0, 90.0)

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        """lon and lat (radians) of all target points, flattened in (lat, lon) order."""
        lon2d, lat2d = np.meshgrid(np.radians(self.lon), np.radians(self.lat))
        return lon2d.ravel(), lat2d.ravel()


def locate_cells(grid: IconGrid, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Find the ICON cell that contains each point.

    Args:
        lon, lat: Point coordinates in radians, 1-D.

    Returns:
        Cell index for each point, -1 where the point is in no cell.
    """
    centres = _xyz(grid.clon, grid.clat)
    vertices = _xyz(grid.vlon, grid.vlat)  # (ncells, 3, 3)
    points = _xyz(lon, lat)

    # A point is inside a spherical triangle when it is on the same side of the three
    # great circles through the edges. The normals of these circles:
    normals = np.cross(vertices, np.roll(vertices, -1, axis=1))  # (ncells, 3, 3)
    # Make them all point inwards, whatever the vertex order in the file
    orientation = np.sign(np.einsum("ij,ij->i", normals[:, 0], vertices[:, 2]))
    normals *= orientation[:, None, None]

    # The containing cell is almost always the one with the nearest centre, otherwise
    # one of its neighbours. No cell can contain a point farther from its centre than
    # its own vertices are.
    max_radius = np.linalg.norm(vertices - centres[:, None, :], axis=-1).max()
    n_candidates = min(6, grid.ncells)
    tree = cKDTree(centres)
    dist, candidates = tree.query(
        points, k=n_candidates, distance_upper_bound=max_radius * 1.01, workers=-1
    )
    dist = dist.reshape(points.shape[0], n_candidates)
    candidates = candidates.reshape(points.shape[0], n_candidates)

    result = np.full(points.shape[0], -1, dtype=np.int64)
    pending = np.flatnonzero(np.isfinite(dist[:, 0]))
    for k in range(n_candidates):
        if pending.size == 0:
            break
        pending = pending[np.isfinite(dist[pending, k])]
        cell = candidates[pending, k]
        side = np.einsum("nej,nj->ne", normals[cell], points[pending])
        inside = (side >= -1e-12).all(axis=1)
        result[pending[inside]] = cell[inside]
        pending = pending[~inside]
    return result


def _nearest_weights(grid: IconGrid, target: LatLonGrid):
    lon, lat = target.points()
    cell = locate_cells(grid, lon, lat)
    valid = cell >= 0
    rows = np.flatnonzero(valid)
    weights = sp.csr_matrix(
        (np.ones(rows.size), (rows, cell[valid])), shape=(target.size, grid.ncells)
    )
    return weights, valid


def _stereographic(points: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """
    Stereographic projection of unit vectors onto the plane tangent at `centre`.

    It maps circles to circles, so the Delaunay triangulation of the projected points
    is the Delaunay triangulation on the sphere.
    """
    centre = centre / np.linalg.norm(centre)
    helper = (
        np.array([0.0, 0.0, 1.0]) if abs(centre[2]) < 0.9 else np.array([1.0, 0, 0])
    )
    e1 = np.cross(helper, centre)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(centre, e1)
    scale = 2.0 / (1.0 + points @ centre)
    return np.stack([points @ e1 * scale, points @ e2 * scale], axis=-1)


def _linear_weights(grid: IconGrid, target: LatLonGrid):
    lon, lat = target.points()
    cell = locate_cells(grid, lon, lat)
    valid = cell >= 0
    centres = _xyz(grid.clon, grid.clat)
    points = _xyz(lon, lat)
    # Two cell centres that belong to the same Delaunay triangle are never farther
    # apart than this. Longer triangles are the slivers that close the outline of a
    # limited-area domain.
    max_edge = 2.5 * np.sqrt(grid.cell_areas())

    rows_out, cols_out, data_out = [], [], []
    done = np.zeros(target.size, dtype=bool)

    if grid.is_global():
        # One projection cannot show the whole sphere: triangulate each half with
        # some overlap, and use it for the target points in that half
        poles = [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]
        parts = [
            (pole, np.flatnonzero(centres @ pole > -0.5), valid & (points @ pole >= 0))
            for pole in poles
        ]
        parts[1] = (parts[1][0], parts[1][1], parts[1][2] & ~parts[0][2])
    else:
        parts = [(centres.mean(axis=0), np.arange(grid.ncells), valid)]

    for centre, cells, wanted in parts:
        rows = np.flatnonzero(wanted)
        if rows.size == 0 or cells.size < 3:
            continue
        tri = Delaunay(_stereographic(centres[cells], centre))
        xy = _stereographic(points[rows], centre)
        simplex = tri.find_simplex(xy)
        found = simplex >= 0
        rows, xy, simplex = rows[found], xy[found], simplex[found]

        transform = tri.transform[simplex]  # (n, 3, 2)
        b = np.einsum("nij,nj->ni", transform[:, :2], xy - transform[:, 2])
        bary = np.column_stack([b, 1.0 - b.sum(axis=1)])
        corners = cells[tri.simplices[simplex]]  # (n, 3)

        corner_xyz = centres[corners]
        edge_length = np.linalg.norm(
            corner_xyz - np.roll(corner_xyz, -1, axis=1), axis=-1
        ).max(axis=1)
        good = edge_length < max_edge[corners].max(axis=1)
        rows, bary, corners = rows[good], np.clip(bary[good], 0.0, 1.0), corners[good]
        bary /= bary.sum(axis=1, keepdims=True)

        rows_out.append(np.repeat(rows, 3))
        cols_out.append(corners.ravel())
        data_out.append(bary.ravel())
        done[rows] = True

    # Points inside a cell on the rim of the domain, but outside the triangulation of
    # the cell centres: take the value of that cell
    rest = np.flatnonzero(valid & ~done)
    rows_out.append(rest)
    cols_out.append(cell[rest])
    data_out.append(np.ones(rest.size))

    weights = sp.csr_matrix(
        (
            np.concatenate(data_out),
            (np.concatenate(rows_out), np.concatenate(cols_out)),
        ),
        shape=(target.size, grid.ncells),
    )
    return weights, valid


def _clip(
    poly: np.ndarray, count: np.ndarray, axis: int, value: np.ndarray, keep_above: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Clip many polygons at once against the line `coordinate[axis] = value`
    (Sutherland-Hodgman).

    Args:
        poly: (n, m, 2) vertices, padded; only the first `count` of each row are used.
        count: (n,) number of vertices of each polygon.
        value: (n,) position of the clip line for each polygon.
        keep_above: Keep the part with coordinate >= value, else the part <= value.
    """
    n, m, _ = poly.shape
    slot = np.arange(m)[None, :]
    used = slot < count[:, None]
    previous = np.where(slot == 0, count[:, None] - 1, slot - 1)
    previous = np.clip(previous, 0, m - 1)
    prev = np.take_along_axis(poly, previous[:, :, None], axis=1)

    sign = 1.0 if keep_above else -1.0
    dist_cur = sign * (poly[:, :, axis] - value[:, None])
    dist_prev = sign * (prev[:, :, axis] - value[:, None])
    in_cur = dist_cur >= 0
    in_prev = dist_prev >= 0

    # Where the edge prev -> cur crosses the line
    crosses = used & (in_cur != in_prev)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(crosses, dist_prev / (dist_prev - dist_cur), 0.0)
    crossing = prev + t[:, :, None] * (poly - prev)
    crossing[:, :, axis] = np.broadcast_to(value[:, None], (n, m))

    # Each edge gives its crossing point (if any), then its end point (if inside)
    out = np.empty((n, 2 * m, 2))
    out[:, 0::2] = crossing
    out[:, 1::2] = poly
    keep = np.empty((n, 2 * m), dtype=bool)
    keep[:, 0::2] = crosses
    keep[:, 1::2] = used & in_cur

    # Move the kept points to the front of each row, in order
    order = np.argsort(~keep, axis=1, kind="stable")[:, :m]
    new_count = keep.sum(axis=1)
    if new_count.max(initial=0) > m:
        raise RuntimeError("Polygon clipping ran out of vertex slots")
    return np.take_along_axis(out, order[:, :, None], axis=1), new_count


def _polygon_area(poly: np.ndarray, count: np.ndarray) -> np.ndarray:
    """
    Area on the unit sphere of polygons whose edges are straight lines in (lon, lat).

    By Green's theorem the area is the integral of -sin(lat) dlon around the outline,
    which has a closed form on each edge.
    """
    n, m, _ = poly.shape
    slot = np.arange(m)[None, :]
    used = slot < count[:, None]
    following = np.where(slot + 1 >= count[:, None], 0, slot + 1)
    nxt = np.take_along_axis(poly, following[:, :, None], axis=1)

    lon1, lat1 = poly[:, :, 0], poly[:, :, 1]
    lon2, lat2 = nxt[:, :, 0], nxt[:, :, 1]
    dlat = lat2 - lat1
    flat = np.abs(dlat) < 1e-13
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_sin = np.where(
            flat,
            np.sin(0.5 * (lat1 + lat2)),
            (np.cos(lat1) - np.cos(lat2)) / np.where(flat, 1.0, dlat),
        )
    # Subtracting a constant does not change a closed integral, but keeps the terms
    # small, which matters for cells that are tiny compared to the sphere
    reference = np.sin(poly[:, :1, 1])
    terms = np.where(used, (lon2 - lon1) * (mean_sin - reference), 0.0)
    return np.abs(terms.sum(axis=1))


def _conservative_weights(
    grid: IconGrid, target: LatLonGrid, min_coverage: float = 0.5
):
    lon_edges = np.radians(target.lon_edges)
    lat_edges = np.radians(target.lat_edges)
    nlat, nlon = target.shape

    # Cell outlines in (lon, lat), with the longitudes of each cell continuous and
    # the cell placed inside (or next to) the longitude range of the target grid.
    centre_lon = lon_edges[0] + (grid.clon - lon_edges[0]) % (2 * np.pi)
    vlon = centre_lon[:, None] + _wrap(grid.vlon - grid.clon[:, None])
    vlat = grid.vlat

    # Cell edges are great circles, which we treat as straight lines in (lon, lat).
    # That is very accurate for small cells away from the poles. Where it is not,
    # cut every edge into short pieces first.
    edge = np.linalg.norm(
        _xyz(vlon, vlat) - np.roll(_xyz(vlon, vlat), -1, axis=1), axis=-1
    ).max(axis=1)
    curved = edge * np.tan(np.minimum(np.abs(vlat).max(axis=1), 1.5)) > 2e-2
    at_pole = np.abs(vlat) > np.pi / 2 - 1e-9

    rows_out, cols_out, data_out = [], [], []
    for pieces, cells in ((1, np.flatnonzero(~curved)), (16, np.flatnonzero(curved))):
        if cells.size == 0:
            continue
        poly, count = _cell_outlines(vlon[cells], vlat[cells], at_pole[cells], pieces)
        rows, cols, data = _overlap_areas(poly, count, cells, lon_edges, lat_edges)
        rows_out.append(rows)
        cols_out.append(cols)
        data_out.append(data)

    weights = sp.csr_matrix(
        (
            np.concatenate(data_out),
            (np.concatenate(rows_out), np.concatenate(cols_out)),
        ),
        shape=(target.size, grid.ncells),
    )
    weights.sum_duplicates()

    covered = np.asarray(weights.sum(axis=1)).ravel()
    target_area = (
        np.diff(np.sin(lat_edges))[:, None] * np.diff(lon_edges)[None, :]
    ).ravel()
    valid = covered >= min_coverage * target_area

    # Mean over the covered part of each target cell
    scale = np.zeros_like(covered)
    scale[valid] = 1.0 / covered[valid]
    weights = sp.diags(scale) @ weights
    return weights.tocsr(), valid


def _cell_outlines(
    vlon: np.ndarray, vlat: np.ndarray, at_pole: np.ndarray, pieces: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Outlines of triangles as polygons in (lon, lat), padded for `_clip`.

    Args:
        vlon, vlat: (n, 3) vertices; the longitudes of each cell must be continuous.
        at_pole: (n, 3) whether the vertex is on a pole.
        pieces: Number of straight pieces that stand for each (great circle) edge.
    """
    n = vlon.shape[0]
    # Each edge gives `pieces` points, a vertex on a pole one more, and each of the
    # four clips can add one
    m = 3 * pieces + 1 + 4
    poly = np.zeros((n, m, 2))
    count = np.zeros(n, dtype=np.int64)
    xyz = _xyz(vlon, vlat)
    fractions = np.arange(pieces) / pieces

    for i in range(n) if (pieces > 1 or at_pole.any()) else ():
        outline = []
        for k in range(3):
            nxt = (k + 1) % 3
            if at_pole[i, k]:
                # A pole has no longitude of its own: the outline runs along the pole
                # between the longitudes of the two other vertices
                outline.append((vlon[i, (k - 1) % 3], vlat[i, k]))
                outline.append((vlon[i, nxt], vlat[i, k]))
                continue
            if at_pole[i, nxt] or pieces == 1:
                # Edges towards a pole are meridians, which are straight already
                outline.append((vlon[i, k], vlat[i, k]))
                continue
            points = xyz[i, k] + fractions[:, None] * (xyz[i, nxt] - xyz[i, k])
            lon = np.arctan2(points[:, 1], points[:, 0])
            lat = np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1]))
            lon = vlon[i, k] + _wrap(lon - vlon[i, k])
            outline += list(zip(lon, lat))
        poly[i, : len(outline)] = outline
        count[i] = len(outline)

    if pieces == 1 and not at_pole.any():
        poly[:, :3, 0] = vlon
        poly[:, :3, 1] = vlat
        count[:] = 3
    return poly, count


def _overlap_areas(
    poly: np.ndarray,
    count: np.ndarray,
    cell_index: np.ndarray,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Area of the overlap of each polygon with each cell of the lat/lon grid.

    Returns:
        Flat target cell index, `cell_index` of the polygon and area, for all pairs
        that overlap.
    """
    nlon, nlat = lon_edges.size - 1, lat_edges.size - 1
    m = poly.shape[1]

    # Cells that stick out of the target's longitude range may come back in on the
    # other side (global target grids): add a copy shifted by a full turn
    used = np.arange(m)[None, :] < count[:, None]
    over = np.flatnonzero(
        np.where(used, poly[:, :, 0], -np.inf).max(axis=1) > lon_edges[-1]
    )
    under = np.flatnonzero(
        np.where(used, poly[:, :, 0], np.inf).min(axis=1) < lon_edges[0]
    )
    east, west = poly[over].copy(), poly[under].copy()
    east[:, :, 0] -= 2 * np.pi
    west[:, :, 0] += 2 * np.pi
    poly = np.concatenate([poly, east, west])
    count = np.concatenate([count, count[over], count[under]])
    cell_index = np.concatenate([cell_index, cell_index[over], cell_index[under]])

    # Range of target cells that each polygon may overlap
    used = np.arange(m)[None, :] < count[:, None]
    lo = np.where(used[:, :, None], poly, np.inf).min(axis=1)
    hi = np.where(used[:, :, None], poly, -np.inf).max(axis=1)
    i0 = np.clip(np.searchsorted(lon_edges, lo[:, 0], side="right") - 1, 0, nlon)
    i1 = np.clip(np.searchsorted(lon_edges, hi[:, 0], side="left"), 0, nlon)
    j0 = np.clip(np.searchsorted(lat_edges, lo[:, 1], side="right") - 1, 0, nlat)
    j1 = np.clip(np.searchsorted(lat_edges, hi[:, 1], side="left"), 0, nlat)
    ni = np.maximum(i1 - i0, 0)
    nj = np.maximum(j1 - j0, 0)
    n_pairs = ni * nj

    rows_out, cols_out, data_out = [], [], []
    # Work through the (polygon, target cell) pairs in blocks to bound memory use
    block = 4_000_000 // m
    cumulative = np.concatenate([[0], np.cumsum(n_pairs)])
    start_poly = 0
    while start_poly < poly.shape[0]:
        end_poly = int(
            np.searchsorted(cumulative, cumulative[start_poly] + block, side="right")
        )
        end_poly = min(max(end_poly - 1, start_poly + 1), poly.shape[0])
        sel = np.arange(start_poly, end_poly)
        start_poly = end_poly

        which = np.repeat(sel, n_pairs[sel])
        if which.size == 0:
            continue
        offset = np.arange(which.size) - np.repeat(
            cumulative[sel] - cumulative[sel[0]], n_pairs[sel]
        )
        ii = i0[which] + offset % ni[which]
        jj = j0[which] + offset // ni[which]

        p, c = poly[which], count[which]
        p, c = _clip(p, c, 0, lon_edges[ii], keep_above=True)
        p, c = _clip(p, c, 0, lon_edges[ii + 1], keep_above=False)
        p, c = _clip(p, c, 1, lat_edges[jj], keep_above=True)
        p, c = _clip(p, c, 1, lat_edges[jj + 1], keep_above=False)
        area = _polygon_area(p, c)

        overlap = (c >= 3) & (area > 0)
        rows_out.append((jj * nlon + ii)[overlap])
        cols_out.append(cell_index[which][overlap])
        data_out.append(area[overlap])

    return (
        np.concatenate(rows_out) if rows_out else np.zeros(0, dtype=np.int64),
        np.concatenate(cols_out) if cols_out else np.zeros(0, dtype=np.int64),
        np.concatenate(data_out) if data_out else np.zeros(0),
    )


class Regridder:
    """
    Interpolates fields from an ICON grid to a regular lat/lon grid.

    Build one with `Regridder.build` (or `Regridder.cached` to keep the weights on
    disk), then call it on arrays whose last dimension is the ICON cell dimension.
    """

    def __init__(
        self,
        weights: sp.csr_matrix,
        valid: np.ndarray,
        target: LatLonGrid,
        method: str,
        grid_uuid: str = "",
    ):
        self.weights = weights.tocsr()
        self.weights.sort_indices()
        self.valid = np.asarray(valid, dtype=bool)
        self.target = target
        self.method = method
        self.grid_uuid = grid_uuid
        self._by_dtype: dict[np.dtype, sp.csr_matrix] = {}

    @property
    def ncells(self) -> int:
        return self.weights.shape[1]

    @classmethod
    def build(
        cls, grid: IconGrid, target: LatLonGrid | None = None, method: str = "linear"
    ) -> "Regridder":
        """
        Compute the weights.

        Args:
            grid: The ICON grid.
            target: The lat/lon grid. Defaults to `LatLonGrid.covering(grid)`.
            method: One of `nearest`, `linear`, `conservative`.
        """
        if target is None:
            target = LatLonGrid.covering(grid)
        if method == "nearest":
            weights, valid = _nearest_weights(grid, target)
        elif method == "linear":
            weights, valid = _linear_weights(grid, target)
        elif method == "conservative":
            weights, valid = _conservative_weights(grid, target)
        else:
            raise ValueError(f"Unknown method '{method}', expected one of {METHODS}")
        return cls(weights, valid, target, method, grid.uuid)

    @classmethod
    def cached(
        cls,
        grid_file: Path | str,
        cache_dir: Path | str,
        target: LatLonGrid | None = None,
        method: str = "linear",
    ) -> "Regridder":
        """
        Like `build`, but keeps the weights in `cache_dir` and reuses them if they
        were computed for the same ICON grid, target grid and method.
        """
        grid_file = Path(grid_file)
        cache_dir = Path(cache_dir)
        with netCDF4.Dataset(grid_file) as ds:
            uuid = str(getattr(ds, "uuidOfHGrid", "")) or grid_file.stem
        cache_dir.mkdir(parents=True, exist_ok=True)

        candidates = sorted(cache_dir.glob(f"regrid_{uuid}_{method}_*.npz"))
        for path in candidates:
            try:
                regridder = cls.load(path)
            except (OSError, ValueError, KeyError):
                continue
            if target is None or regridder.target_matches(target):
                return regridder

        grid = IconGrid.from_file(grid_file)
        regridder = cls.build(grid, target, method)
        nlat, nlon = regridder.target.shape
        index = len(candidates)
        path = cache_dir / f"regrid_{uuid}_{method}_{nlat}x{nlon}_{index}.npz"
        regridder.save(path)
        return regridder

    def target_matches(self, target: LatLonGrid) -> bool:
        return (
            self.target.shape == target.shape
            and np.allclose(self.target.lon, target.lon)
            and np.allclose(self.target.lat, target.lat)
        )

    def save(self, path: Path | str):
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(
            tmp,
            version=CACHE_VERSION,
            data=self.weights.data,
            indices=self.weights.indices,
            indptr=self.weights.indptr,
            shape=np.array(self.weights.shape),
            valid=self.valid,
            lon=self.target.lon,
            lat=self.target.lat,
            method=self.method,
            grid_uuid=self.grid_uuid,
        )
        tmp.rename(path)

    @classmethod
    def load(cls, path: Path | str) -> "Regridder":
        with np.load(path) as f:
            if int(f["version"]) != CACHE_VERSION:
                raise ValueError(f"{path} was written by another version")
            weights = sp.csr_matrix(
                (f["data"], f["indices"], f["indptr"]), shape=tuple(f["shape"])
            )
            return cls(
                weights,
                f["valid"],
                LatLonGrid(f["lon"], f["lat"]),
                str(f["method"]),
                str(f["grid_uuid"]),
            )

    def _weights_as(self, dtype: np.dtype) -> sp.csr_matrix:
        if dtype not in self._by_dtype:
            self._by_dtype[dtype] = self.weights.astype(dtype)
        return self._by_dtype[dtype]

    def __call__(
        self, data: np.ndarray, fill_value: float = np.nan, threads: int = 1
    ) -> np.ndarray:
        """
        Interpolate `data`.

        Args:
            data: Array whose last dimension is the cell dimension.
            fill_value: Value for target points outside the model domain.
            threads: Number of threads; each one handles some of the leading
                dimensions (levels, times).

        Returns:
            Array with the last dimension replaced by (lat, lon). Floating point
            input keeps its precision, anything else becomes float64.
        """
        data = np.asarray(data)
        if data.shape[-1] != self.ncells:
            raise ValueError(
                f"Last dimension has {data.shape[-1]} points, the grid has {self.ncells} cells"
            )
        dtype = data.dtype if data.dtype.kind == "f" else np.dtype(np.float64)
        weights = self._weights_as(dtype)
        leading = data.shape[:-1]
        flat = np.ascontiguousarray(data.reshape(-1, self.ncells), dtype=dtype)
        out = np.empty((flat.shape[0], self.target.size), dtype=dtype)
        invalid = ~self.valid

        def one(k: int):
            out[k] = weights @ flat[k]
            out[k, invalid] = fill_value

        if threads > 1 and flat.shape[0] > 1:
            with ThreadPoolExecutor(threads) as pool:
                list(pool.map(one, range(flat.shape[0])))
        else:
            for k in range(flat.shape[0]):
                one(k)
        return out.reshape(*leading, *self.target.shape)

    def regrid_dataset(
        self, ds: xr.Dataset, cell_dim: str | None = None, threads: int = 1
    ) -> xr.Dataset:
        """
        Interpolate all variables of a dataset that are defined on ICON cells.

        Variables without the cell dimension are copied as they are; the cell
        coordinates (`clon`, `clat` and their bounds) and variables on edges or
        vertices are dropped. Works with numpy- and dask-backed datasets.
        """
        if cell_dim is None:
            cell_dim = find_cell_dim(ds, self.ncells)

        unstructured = {
            dim
            for dim in ds.dims
            if dim != cell_dim
            and (
                str(dim).startswith("ncells")
                or dim in ("vertices", "nv", "edge", "vertex")
            )
        }
        drop = [
            name
            for name, var in ds.variables.items()
            if unstructured & set(var.dims)
            or (cell_dim in var.dims and name in ds.coords)
        ]
        ds = ds.drop_vars(drop)

        nlat, nlon = self.target.shape
        out = {}
        for name, var in ds.data_vars.items():
            if cell_dim not in var.dims:
                out[name] = var
                continue
            dtype = var.dtype if var.dtype.kind == "f" else np.dtype(np.float64)
            regridded = xr.apply_ufunc(
                self.__call__,
                var,
                kwargs={"threads": threads},
                input_core_dims=[[cell_dim]],
                output_core_dims=[["lat", "lon"]],
                dask="parallelized",
                output_dtypes=[dtype],
                dask_gufunc_kwargs={"output_sizes": {"lat": nlat, "lon": nlon}},
                keep_attrs=True,
            )
            for attr in ("CDI_grid_type", "number_of_grid_in_reference", "coordinates"):
                regridded.attrs.pop(attr, None)
            regridded.encoding.pop("coordinates", None)
            out[name] = regridded

        result = xr.Dataset(out, attrs=ds.attrs)
        result = result.assign_coords(
            lat=(
                "lat",
                self.target.lat,
                {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"},
            ),
            lon=(
                "lon",
                self.target.lon,
                {"standard_name": "longitude", "units": "degrees_east", "axis": "X"},
            ),
        )
        result.attrs["regrid_method"] = self.method
        return result


def find_cell_dim(ds: xr.Dataset, ncells: int) -> str:
    """Name of the dimension of `ds` that runs over the ICON cells."""
    for name in ("ncells", "cell", "cells"):
        if name in ds.sizes and ds.sizes[name] == ncells:
            return name
    matches = [str(dim) for dim, size in ds.sizes.items() if size == ncells]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"Cannot tell which dimension is the ICON cell dimension ({ncells} cells); "
        f"dimensions are {dict(ds.sizes)}"
    )
