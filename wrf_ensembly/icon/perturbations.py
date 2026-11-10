"""
Perturbation fields on the ICON grid.

WRF's fields are white noise smoothed with a Gaussian filter, which needs a regular
grid. Here the same is done on a lat/lon grid over the domain with about the spacing
of the ICON grid, and the result is read off at the cell centres. `gaussian_sigma`
therefore still counts grid spacings.
"""

import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import netCDF4
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from wrf_ensembly import perturbations
from wrf_ensembly.console import logger
from wrf_ensembly.icon import namelists
from wrf_ensembly.icon.regrid import IconGrid

if TYPE_CHECKING:
    from wrf_ensembly.experiment import Experiment


class CellSampler:
    """
    Bilinear interpolation from a regular lat/lon grid over the domain of an ICON grid
    to its cell centres.

    Args:
        margin: How many grid points the lat/lon grid extends beyond the domain
    """

    def __init__(self, grid: IconGrid, margin: int = 0):
        if grid.is_global():
            raise ValueError("Perturbations need a limited-area grid")

        # Longitudes relative to the middle of the domain, so that one across the
        # dateline stays in one piece
        lon0 = np.arctan2(np.sin(grid.clon).mean(), np.cos(grid.clon).mean())
        lon = np.degrees((grid.clon - lon0 + np.pi) % (2 * np.pi) - np.pi)
        lat = np.degrees(grid.clat)

        # The same distance in both directions: degrees of longitude are shorter away
        # from the equator
        dlat = grid.resolution_deg()
        dlon = dlat / max(np.cos(np.radians(lat.mean())), 0.1)

        x = (lon - lon.min()) / dlon + margin
        y = (lat - lat.min()) / dlat + margin
        self.shape = (
            int(np.ceil(y.max())) + 2 + margin,
            int(np.ceil(x.max())) + 2 + margin,
        )
        self._j = np.floor(y).astype(int)
        self._i = np.floor(x).astype(int)
        self._fy = y - self._j
        self._fx = x - self._i

    def __call__(self, field: np.ndarray) -> np.ndarray:
        """`field` (..., lat, lon) at the cell centres, (..., cell)"""

        j, i, fy, fx = self._j, self._i, self._fy, self._fx
        return (
            field[..., j, i] * (1 - fy) * (1 - fx)
            + field[..., j, i + 1] * (1 - fy) * fx
            + field[..., j + 1, i] * fy * (1 - fx)
            + field[..., j + 1, i + 1] * fy * fx
        )


def generate_perturbation_ensemble(
    n_members: int,
    shape: tuple[int, ...],
    mean: float,
    sd: float,
    sigma: float = 2.5,
    boundary: int = 0,
    min_value: float | None = None,
    max_value: float | None = None,
    *,
    grid_file: Path,
) -> np.ndarray:
    """
    Perturbation fields for all members of a variable with the cells as its last
    dimension, like `wrf_ensembly.perturbations.generate_perturbation_ensemble` for a
    regular grid. The noise is smoothed along the other dimensions (levels) as well.

    Args:
        shape: Shape of the variable, (..., cell)
        grid_file: The ICON grid file

    Returns:
        Array of shape (n_members, *shape)
    """

    if boundary > 0:
        raise ValueError("`boundary` is not supported on the ICON grid")

    # The filter mirrors the noise at the edges of the array, which makes the field
    # there less smooth and its variance larger. Keep that outside of the domain.
    sampler = CellSampler(IconGrid.from_file(grid_file), margin=int(np.ceil(4 * sigma)))
    if shape[-1] != sampler._i.size:
        raise ValueError(
            f"The variable has {shape[-1]} cells, the grid in {grid_file} has "
            f"{sampler._i.size}"
        )

    out = np.empty((n_members, *shape))
    for member in range(n_members):
        # One member at a time: on a fine grid, all at once would not fit in memory
        x = np.random.normal(1, 10, (*shape[:-1], *sampler.shape))
        x = sampler(gaussian_filter(x, sigma=sigma))
        x = (x - x.mean()) / x.std()
        out[member] = x * sd + mean

    if min_value is not None:
        np.maximum(out, min_value, out=out)
    if max_value is not None:
        np.minimum(out, max_value, out=out)
    return out


def generate(exp: "Experiment"):
    """
    Writes the perturbation fields of the first cycle, the only one that is perturbed,
    to `perturbations/perts_cycle_0.nc` in the diagnostics directory
    """

    from wrf_ensembly.icon.model import for_member

    icon = exp.cfg.icon
    assert icon is not None
    perturbations.generate_perturbations_for_cycle(
        cycle_i=0,
        perturbations_cfg=exp.cfg.perturbations,
        reapplied_parameters=[],
        n_members=exp.cfg.assimilation.n_members,
        experiment_name=exp.cfg.metadata.name,
        ic_path=for_member(icon.initial_conditions, 0),
        diag_dir=exp.paths.data_diag,
        generate=partial(generate_perturbation_ensemble, grid_file=icon.grid_file),
    )


def apply(exp: "Experiment", member_i: int):
    """
    Adds the perturbations to a member's copy of the IFS initial conditions, which the
    first cycle starts from. ICON interpolates that file to its levels and integrates
    the pressure hydrostatically when it reads it.
    """

    if exp.current_cycle_i != 0:
        # Later cycles start from the first guess, which is not perturbed
        logger.info(f"Member {member_i}: With ICON, only the first cycle is perturbed")
        return

    pert_file = exp.paths.data_diag / "perturbations" / "perts_cycle_0.nc"
    if not pert_file.exists():
        raise FileNotFoundError(
            f"No perturbation file at {pert_file}, run `ensemble generate-perturbations`"
        )
    initial = exp.paths.member_path(member_i) / namelists.IFS_LINK
    if not initial.exists():
        raise FileNotFoundError(
            f"Member {member_i}: No initial conditions at {initial}, run `ensemble setup`"
        )
    if initial.is_symlink():
        raise ValueError(
            f"Member {member_i}: {initial} is a link to the shared initial conditions, "
            "run `ensemble setup` again to give the member its own copy"
        )

    with (
        xr.open_dataset(pert_file) as perts,
        netCDF4.Dataset(initial, "r+") as ds,
    ):
        for name, var in perts.sel(member=member_i).data_vars.items():
            if name not in ds.variables:
                raise ValueError(f"Member {member_i}: No variable {name} in {initial}")
            operation = json.loads(var.attrs["cfg"])["operation"]
            field = var.to_numpy()
            current = ds[name][:]
            match operation:
                case "add":
                    ds[name][:] = current + field
                case "multiply":
                    ds[name][:] = current * field
                case "assign":
                    ds[name][:] = np.broadcast_to(field, current.shape)
                case _:
                    raise ValueError(f"Unknown perturbation operation: {operation}")
            logger.debug(f"Member {member_i}: Perturbed {name} ({operation})")

    logger.info(f"Member {member_i}: Applied perturbations to {initial.name}")
