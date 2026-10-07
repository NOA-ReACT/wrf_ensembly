"""
Writing into WRF restart files, for the `restart` cycling mode.

Restart files keep both time levels of the prognostic dynamics fields (`U_1`/`U_2`,
`THM_1`/`THM_2`, ...), where `_2` is the current state and `_1` the state one time step
earlier. WRF overwrites `_1` with `_2` at the start of the first time step, but with
`use_theta_m = 1` the physics read `THM_1` before that. A new state goes into `_2`, and
the same change is added to `_1`, so both levels keep their relation. Everything else
(moisture, chemistry, physics) has one level and keeps its name.

After changing the state, call `rebalance.rebalance` so that the pressure and density
match it. Take `rebalance.balanced_fields` before the change and pass it along, so that
only the change is applied.
"""

import netCDF4

from wrf_ensembly import update_bc
from wrf_ensembly.console import logger


def field_names(ds: netCDF4.Dataset, name: str) -> list[str]:
    """
    The names a field is stored under in a WRF restart file: both time levels for the
    prognostic dynamics fields, otherwise just `name`. Empty if the file doesn't have it.
    """

    if f"{name}_2" in ds.variables:
        return [f"{name}_1", f"{name}_2"]
    if name in ds.variables:
        return [name]
    return []


def write_state(
    restart: netCDF4.Dataset, source: netCDF4.Dataset, names: list[str]
) -> list[str]:
    """
    Copies fields from a wrfinput/wrfout-style file (e.g. the analysis) into a restart
    file, in place. Both files must be at the same time. For fields with two time levels,
    the field goes into `_2` and the change is added to `_1`.

    Args:
        restart: The restart file, opened for writing
        source: The file to copy from, fields under their plain names (U, THM, ...)
        names: Which fields to copy. Fields missing from either file are skipped with a
               warning.

    Returns:
        The names of the fields that were copied
    """

    restart_time = update_bc.parse_wrf_times(restart["Times"])[0]
    source_time = update_bc.parse_wrf_times(source["Times"])[0]
    if restart_time != source_time:
        raise ValueError(
            f"Restart file is at {restart_time} but the source is at {source_time}"
        )

    written = []
    for name in names:
        targets = field_names(restart, name)
        if name not in source.variables or not targets:
            where = "source" if name not in source.variables else "restart file"
            logger.warning(f"{name} not in the {where}, not copied")
            continue
        field = source[name][:]
        if len(targets) == 2:
            level_1, level_2 = targets
            restart[level_1][:] = restart[level_1][:] + (field - restart[level_2][:])
            restart[level_2][:] = field
        else:
            restart[targets[0]][:] = field
        written.append(name)
    return written
