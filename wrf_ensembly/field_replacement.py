"""
Replacing fields of a member's initial state (wrfinput or WRF restart file) with values
from a file the user provides, `[field_replacement]` in the config. Meant for fields
another model simulates better, e.g. OH from a global chemistry model.

A source file holds each replaced field either with exactly the shape of the field in
the target file (shared by all members), or with an extra leading `member` dimension
(one value per member). Fields with two time levels in restart files get the new value
in `_2` and the change added to `_1`, as for the analysis (see `restart.set_field`).

Everything here works on open files, the `Experiment` decides which files to use.
"""

import netCDF4
import numpy as np

from wrf_ensembly import restart, update_bc

MEMBER_DIM = "member"
"""Name of the leading dimension of per-member fields in source files"""


def _target_shape(target: netCDF4.Dataset, name: str) -> tuple[int, ...]:
    names = restart.field_names(target, name)
    if not names:
        raise ValueError(f"{name} not in {target.filepath()}")
    return target[names[-1]].shape


def check_source(
    source: netCDF4.Dataset,
    target: netCDF4.Dataset,
    variables: list[str],
    n_members: int,
    check_time: bool = True,
) -> dict[str, bool]:
    """
    Checks that a source file can replace `variables` in `target`: they exist in both,
    the shapes match, and if the source has `Times`, it's at the target's time (unless
    `check_time` is off, to check files of other cycles against one target).

    Returns:
        For each variable, whether the source has one value per member

    Raises:
        ValueError: describing the first problem found
    """

    source_name = source.filepath()
    if check_time and "Times" in source.variables:
        source_time = update_bc.parse_wrf_times(source["Times"])[0]
        target_time = update_bc.parse_wrf_times(target["Times"])[0]
        if source_time != target_time:
            raise ValueError(
                f"{source_name} is at {source_time}, but {target.filepath()} is at "
                f"{target_time}"
            )

    per_member = {}
    for name in variables:
        if name not in source.variables:
            raise ValueError(f"{name} not in {source_name}")
        shape = _target_shape(target, name)
        var = source[name]
        if var.shape == shape:
            per_member[name] = False
        elif (
            var.dimensions
            and var.dimensions[0] == MEMBER_DIM
            and var.shape == (n_members, *shape)
        ):
            per_member[name] = True
        else:
            raise ValueError(
                f"{name} in {source_name} has shape {var.shape} "
                f"{var.dimensions}, expected {shape} or ({MEMBER_DIM}={n_members}, "
                f"*{shape})"
            )
    return per_member


def replace_fields(
    target: netCDF4.Dataset,
    source: netCDF4.Dataset,
    variables: list[str],
    member_i: int,
    n_members: int,
) -> list[str]:
    """
    Writes `variables` from `source` into `target`, in place, after checking them with
    `check_source`. For a restart file, rebalance afterwards if a replaced field is part
    of the balance (see `rebalance.py`).

    Args:
        target: The member's initial state (wrfinput or restart file), opened for writing
        source: The file with the new values
        variables: Which fields to replace
        member_i: The member, to pick its values from per-member fields
        n_members: Ensemble size, the expected length of the `member` dimension

    Returns:
        The replaced fields
    """

    per_member = check_source(source, target, variables, n_members)
    for name in variables:
        value = source[name][member_i] if per_member[name] else source[name][:]
        # A fill value would be written as a number into a field WRF reads
        if np.ma.is_masked(value) or not np.all(np.isfinite(value)):
            raise ValueError(
                f"{name} in {source.filepath()} has missing or non-finite values"
                + (f" for member {member_i}" if per_member[name] else "")
            )
        restart.set_field(target, name, np.ma.getdata(value))

    target.replaced_fields = f"{', '.join(variables)} from {source.filepath()}"
    return list(variables)
