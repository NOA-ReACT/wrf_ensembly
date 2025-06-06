"""
Python implementation for ensemble statistics, possible candidate for replacing nco and cdo.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import netCDF4
import numpy as np

from wrf_ensembly.console import logger


@dataclass
class NetCDFVariable:
    """
    Represents a variable in a netCDF file.
    If the variable is non-numeric or a coordinate variable, the value is stored in `constant_value` which is
    copied to the output file from the first member file.
    """

    name: str
    dimensions: tuple[str, ...]
    attributes: dict[str, str]
    dtype: np.dtype

    constant_value = None


# List of coordinate variables name in wrf-ensembly forecast files.
COORDINATE_VARIABLES = {"XLAT", "XLONG", "x", "y", "z", "t"}


@dataclass
class NetCDFFile:
    """Represents a netCDF file (variables, attributes, dimensions). Contains no data."""

    dimensions: dict[str, int]
    variables: dict[str, NetCDFVariable]
    global_attributes: dict[str, str]


def get_structure(file: Path) -> NetCDFFile:
    """
    Given a netCDF file, return its structure (dimensions, variables, attributes).
    No data is read.
    """

    dims: dict[str, int] = {}
    variables: dict[str, NetCDFVariable] = {}
    attrs: dict[str, str] = {}

    with netCDF4.Dataset(file, "r") as ds:
        for dim_name, dim in ds.dimensions.items():
            dims[dim_name] = len(dim)

        for var_name, var in ds.variables.items():
            variables[var_name] = NetCDFVariable(
                name=var_name,
                dimensions=var.dimensions,
                attributes={attr: getattr(var, attr) for attr in var.ncattrs()},
                dtype=var.dtype,
            )

            # Check if the variable is non-numeric or a coordinate variable and store the value as a constant
            if (
                not np.issubdtype(var.dtype, np.number)
                or var_name in COORDINATE_VARIABLES
            ):
                variables[var_name].constant_value = var[:]

        attrs = {attr: getattr(ds, attr) for attr in ds.ncattrs()}

    return NetCDFFile(dims, variables, attrs)


def create_file(
    path: Path,
    template: NetCDFFile,
    zlib=True,
    complevel: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = 2,
) -> netCDF4.Dataset:
    """
    Creates a netCDF4 file at the given path with the structure of the template.
    The opened file is returned in write mode.

    The `time` dimension is created as an unlimited dimension, regardless of the original size.

    Args:
        path: Path to the output file.
        template: Template structure to copy.
        zlib: Whether to use zlib compression.
        complevel: If using zlib, what compression level to use (0-9).
    """

    output_dir = path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    ds = netCDF4.Dataset(path, "w", format="NETCDF4")

    for dim_name, dim_size in template.dimensions.items():
        if dim_name.lower() == "t":
            # Create time as an unlimited dimension
            ds.createDimension(dim_name, None)
        else:
            ds.createDimension(dim_name, dim_size)

    for var_name, var_tmpl in template.variables.items():
        var = ds.createVariable(
            var_name,
            var_tmpl.dtype,
            var_tmpl.dimensions,
            fill_value=var_tmpl.attributes.get("_FillValue", None),
            zlib=zlib,
            complevel=complevel,
        )

        for attr_name, attr_value in var_tmpl.attributes.items():
            if attr_name == "_FillValue":
                continue
            var.setncattr(attr_name, attr_value)

        if var_tmpl.constant_value is not None:
            var[:] = var_tmpl.constant_value

    for attr_name, attr_value in template.global_attributes.items():
        ds.setncattr(attr_name, attr_value)

    # Add a comment for provenance
    ds.setncattr("wrf_ensembly", "Created by wrf_ensembly")

    return ds


@dataclass
class WelfordState:
    """Represents the state of Welford's algorithm for variance calculation."""

    count: int
    mean: np.ndarray
    m2: np.ndarray


def welford_update(state: WelfordState, new_value: np.ndarray) -> None:
    """
    Welford's algorithm for updating mean and variance incrementally.
    The `state` argument is updated in place with the new values.
    """

    state.count += 1
    delta = new_value - state.mean
    state.mean += delta / state.count
    delta2 = new_value - state.mean
    state.m2 += delta * delta2


def welford_finalise(
    state: WelfordState,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Finalise the Welford's algorithm to get variance and standard deviation.
    """

    if state.count < 2:
        return state.mean, np.full_like(state.mean, np.nan)

    variance = state.m2 / (state.count - 1)
    stddev = np.sqrt(variance)

    return state.mean, stddev


def compute_ensemble_statistics(
    member_files: list[Path],
    output_mean_file: Path,
    output_std_file: Path,
    zlib=True,
    complevel: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = 2,
):
    """
    Compute ensemble mean and standard deviation for a list of member files.
    The output files are created with the same structure as the member files.

    Statistics will not be computed for coordinate variables or non-numeric variables.

    Args:
        member_files: List of paths to the member files for a given cycle.
        output_mean_file: Path to the output mean file.
        output_std_file: Path to the output standard deviation file.
        zlib: Whether to use zlib compression.
        complevel: If using zlib, what compression level to use (0-9).
    """

    if not member_files:
        raise ValueError("No member files provided.")

    output_mean_file.unlink(missing_ok=True)
    output_std_file.unlink(missing_ok=True)

    template = get_structure(member_files[0])
    mean_ds = create_file(output_mean_file, template, zlib, complevel)
    std_ds = create_file(output_std_file, template, zlib, complevel)

    results: dict[str, WelfordState] = {}
    for i, path in enumerate(member_files):
        logger.info(f"Processing member {i + 1}/{len(member_files)}: {path}")

        # For each variable, run welford_update and keep the state in `results`
        with netCDF4.Dataset(path, "r") as ds:
            for var_name, var in ds.variables.items():
                # Skip non-numeric variables and coordinate variables
                if (
                    not np.issubdtype(var.dtype, np.number)
                    or var_name in COORDINATE_VARIABLES
                ):
                    continue

                if var_name not in results:
                    results[var_name] = WelfordState(
                        0, np.zeros(var.shape), np.zeros(var.shape)
                    )

                welford_update(results[var_name], var[:])

    # Then use `welford_finalise` to get the mean and stddev for each variable,
    # which are then written to the output files.
    for var_name, state in results.items():
        mean, stddev = welford_finalise(state)

        # Write mean and stddev to the output files
        mean_var = mean_ds.variables[var_name]
        std_var = std_ds.variables[var_name]

        mean_var[:] = mean
        std_var[:] = stddev

    mean_ds.close()
    std_ds.close()
