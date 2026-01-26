"""
Statistics utilities for ensemble postprocessing.

Provides Welford's algorithm for online computation of mean and variance,
as well as utilities for creating NetCDF files from templates.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import netCDF4
import numpy as np
import xarray as xr

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


def get_structure_from_xarray(
    ds: xr.Dataset, reference_time: np.datetime64 | None = None
) -> NetCDFFile:
    """
    Extract NetCDFFile structure from an xarray Dataset.

    This is useful when you have an in-memory processed dataset and need
    to create output files with the same structure.

    Args:
        ds: xarray Dataset to extract structure from.
        reference_time: Reference time for the time coordinate. If provided,
            the time variable will be stored as integer minutes since this time.
            If None and the time variable is datetime64, uses the first time value.

    Returns:
        NetCDFFile with dimensions, variables, and attributes from the dataset.
    """
    dims: dict[str, int] = {}
    variables: dict[str, NetCDFVariable] = {}
    attrs: dict[str, str] = {}

    for dim_name, dim_size in ds.sizes.items():
        dims[dim_name] = dim_size

    for var_name in list(ds.data_vars) + list(ds.coords):
        var = ds[var_name]
        var_attrs = dict(var.attrs)

        # Special handling for time coordinate (datetime64 -> int64 minutes)
        if var_name == "t" and np.issubdtype(var.dtype, np.datetime64):
            # Determine reference time
            if reference_time is None:
                reference_time = var.values.flatten()[0]

            # Convert reference time to string for units attribute
            ref_time_str = np.datetime_as_string(reference_time, unit="s").replace(
                "T", " "
            )
            var_attrs["units"] = f"minutes since {ref_time_str}"
            var_attrs["calendar"] = "standard"

            nc_var = NetCDFVariable(
                name=var_name,
                dimensions=var.dims,
                attributes=var_attrs,
                dtype=np.dtype("int32"),
            )
            # Don't store constant value - time will be written incrementally
        else:
            nc_var = NetCDFVariable(
                name=var_name,
                dimensions=var.dims,
                attributes=var_attrs,
                dtype=var.dtype,
            )

            # Store constant values for coordinate variables and non-numeric types
            if (
                not np.issubdtype(var.dtype, np.number)
                or var_name in COORDINATE_VARIABLES
            ):
                nc_var.constant_value = var.values

        variables[var_name] = nc_var

    attrs = dict(ds.attrs)

    return NetCDFFile(dims, variables, attrs)


def create_welford_accumulators(template_ds: xr.Dataset) -> dict[str, WelfordState]:
    """
    Initialize Welford accumulators from a template xarray Dataset.

    Creates accumulators for all numeric, non-coordinate variables in the dataset.

    Args:
        template_ds: xarray Dataset to use as a template for variable shapes.

    Returns:
        Dictionary mapping variable names to their WelfordState accumulators.
    """
    accumulators = {}

    for var_name in template_ds.data_vars:
        if var_name in COORDINATE_VARIABLES:
            continue

        var = template_ds[var_name]
        if not np.issubdtype(var.dtype, np.number):
            continue

        shape = var.shape
        accumulators[var_name] = WelfordState(
            count=0,
            mean=np.zeros(shape, dtype=np.float64),
            m2=np.zeros(shape, dtype=np.float64),
        )

    return accumulators


def update_accumulators_from_dataset(
    accumulators: dict[str, WelfordState],
    ds: xr.Dataset,
) -> None:
    """
    Update Welford accumulators with data from an xarray Dataset (in-place).

    Args:
        accumulators: Dictionary of WelfordState objects to update.
        ds: xarray Dataset containing the new values.
    """
    for var_name, state in accumulators.items():
        if var_name in ds:
            welford_update(state, ds[var_name].values)


def finalize_accumulators(
    accumulators: dict[str, WelfordState],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Finalize Welford accumulators and return mean and standard deviation.

    Args:
        accumulators: Dictionary of WelfordState objects to finalize.

    Returns:
        Tuple of (means, stddevs) dictionaries mapping variable names to arrays.
    """
    means = {}
    stddevs = {}

    for var_name, state in accumulators.items():
        mean, sd = welford_finalise(state)
        means[var_name] = mean
        stddevs[var_name] = sd

    return means, stddevs
