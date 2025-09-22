"""Functions related to handling WRF-Ensembly observation files under the context of an experiment."""

import numpy as np
import pandas as pd
import pyproj
import xarray as xr


def project_locations_to_wrf(
    df: pd.DataFrame, transformer: pyproj.Transformer
) -> pd.DataFrame:
    """
    Projects the latitude and longitude columns of a dataframe to the WRF domain's (x, y) coordinates,
    using the given transformer (create it with `wrf::get_wrf_proj_transformer()`).

    Args:
        df: The dataframe containing 'latitude' and 'longitude' columns (WRF-Ensembly observation file)
        transformer: The pyproj Transformer to use for the projection, created with `wrf::get_wrf_proj_transformer()`

    Returns:
        A new dataframe with added 'x' and 'y' columns. The original dataframe is not modified.
    """

    x, y = transformer.transform(df["longitude"].values, df["latitude"].values)
    df = df.copy()
    df["x"] = x
    df["y"] = y
    return df


def reconstruct_array(
    df: pd.DataFrame, fill_value=np.nan, trim_all_nan_slices=True
) -> xr.DataArray:
    """
    The WRF-Ensembly observation dataframes have an 'orig_coords' column which contains
    the original shape and indices of the observation in the original file. This function
    takes the dataframe and reconstructs the original array, filling missing values with `fill_value`.

    Args:
        df: The dataframe containing the observations, with an 'orig_coords' column.
            Please filter to only include observations from a single original file and quantity.
        fill_value: The value to use for missing entries in the reconstructed array
        trim_all_nan_slices: If True, drop slices along any dimension where all values are
            NaN. Useful when plotting because the spatial filtering may have removed
            entire rows/columns.

    Returns:
        An xarray DataArray with the reconstructed data. The latitude, longitude, z and time coordinates will be included.
    """

    if "orig_coords" not in df.columns:
        raise ValueError("Dataframe must contain 'orig_coords' column")

    # This would be a job for `len(orig_coords.unique()) == 1` but you can't do unique on dicts
    first_orig_coords = df["orig_coords"].iloc[0]
    for i in range(1, df.shape[0]):
        other_row = df["orig_coords"].iloc[i]
        if (other_row["names"] != first_orig_coords["names"]).any():
            raise ValueError(
                "Dataframe must only contain observations from a single original file"
            )
        if (other_row["shape"] != first_orig_coords["shape"]).any():
            raise ValueError(
                "Dataframe must only contain observations from a single original file"
            )

    orig_coords = df["orig_coords"].iloc[0]
    shape = orig_coords["shape"]
    indices = orig_coords["indices"]
    names = orig_coords["names"]

    if len(shape) != len(names):
        raise ValueError(
            "'shape' and 'names' in 'orig_coords' must have the same length"
        )
    if len(indices) != len(names):
        raise ValueError(
            "'indices' and 'names' in 'orig_coords' must have the same length"
        )

    array = np.full(shape, fill_value)
    latitude = np.full(shape, np.nan)
    longitude = np.full(shape, np.nan)
    z = np.full(shape, np.nan)
    time = np.full(shape, np.nan, dtype="datetime64[ns]")

    for i, row in df.iterrows():
        idx = tuple(row["orig_coords"]["indices"])
        array[idx] = row["value"]
        latitude[idx] = row["latitude"]
        longitude[idx] = row["longitude"]
        z[idx] = row["z"]
        time[idx] = pd.to_datetime(row["time"])

    coords = {
        "time": (names, time),
        "latitude": (names, latitude),
        "longitude": (names, longitude),
        "z": (names, z),
    }
    for i, name in enumerate(names):
        coords[name] = np.arange(shape[i])

    data_array = xr.DataArray(array, dims=names, coords=coords)

    if trim_all_nan_slices:
        # Drop slices along any dimension where all values are NaN
        for dim in names:
            data_array = data_array.dropna(dim=dim, how="all")

    return data_array
