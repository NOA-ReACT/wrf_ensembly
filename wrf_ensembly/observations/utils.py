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
    df: pd.DataFrame,
    value_columns: list[str] | None = None,
    fill_value=np.nan,
    trim_all_nan_slices: bool = True,
) -> xr.Dataset:
    """
    Reconstruct the original array shape from a flat observation DataFrame using the
    `orig_coords` column (which contains the original shape, indices, and dimension
    names from the source file).

    Missing entries are filled with `fill_value`. Coordinate fields (latitude,
    longitude, z, time, qc_flag) are always reconstructed alongside the requested
    value columns.

    Args:
        df: DataFrame containing observations with an `orig_coords` column.
            Must contain observations from a single original file and quantity
            (i.e., all rows must share the same `orig_coords.shape` and
            `orig_coords.names`).
        value_columns: Which columns to include as data variables in the output
            Dataset. Defaults to ``["value", "value_uncertainty"]``. Useful for
            including precomputed model equivalents, e.g.
            ``["value", "value_uncertainty", "model_equivalent"]``.
        fill_value: Fill value for array positions with no observation.
        trim_all_nan_slices: If True, drop slices along any dimension where all
            values are NaN. Useful when spatial filtering has removed entire
            rows or columns from the original grid.

    Returns:
        xarray Dataset with reconstructed data variables and coordinate fields
        (latitude, longitude, z, time, qc_flag) mapped back to their original
        positions.

    Raises:
        ValueError: If `orig_coords` is missing, or rows have inconsistent
            shapes/names.
    """

    if "orig_coords" not in df.columns:
        raise ValueError("DataFrame must contain an 'orig_coords' column")
    if df.empty:
        raise ValueError("DataFrame is empty")

    value_columns = value_columns or ["value", "value_uncertainty"]

    # Validate consistency across rows
    shapes = df["orig_coords"].apply(lambda r: tuple(r["shape"]))
    names_series = df["orig_coords"].apply(lambda r: tuple(r["names"]))
    if shapes.nunique() != 1:
        raise ValueError(
            "All rows must share the same orig_coords.shape "
            "(DataFrame must contain observations from a single original file)"
        )
    if names_series.nunique() != 1:
        raise ValueError(
            "All rows must share the same orig_coords.names "
            "(DataFrame must contain observations from a single original file)"
        )

    # Extract shape, names, build index arrays
    first = df["orig_coords"].iloc[0]
    shape = tuple(first["shape"])
    names = list(first["names"])

    if len(shape) != len(names):
        raise ValueError(
            "'shape' and 'names' in 'orig_coords' must have the same length"
        )

    indices = np.array(
        df["orig_coords"].apply(lambda r: tuple(r["indices"])).tolist(),
        dtype=np.intp,
    )
    idx_tuple = tuple(indices[:, k] for k in range(indices.shape[1]))

    # Reconstruct the value columns
    data_vars: dict[str, tuple[list[str], np.ndarray]] = {}
    for col in value_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame")
        arr = np.full(shape, fill_value, dtype=np.float64)
        arr[idx_tuple] = df[col].values
        data_vars[col] = (names, arr)

    # Reconstruct coordinate fields
    latitude = np.full(shape, np.nan)
    longitude = np.full(shape, np.nan)
    z = np.full(shape, np.nan)
    qc_flag = np.full(shape, np.nan)
    latitude[idx_tuple] = df["latitude"].values
    longitude[idx_tuple] = df["longitude"].values
    z[idx_tuple] = df["z"].values
    qc_flag[idx_tuple] = df["qc_flag"].values

    time = np.full(shape, np.datetime64("NaT"), dtype="datetime64[ns]")
    time[idx_tuple] = pd.to_datetime(df["time"]).values

    coords: dict[str, tuple[list[str], np.ndarray] | np.ndarray] = {
        "time": (names, time),
        "latitude": (names, latitude),
        "longitude": (names, longitude),
        "z": (names, z),
        "qc_flag": (names, qc_flag),
    }
    # Dimension index coordinates
    for i, name in enumerate(names):
        coords[name] = np.arange(shape[i])

    dataset = xr.Dataset(data_vars, coords=coords)

    if trim_all_nan_slices:
        for dim in names:
            # Use the first value column as the reference for trimming
            dataset = dataset.dropna(dim=dim, how="all", subset=[value_columns[0]])

    return dataset
