"""Functions related to handling WRF-Ensembly observation files under the context of an experiment."""

import pandas as pd
import pyproj


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
