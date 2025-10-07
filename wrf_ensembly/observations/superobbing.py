"""Contains downsampling (super-orbing) methods for reducing the resolution of observations."""

import numpy as np
import pandas as pd
import duckdb

from wrf_ensembly import wrf
from wrf_ensembly.config import DomainControlConfig, SuperobbingConfig


def superobb_grid_binning_duckdb(
    con: duckdb.DuckDBPyConnection,
    config: SuperobbingConfig,
    domain: DomainControlConfig,
    instrument: str,
    quantity: str,
) -> pd.DataFrame:
    """
    Does spatio-temporal binning to create superobservations using DuckDB for memory efficiency.

    This function performs the binning entirely in DuckDB to avoid loading large datasets into memory.
    The output uncertainty assumes independent errors.

    This is the recommended approach for large datasets as it:
    - Reduces memory usage by performing aggregations in DuckDB
    - Avoids loading all observations into pandas DataFrames
    - Scales better with dataset size

    Args:
        con: DuckDB connection with observations table
        config: Configuration specifying the clustering radii
        domain: Domain configuration for coordinate transformations
        instrument: Instrument name to filter observations
        quantity: Quantity name to filter observations

    Returns:
        pd.DataFrame: DataFrame containing the superobservations
    """

    # Build the SQL query for binning and aggregation
    z_bin_clause = ""
    z_bin_select = ""
    z_bin_group = ""
    z_center_calc = ""

    if config.spatial_radius_z is not None:
        z_bin_clause = f", CAST(z / {config.spatial_radius_z} AS INTEGER) as z_bin"
        z_bin_select = ", z_bin"
        z_bin_group = ", z_bin"
        z_center_calc = f", (z_bin + 0.5) * {config.spatial_radius_z} as z"
    else:
        z_center_calc = ", AVG(z) as z"

    # Main aggregation query
    superobs_query = f"""
    WITH binned_obs AS (
        SELECT *,
            CAST(x / {config.spatial_radius_x_meters} AS INTEGER) as x_bin,
            CAST(y / {config.spatial_radius_y_meters} AS INTEGER) as y_bin,
            CAST(EXTRACT(EPOCH FROM time) / {config.temporal_radius_seconds} AS INTEGER) as t_bin{z_bin_clause}
        FROM observations
        WHERE instrument = '{instrument}' AND quantity = '{quantity}'
    )
    SELECT
        -- Keep first of categorical columns
        ANY_VALUE(instrument) as instrument,
        ANY_VALUE(quantity) as quantity,
        ANY_VALUE(z_type) as z_type,
        ANY_VALUE(qc_flag) as qc_flag,
        ANY_VALUE(orig_coords) as orig_coords,
        ANY_VALUE(orig_filename) as orig_filename,
        ANY_VALUE(metadata) as metadata,

        -- Bin identifiers for merging later
        x_bin, y_bin, t_bin{z_bin_select},

        -- Use bin centers for coordinates
        -- (x_bin + 0.5) * {config.spatial_radius_x_meters} as x,
        -- (y_bin + 0.5) * {config.spatial_radius_y_meters} as y
        -- {z_center_calc},
        AVG(x) as x,
        AVG(y) as y,
        AVG(z) as z,
        TO_TIMESTAMP((t_bin + 0.5) * {config.temporal_radius_seconds}) as time,

        -- Aggregated values
        AVG(value) as value,
        SQRT(SUM(value_uncertainty * value_uncertainty) / COUNT(*)) as value_uncertainty,

        -- For downsampling info
        COUNT(*) as n_observations,
        MAX(EXTRACT(EPOCH FROM time)) - MIN(EXTRACT(EPOCH FROM time)) as time_spread_seconds,
        AVG(x) as centroid_x,
        AVG(y) as centroid_y,
        --SQRT(
        --    POWER(x - AVG(x), 2) + POWER(y - AVG(y), 2)
        --) as spatial_spread_meters
        -- TODO: spatial spread needs to be calculated in a second query

    FROM binned_obs
    GROUP BY x_bin, y_bin, t_bin{z_bin_group}
    """

    # Execute the query and get the result
    # We have to bring it back to python land so we can use pyproject to compute lat/lon
    superobs = con.execute(superobs_query).fetchdf()

    if superobs.empty:
        return superobs

    # Create the downsampling_info structure
    superobs["downsampling_info"] = superobs.apply(
        lambda row: {
            "method": "grid_binning",
            "n_observations": int(row["n_observations"]),
            "time_spread_seconds": float(row["time_spread_seconds"]),
            "spatial_spread_meters": -1.0,  # Placeholder, needs proper calculation TODO
        },
        axis=1,
    )

    # Drop the temporary columns
    drop_cols = [
        "n_observations",
        "time_spread_seconds",
        "centroid_x",
        "centroid_y",
        # "spatial_spread_meters",
    ]
    drop_cols.extend([col for col in superobs.columns if "_bin" in col])
    superobs = superobs.drop(columns=drop_cols)

    # Compute lat/lon from x/y using the transformer
    transformer = wrf.get_wrf_reverse_proj_transformer(domain)
    lons, lats = transformer.transform(
        superobs["x"].to_numpy(), superobs["y"].to_numpy()
    )
    superobs["longitude"] = lons
    superobs["latitude"] = lats

    return superobs


def superobb_grid_binning(
    df: pd.DataFrame, config: SuperobbingConfig, domain: DomainControlConfig
) -> pd.DataFrame:
    """
    Does spatio-temporal binning to create superobservations.

    This is the original pandas-based implementation that loads all observations into memory.
    For large datasets, consider using `superobb_grid_binning_duckdb` instead for better memory efficiency.

    The output uncertainty assumes independent errors, which is probably not true. Or maybe it is.
    Who knows.

    Args:
        df: Input dataframe with columns: x, y, z, time, value, value_uncertainty
        Additional columns are preserved in the output
        config: Configuration specifying the clustering radii, see `config.py`
    Returns:
        pd.DataFrame: DataFrame containing the superobservations
    """

    if df.empty:
        return df.copy()
    required_cols = ["x", "y", "z", "time", "value", "value_uncertainty"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df_work = df.copy()

    # Create an epoch column to use seconds for binning
    df_work["t_seconds"] = df_work["time"].apply(lambda x: x.timestamp())

    # Put each observation into a spatio-temporal bin
    df_work["x_bin"] = (df_work["x"] // config.spatial_radius_x_meters).astype(int)
    df_work["y_bin"] = (df_work["y"] // config.spatial_radius_y_meters).astype(int)
    df_work["t_bin"] = (df_work["t_seconds"] // config.temporal_radius_seconds).astype(
        int
    )
    if config.spatial_radius_z is not None:
        df_work["z_bin"] = (df_work["z"] // config.spatial_radius_z).astype(int)
        bin_cols = ["x_bin", "y_bin", "z_bin", "t_bin"]
    else:
        bin_cols = ["x_bin", "y_bin", "t_bin"]

    superobs = df_work.groupby(bin_cols, as_index=False).agg(
        {
            "instrument": "first",
            "quantity": "first",
            "z_type": "first",
            "qc_flag": "first",
            "orig_coords": "first",  # misleading?
            "orig_filename": "first",
            "metadata": "first",  # trouble?
            "z": "mean",
            "x": "mean",
            "y": "mean",
            "t_seconds": "mean",
            "time": "median",
            "value": "mean",
            "value_uncertainty": lambda x: (
                np.sqrt(np.sum(x**2)) / len(x)
            ),  # Combined uncertainty assuming independent errors
        }
    )

    # Use bin centers for new x, y, z, t
    superobs["x"] = (superobs["x_bin"] + 0.5) * config.spatial_radius_x_meters
    superobs["y"] = (superobs["y_bin"] + 0.5) * config.spatial_radius_y_meters
    if config.spatial_radius_z is not None:
        superobs["z"] = (superobs["z_bin"] + 0.5) * config.spatial_radius_z
    superobs["t_seconds"] = (superobs["t_bin"] + 0.5) * config.temporal_radius_seconds
    superobs["time"] = pd.to_datetime(superobs["t_seconds"], unit="s")

    # Compute lat/lon from x/y
    transformer = wrf.get_wrf_reverse_proj_transformer(domain)
    lons, lats = transformer.transform(
        superobs["x"].to_numpy(), superobs["y"].to_numpy()
    )
    superobs["longitude"] = lons
    superobs["latitude"] = lats

    # Add downsampling info to each group
    def compute_downsampling_info(group):
        n_obs = len(group)
        time_spread = group["t_seconds"].max() - group["t_seconds"].min()
        centroid_x, centroid_y = group["x"].mean(), group["y"].mean()
        spatial_distances = np.sqrt(
            (group["x"] - centroid_x) ** 2 + (group["y"] - centroid_y) ** 2
        )
        spatial_spread = np.sqrt(np.mean(spatial_distances**2))

        return {
            "method": "grid_binning",
            "n_observations": n_obs,
            "time_spread_seconds": time_spread,
            "spatial_spread_meters": spatial_spread,
        }

    downsampling_info = (
        df_work.groupby(bin_cols)[["t_seconds", "x", "y"]]
        .apply(compute_downsampling_info)
        .reset_index(name="downsampling_info")
    )
    superobs = superobs.merge(downsampling_info, on=bin_cols)

    # Drop bin columns and working columns
    superobs = superobs.drop(
        columns=["t_seconds"] + [col for col in bin_cols if "_bin" in col]
    )

    return superobs
