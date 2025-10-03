"""Contains downsampling (super-orbing) methods for reducing the resolution of observations."""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from wrf_ensembly import wrf
from wrf_ensembly.config import DomainControlConfig, SuperorbingConfig


def superorb_dbscan(
    df: pd.DataFrame, config: SuperorbingConfig, domain: DomainControlConfig
) -> pd.DataFrame:
    """
    Does spatio-temporal clustering and uncertainty-weighted averaging to create superobservations.

    The output uncertainity assumes independent errors, which is probably not true. Or maybe it is.
    Who knows.

    Args:
        df: Input dataframe with columns: x, y, z, time, value, value_uncertainty
        Additional columns are preserved in the output
    config: Configuration specifying the clustering radii, see `config.py`

    Returns:
    --------
    The resampled dataframe with same schema as input, plus additional columns:
        - obs_count: number of observations that went into each superob
        - time_spread_seconds: temporal spread of observations in each superob
        - spatial_spread_meters: spatial spread of observations in each superob
    """

    if df.empty:
        return df.copy()
    required_cols = ["x", "y", "z", "time", "value", "value_uncertainty"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df_work = df.copy()

    # Convert time to seconds since epoch for clustering
    df_work["t_seconds"] = df_work["time"].apply(lambda x: x.timestamp())

    # Build the feature matrix for clustering
    features = []
    feature_names = []

    features.extend(
        [  # Normalize by radius, then we can use eps=1.0 in DBSCAN
            df_work["x"].to_numpy() / config.spatial_radius_x_meters,
            df_work["y"].to_numpy() / config.spatial_radius_y_meters,
            df_work["t_seconds"].to_numpy() / config.temporal_radius_seconds,
        ]
    )
    feature_names.extend(["x_norm", "y_norm", "t_norm"])

    if config.spatial_radius_z is not None:
        features.append(df_work["z"].to_numpy() / config.spatial_radius_z)
        feature_names.append("z_norm")

    # Stack features into matrix (n_obs x n_features)
    feature_matrix = np.column_stack(features)

    # Perform DBSCAN clustering
    # eps=1.0 because we normalized by the desired radii
    # min_samples=1 ensures every observation gets assigned to a cluster
    clustering = DBSCAN(eps=0.5, min_samples=5, metric="euclidean")
    cluster_labels = clustering.fit_predict(feature_matrix)

    df_work["cluster"] = cluster_labels

    # Group by clusters and compute superobs
    grouped = df_work.groupby("cluster")

    def compute_superob(group):
        """Compute uncertainty-weighted superobservation for a cluster"""
        n_obs = len(group)

        if n_obs == 1:
            # Single observation - just copy it
            row = group.iloc[0]
            result = row.to_dict()
            result["downsampling_info"] = {
                "method": "dbscan",
                "n_observations": 1,
                "time_spread_seconds": 0.0,
                "spatial_spread_meters": 0.0,
            }
            # Remove working columns
            result.pop("t_seconds", None)
            result.pop("cluster", None)
            return pd.DataFrame([result])

        # For any observations without uncertainty, we set it to '1' because that is
        # a neutral value for weighting
        group["uncertainty"] = group["value_uncertainty"].fillna(1.0)

        # Multiple observations - compute uncertainty-weighted averages
        # Weights are inverse variance
        weights = 1.0 / (group["uncertainty"] ** 2)
        total_weight = weights.sum()

        weighted_value = np.average(group["value"], weights=weights)
        weighted_x = np.average(group["x"], weights=weights)
        weighted_y = np.average(group["y"], weights=weights)
        weighted_z = np.average(group["z"], weights=weights)

        # Use median instead of mean for time to reduce sensitivity to outliers
        representative_time = group["time"].median()

        # Combined uncertainty, assuming independent observations, ha ha ha :(
        combined_uncertainty = 1.0 / np.sqrt(total_weight)

        # Compute spread statistics for reference
        time_spread = group["t_seconds"].max() - group["t_seconds"].min()
        centroid_x, centroid_y = weighted_x, weighted_y
        spatial_distances = np.sqrt(
            (group["x"] - centroid_x) ** 2 + (group["y"] - centroid_y) ** 2
        )
        spatial_spread = np.sqrt(np.mean(spatial_distances**2))

        # These statistics are going inside the `downsampling_info` struct column
        downsampling_info = {
            "method": "dbscan",
            "n_observations": n_obs,
            "time_spread_seconds": time_spread,
            "spatial_spread_meters": spatial_spread,
        }

        # Handle other columns by taking the most common value or first value
        result = {
            "x": weighted_x,
            "y": weighted_y,
            "z": weighted_z,
            "t": representative_time,
            "value": weighted_value,
            "uncertainty": combined_uncertainty,
            "downsampling_info": downsampling_info,
        }
        extra_cols = set(group.columns) - {
            "x",
            "y",
            "z",
            "t",
            "value",
            "uncertainty",
            "t_seconds",
            "cluster",
        }
        for col in extra_cols:
            result[col] = group[col].iloc[0]

        return pd.DataFrame([result])

    # Apply superob computation to each cluster, then convert to a DataFrame
    result_df = grouped.apply(compute_superob).reset_index(drop=True)

    return result_df


def superorb_grid_binning(
    df: pd.DataFrame, config: SuperorbingConfig, domain: DomainControlConfig
) -> pd.DataFrame:
    """
    Does spatio-temporal binning to create superobservations.

    The output uncertainity assumes independent errors, which is probably not true. Or maybe it is.
    Who knows.

    Args:
        df: Input dataframe with columns: x, y, z, time, value, value_uncertainty
        Additional columns are preserved in the output
    config: Configuration specifying the clustering radii, see `config.py`
    Returns:
    --------
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
