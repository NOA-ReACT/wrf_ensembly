"""Contains downsampling (super-orbing) methods for reducing the resolution of observations."""

import duckdb
import numpy as np
import pandas as pd

from wrf_ensembly import wrf
from wrf_ensembly.config import DomainControlConfig, SuperobbingConfig
from wrf_ensembly.console import logger


def greedy_maxmin_clustering(
    points: np.ndarray,
    min_distances: dict,
    active_dims: list | None = None,
    max_centers: int | None = None,
) -> np.ndarray:
    """
    Greedy farthest-point insertion for maxmin clustering with per-coordinate distances.

    Args:
        points: (N, D) array of points
        min_distances: dict mapping dimension index to minimum distance e.g., {0: 1.0, 2: 2.0}
        active_dims: list of dimension indices to use (default: all)
        max_centers: optional limit on number of centers

    Returns:
        (K, D) array of cluster centers
    """
    n_points, n_dims = points.shape

    if active_dims is None:
        active_dims = list(range(n_dims))

    # Extract active dimensions and create scaling array
    points_active = points[:, active_dims]
    n_active_dims = len(active_dims)

    # Create scaling vector for dimensions
    scaling = np.ones(n_active_dims)
    for i, dim in enumerate(active_dims):
        if dim in min_distances:
            scaling[i] = 1.0 / min_distances[dim]

    logger.info(f"Starting maxmin clustering with {n_points} points")
    logger.info(f"Active dimensions: {active_dims}, distances: {min_distances}")

    # Initialize with first point as center
    centers_idx = [0]

    # Track minimum distance from each point to nearest center
    # Initialize with infinity so first update captures all points
    min_dists_to_centers = np.full(n_points, np.inf, dtype=np.float64)

    iteration = 0
    while True:
        iteration += 1

        # Get the newly added center (last in list)
        new_center = points_active[centers_idx[-1]]

        # Compute squared distances from all points to the NEW center only (avoid sqrt)
        diffs = points_active - new_center  # (N, D)
        scaled_diffs = diffs * scaling  # Vectorized scaling
        distances_to_new_sq = np.sum(scaled_diffs * scaled_diffs, axis=1)  # (N,)

        # Update minimum squared distances: for each point, keep minimum
        min_dists_to_centers = np.minimum(min_dists_to_centers, distances_to_new_sq)

        # Find farthest point from any center
        farthest_idx = np.argmax(min_dists_to_centers)
        farthest_dist_sq = min_dists_to_centers[farthest_idx]

        if farthest_dist_sq < 1.0:  # In scaled space, threshold is 1.0 (squared)
            logger.info(
                f"Stopping: max scaled distance {np.sqrt(farthest_dist_sq):.4f} < 1.0"
            )
            break

        if max_centers and len(centers_idx) >= max_centers:
            logger.info(f"Stopping: reached max_centers limit ({max_centers})")
            break

        centers_idx.append(int(farthest_idx))

        if len(centers_idx) % 100 == 0:
            logger.info(
                f"Found {len(centers_idx)} centers, farthest scaled distance: {np.sqrt(farthest_dist_sq):.4f}"
            )

    logger.info(f"Final: {len(centers_idx)} cluster centers in {iteration} iterations")
    return points[centers_idx]


def assign_cluster_ids_duckdb_minmax(
    con: duckdb.DuckDBPyConnection,
    config: SuperobbingConfig,
    instrument: str | None = None,
    quantity: str | None = None,
):
    """Assign cluster IDs to observations in the database using maxmin clustering."""

    # Build filter clause
    where_clauses = []
    if instrument is not None:
        where_clauses.append(f"instrument = '{instrument}'")
    if quantity is not None:
        where_clauses.append(f"quantity = '{quantity}'")

    # Only consider points that have values
    where_clauses.append("value IS NOT NULL")

    where_clause = ""
    if where_clauses:
        where_clause = "WHERE " + " AND ".join(where_clauses)

    # Fetch points from the database
    query = f"""
    SELECT
        rowid,
        x,
        y,
        z,
        EXTRACT(EPOCH FROM time) as time_epoch
    FROM observations
    {where_clause}
    """
    df_points = con.execute(query).fetchdf()

    points_array = df_points[["x", "y", "z", "time_epoch"]].to_numpy()
    # Dimension indices: 0=x, 1=y, 2=z, 3=time
    min_distances = {
        0: config.spatial_radius_x_meters,
        1: config.spatial_radius_y_meters,
    }
    if config.spatial_radius_z is not None:
        min_distances[2] = config.spatial_radius_z
    if config.temporal_radius_seconds is not None:
        min_distances[3] = config.temporal_radius_seconds

    # First find the centers
    centers = greedy_maxmin_clustering(
        points_array,
        min_distances,
        active_dims=list(min_distances.keys()),
    )
    # Then refine them with Lloyd's algorithm
    centers = lloyd_refinement(
        points_array,
        centers,
        min_distances,
        active_dims=list(min_distances.keys()),
    )

    n_points = len(points_array)
    batch_size = 10000
    nearest_center_idx = np.zeros(n_points, dtype=np.int32)

    for start_idx in range(0, n_points, batch_size):
        end_idx = min(start_idx + batch_size, n_points)
        batch = points_array[start_idx:end_idx]

        # Compute squared distances for this batch
        diffs = batch[:, np.newaxis, :] - centers[np.newaxis, :, :]
        distances_sq = np.sum(diffs * diffs, axis=2)
        nearest_center_idx[start_idx:end_idx] = np.argmin(distances_sq, axis=1)

    df_points["cluster_id"] = np.char.add("cluster_", nearest_center_idx.astype(str))

    con.register("temp_points", df_points)
    update_query = """
        UPDATE observations
        SET cluster_id = temp_points.cluster_id
        FROM temp_points
        WHERE observations.rowid = temp_points.rowid
        """

    con.execute(update_query)


def lloyd_refinement(
    points: np.ndarray,
    centers: np.ndarray,
    min_distances: dict,
    active_dims: list,
    max_iterations: int = 10,
    tolerance: float = 1e-6,
    batch_size: int = 100000,
) -> np.ndarray:
    """
    Refine cluster centers using Lloyd's algorithm (k-means refinement).
    Reassigns points and recompute centers until convergence.

    https://en.wikipedia.org/wiki/Lloyd%27s_algorithm

    Args:
        points: (N, D) array of points
        centers: (K, D) initial cluster centers
        min_distances: dict mapping dimension index to minimum distance
        active_dims: list of active dimensions
        max_iterations: maximum refinement iterations
        tolerance: convergence threshold for center movement
        batch_size: number of points to process at once (controls memory usage)

    Returns:
        (K, D) refined cluster centers
    """
    points_active = points[:, active_dims]
    centers_active = centers[:, active_dims]
    n_points = len(points_active)
    n_centers = len(centers_active)

    # Precompute scaling array
    scaling = np.ones(len(active_dims))
    for i, dim in enumerate(active_dims):
        if dim in min_distances:
            scaling[i] = 1.0 / min_distances[dim]

    logger.info(
        f"Starting Lloyd's refinement with {n_points} points and {n_centers} centers..."
    )

    # Scale points once for all iterations
    points_scaled = points_active * scaling

    prev_assignments = None

    for iteration in range(max_iterations):
        # Scale centers for this iteration
        centers_scaled = centers_active * scaling

        # Optimized batched computation with chunked center processing
        assignments = np.zeros(n_points, dtype=np.int32)

        # For large center sets, process centers in chunks to improve cache locality
        center_chunk_size = 500  # Process 500 centers at a time

        for start_idx in range(0, n_points, batch_size):
            end_idx = min(start_idx + batch_size, n_points)
            batch = points_scaled[start_idx:end_idx]
            batch_size_actual = end_idx - start_idx

            # Initialize with infinity for this batch
            min_distances_batch = np.full(batch_size_actual, np.inf, dtype=np.float32)
            best_centers = np.zeros(batch_size_actual, dtype=np.int32)

            # Process centers in chunks for better cache performance
            for center_start in range(0, n_centers, center_chunk_size):
                center_end = min(center_start + center_chunk_size, n_centers)
                centers_chunk = centers_scaled[center_start:center_end]

                # Compute squared distances for this chunk (float32 for speed)
                # Shape: (batch_size, n_centers_in_chunk, n_dims)
                diffs = batch[:, np.newaxis, :].astype(np.float32) - centers_chunk[
                    np.newaxis, :, :
                ].astype(np.float32)
                distances_sq = np.sum(diffs * diffs, axis=2)

                # Update minimum distances and best centers for this chunk
                chunk_min_dists = np.min(distances_sq, axis=1)
                chunk_best_centers = np.argmin(distances_sq, axis=1) + center_start

                # Update global minimum for this batch
                update_mask = chunk_min_dists < min_distances_batch
                min_distances_batch[update_mask] = chunk_min_dists[update_mask]
                best_centers[update_mask] = chunk_best_centers[update_mask]

            assignments[start_idx:end_idx] = best_centers

        # Check for early convergence - if assignments don't change much
        if prev_assignments is not None:
            n_changed = np.sum(assignments != prev_assignments)
            change_pct = 100.0 * n_changed / n_points
            if change_pct < 0.01:  # Less than 0.01% changed
                logger.info(
                    f"  Early stopping: only {n_changed} points ({change_pct:.4f}%) changed assignments"
                )
                break

        prev_assignments = assignments.copy()

        # Recompute centers using vectorized operations
        new_centers_active = np.zeros_like(centers_active)
        counts = np.bincount(assignments, minlength=n_centers)

        # Sum points for each cluster (use unscaled points)
        for d in range(len(active_dims)):
            new_centers_active[:, d] = np.bincount(
                assignments, weights=points_active[:, d], minlength=n_centers
            )

        # Average (handle empty clusters by keeping old centers)
        mask = counts > 0
        new_centers_active[mask] /= counts[mask, np.newaxis]
        new_centers_active[~mask] = centers_active[~mask]

        # Check convergence (in original space)
        movement = np.linalg.norm(new_centers_active - centers_active)
        n_changed_str = (
            f", {np.sum(assignments != prev_assignments) if iteration > 0 else 'N/A'} points changed"
            if iteration > 0
            else ""
        )
        logger.info(
            f"  Iteration {iteration + 1}: center movement = {movement:.6f}{n_changed_str}"
        )

        if movement < tolerance:
            logger.info(f"Converged after {iteration + 1} iterations")
            break

        centers_active = new_centers_active

    # Update full center array with refined coordinates
    centers_refined = centers.copy()
    centers_refined[:, active_dims] = centers_active

    return centers_refined


def assign_cluster_ids_duckdb(
    con: duckdb.DuckDBPyConnection,
    config: SuperobbingConfig,
    instrument: str | None = None,
    quantity: str | None = None,
) -> None:
    """
    Assign cluster IDs to observations in the database based on spatio-temporal bins.

    This function updates the observations table in-place, setting the cluster_id column
    for all observations (or filtered by instrument/quantity). The cluster_id groups
    observations that will be merged together during superobbing.

    Args:
        con: DuckDB connection with observations table (must have cluster_id column)
        config: Configuration specifying the clustering radii
        instrument: Optional instrument filter. If None, processes all instruments.
        quantity: Optional quantity filter. If None, processes all quantities.
    """

    # Build filter clause
    where_clauses = []
    if instrument is not None:
        where_clauses.append(f"instrument = '{instrument}'")
    if quantity is not None:
        where_clauses.append(f"quantity = '{quantity}'")

    where_clause = ""
    if where_clauses:
        where_clause = "WHERE " + " AND ".join(where_clauses)

    # Build cluster_id expression based on whether z binning is used
    if config.spatial_radius_z is not None:
        cluster_id_expr = f"""
            CONCAT_WS('_',
                CAST(CAST(x / {config.spatial_radius_x_meters} AS INTEGER) AS VARCHAR),
                CAST(CAST(y / {config.spatial_radius_y_meters} AS INTEGER) AS VARCHAR),
                CAST(CAST(z / {config.spatial_radius_z} AS INTEGER) AS VARCHAR),
                CAST(CAST(EXTRACT(EPOCH FROM time) / {config.temporal_radius_seconds} AS INTEGER) AS VARCHAR)
            )
        """
    else:
        cluster_id_expr = f"""
            CONCAT_WS('_',
                CAST(CAST(x / {config.spatial_radius_x_meters} AS INTEGER) AS VARCHAR),
                CAST(CAST(y / {config.spatial_radius_y_meters} AS INTEGER) AS VARCHAR),
                CAST(CAST(EXTRACT(EPOCH FROM time) / {config.temporal_radius_seconds} AS INTEGER) AS VARCHAR)
            )
        """

    # Update the cluster_id column
    update_query = f"""
        UPDATE observations
        SET cluster_id = {cluster_id_expr}
        {where_clause}
    """

    con.execute(update_query)


def _merge_nearby_superobs(
    superobs: pd.DataFrame,
    spatial_radius_x_meters: float,
    spatial_radius_y_meters: float,
) -> pd.DataFrame:
    """
    Merge superobservations that are closer than the spatial radii.

    This post-processing step ensures that output superobservations respect the minimum
    separation distances. When superobservations are closer than the specified radii,
    they are merged together by averaging their properties.

    Args:
        superobs: DataFrame of superobservations with x, y coordinates
        spatial_radius_x_meters: Minimum separation in x direction
        spatial_radius_y_meters: Minimum separation in y direction

    Returns:
        pd.DataFrame: Merged superobservations with minimum separation guaranteed
    """

    if superobs.empty:
        return superobs

    # Sort by x, y to ensure deterministic order
    superobs = superobs.sort_values(by=["x", "y"]).reset_index(drop=True).copy()

    # Use a simple clustering approach: iteratively find and merge closest pairs
    # that are too close together
    while True:
        # Compute pairwise distances
        x_vals = superobs["x"].values
        y_vals = superobs["y"].values
        x_diff = np.abs(x_vals.reshape(-1, 1) - x_vals.reshape(1, -1))
        y_diff = np.abs(y_vals.reshape(-1, 1) - y_vals.reshape(1, -1))

        # Check if distances respect the minimum separation
        too_close_x = x_diff < spatial_radius_x_meters
        too_close_y = y_diff < spatial_radius_y_meters
        too_close = too_close_x & too_close_y

        # Exclude self-comparisons
        np.fill_diagonal(too_close, False)

        # Find if there are any pairs that are too close
        close_pairs = np.where(too_close)
        if len(close_pairs[0]) == 0:
            # No pairs, we're done
            break

        # Sort pairs deterministically to ensure consistent results
        # Always pick the lexicographically smallest (i, j) pair where i < j
        pairs = list(zip(close_pairs[0], close_pairs[1]))
        pairs = [(min(a, b), max(a, b)) for a, b in pairs]
        pairs = sorted(set(pairs))  # Remove duplicates and sort

        # Merge the first pair found
        i, j = pairs[0]

        # Merge observation j into observation i
        row_i = superobs.iloc[i]
        row_j = superobs.iloc[j]

        # Get numeric columns to average (excluding metadata)
        numeric_cols = superobs.select_dtypes(include=[np.number]).columns

        merged_row = row_i.copy()
        for col in numeric_cols:
            # Longitude and latitude will be recomputed outside this function
            if col not in ["longitude", "latitude"]:
                # Skip if both values are null
                if pd.isna(row_i[col]) and pd.isna(row_j[col]):
                    merged_row[col] = np.nan
                # Use the non-null value if only one is null
                elif pd.isna(row_i[col]):
                    merged_row[col] = row_j[col]
                elif pd.isna(row_j[col]):
                    merged_row[col] = row_i[col]
                # Both have values, compute weighted average
                else:
                    # Weight by n_observations if available, otherwise equal weight
                    if "downsampling_info" in superobs.columns:
                        n_i = row_i["downsampling_info"].get("n_observations", 1)
                        n_j = row_j["downsampling_info"].get("n_observations", 1)
                        merged_row[col] = (row_i[col] * n_i + row_j[col] * n_j) / (
                            n_i + n_j
                        )
                    else:
                        merged_row[col] = (row_i[col] + row_j[col]) / 2

        # Update downsampling info
        if "downsampling_info" in superobs.columns:
            info_i = row_i["downsampling_info"]
            info_j = row_j["downsampling_info"]
            merged_row["downsampling_info"] = {
                "method": "grid_binning_merged",
                "n_observations": info_i.get("n_observations", 0)
                + info_j.get("n_observations", 0),
                "time_spread_seconds": max(
                    info_i.get("time_spread_seconds", 0),
                    info_j.get("time_spread_seconds", 0),
                ),
                "spatial_spread_meters": -1.0,  # TODO: recalculate
            }

        # Keep all cluster IDs that were merged
        merged_row["cluster_id"] = f"{row_i['cluster_id']}+{row_j['cluster_id']}"

        # Remove row j and update row i
        # Important: update row i BEFORE dropping j to maintain correct indexing
        superobs.iloc[i] = merged_row
        superobs = superobs.drop(superobs.index[j]).reset_index(drop=True)

    return superobs


def compute_superobs_from_clusters(
    con: duckdb.DuckDBPyConnection,
    config: SuperobbingConfig,
    domain: DomainControlConfig,
    instrument: str | None = None,
    quantity: str | None = None,
) -> pd.DataFrame:
    """
    Compute superobservations from pre-assigned cluster IDs in the database.

    This function aggregates observations by their cluster_id column. The cluster_id
    must be assigned beforehand using assign_cluster_ids_duckdb().

    After aggregation, a merging step ensures that superobservations respect the minimum
    separation distances specified in the config.

    The output uncertainty assumes independent errors.

    Args:
        con: DuckDB connection with observations table (must have cluster_id assigned)
        config: Configuration specifying the clustering radii (needed for merge step)
        domain: Domain configuration for coordinate transformations
        instrument: Optional instrument filter. If None, processes all instruments.
        quantity: Optional quantity filter. If None, processes all quantities.

    Returns:
        pd.DataFrame: DataFrame containing the superobservations with cluster_id column
    """
    # Build filter clause
    where_clauses = ["cluster_id IS NOT NULL"]
    if instrument is not None:
        where_clauses.append(f"instrument = '{instrument}'")
    if quantity is not None:
        where_clauses.append(f"quantity = '{quantity}'")

    where_clause = "WHERE " + " AND ".join(where_clauses)

    # Main aggregation query - group by cluster_id
    superobs_query = f"""
    SELECT
        -- Cluster identifier
        cluster_id,

        -- Keep first of categorical columns
        ANY_VALUE(instrument) as instrument,
        ANY_VALUE(quantity) as quantity,
        ANY_VALUE(z_type) as z_type,
        ANY_VALUE(qc_flag) as qc_flag,
        ANY_VALUE(orig_coords) as orig_coords,
        ANY_VALUE(orig_filename) as orig_filename,
        ANY_VALUE(metadata) as metadata,

        -- Average coordinates (represents where observations actually are)
        AVG(x) as x,
        AVG(y) as y,
        AVG(z) as z,
        AVG(time) as time,

        -- Aggregated values
        AVG(value) as value,
        SQRT(SUM(value_uncertainty * value_uncertainty) / COUNT(*)) as value_uncertainty,

        -- Downsampling metadata
        COUNT(*) as n_observations,
        MAX(EXTRACT(EPOCH FROM time)) - MIN(EXTRACT(EPOCH FROM time)) as time_spread_seconds

    FROM observations
    {where_clause}
    GROUP BY cluster_id
    """

    # Execute the query and get the result
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
    superobs = superobs.drop(columns=["n_observations", "time_spread_seconds"])

    # Merge nearby superobservations to ensure minimum separation
    # This is required because grid binning doesn't guarantee exact separation
    superobs = _merge_nearby_superobs(
        superobs,
        config.spatial_radius_x_meters,
        config.spatial_radius_y_meters,
    )

    # Compute lat/lon from x/y using the transformer
    transformer = wrf.get_wrf_reverse_proj_transformer(domain)
    lons, lats = transformer.transform(
        superobs["x"].to_numpy(), superobs["y"].to_numpy()
    )
    superobs["longitude"] = lons
    superobs["latitude"] = lats

    return superobs
