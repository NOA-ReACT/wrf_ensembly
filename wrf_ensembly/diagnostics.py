"""Utilities for reading and computing statistics from DART filter diagnostics."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


def _find_copy_indices(
    fields: Iterable[str], copy_metadata: xr.DataArray
) -> dict[str, int]:
    """Find the index of each field name in the CopyMetaData variable."""
    indices = {}
    for field in fields:
        indices[field] = (copy_metadata == field).argmax().item()
    return indices


def read_obs_seq_nc(path: Path) -> pd.DataFrame:
    """
    Read a DART obs_seq NetCDF file (produced by obs_seq_to_netcdf) into a DataFrame.

    Reads all observation types present in the file. Returns a DataFrame with columns:
    obs, obs_variance, prior_mean, prior_spread, posterior_mean, posterior_spread,
    longitude, latitude, z, dart_qc, obs_type, timestamp.

    Args:
        path: Path to the NetCDF file (e.g. data/diagnostics/cycle_0.nc)

    Returns:
        DataFrame with one row per observation
    """
    ds = xr.open_dataset(path).load()

    # Fix fixed-length string fields
    ds["ObsTypesMetaData"] = ds["ObsTypesMetaData"].astype(str).str.strip()
    ds["QCMetaData"] = ds["QCMetaData"].astype(str).str.strip()
    ds["CopyMetaData"] = ds["CopyMetaData"].astype(str).str.strip()

    # CopyMetaData has some padding created with spaces that might not be constant every time
    # So we use some regex to replace multiple spaces with a single space
    ds["CopyMetaData"] = ds["CopyMetaData"].str.replace(r"\s+", " ", regex=True)

    # Build obs_type_id -> name mapping for all types present
    obs_type_ids = {}
    for i in range(ds.sizes.get("ObsTypes", 0)):
        name = ds["ObsTypesMetaData"].isel(ObsTypes=i).item()
        type_id = ds["ObsTypes"].isel(ObsTypes=i).item()
        obs_type_ids[type_id] = name

    # From CopyMetaData, find how many members we have and generate all the per_member value columns
    # The columns are called like `prior ensemble member     XX` and `posterior ensemble member     XX`
    # with a varying amount of spaces. XX is the member ID, starting from 1, no leading zeros.
    # We have fixed the spacing to use a single space instead of varying amounts above.
    max_member = (
        ds["CopyMetaData"].str.contains(r"^prior ensemble member \d+$").sum().item()
    )
    per_member_columns = {
        f"member_{i}_prior": f"prior ensemble member {i + 1}" for i in range(max_member)
    } | {
        f"member_{i}_posterior": f"posterior ensemble member {i + 1}"
        for i in range(max_member)
    }

    # Find copy indices for the fields we need
    column_mappings = {
        "obs": "observation",
        "obs_variance": "observation error variance",
        "prior_mean": "prior ensemble mean",
        "prior_spread": "prior ensemble spread",
        "posterior_mean": "posterior ensemble mean",
        "posterior_spread": "posterior ensemble spread",
    } | per_member_columns
    copy_indices = _find_copy_indices(column_mappings.values(), ds["CopyMetaData"])

    # Find DART QC index
    dart_qc_index = (ds["QCMetaData"] == "DART quality control").argmax().item()

    # "Wide" to long transform
    result = {}
    for col_name, copy_name in column_mappings.items():
        idx = copy_indices[copy_name]
        result[col_name] = ds["observations"].isel(copy=idx).values

    result["longitude"] = ds["location"].isel(locdim=0).values
    result["latitude"] = ds["location"].isel(locdim=1).values
    result["z"] = ds["location"].isel(locdim=2).values
    result["dart_qc"] = ds["qc"].isel(qc_copy=dart_qc_index).values
    result["timestamp"] = pd.to_datetime(ds["time"].values)

    # Observation error is given in variance, convert to stdev
    result["obs_std"] = np.sqrt(result["obs_variance"])

    # Map obs_type IDs to names
    obs_type_raw = ds["obs_type"].values
    result["obs_type"] = np.array(
        [obs_type_ids.get(int(t), f"unknown_{int(t)}") for t in obs_type_raw]
    )

    ds.close()
    return pd.DataFrame(result)


def compute_rank_histogram(
    df: pd.DataFrame, use_posterior: bool = False
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute a rank histogram from a diagnostics DataFrame.

    Ranks each observation within the actual ensemble member values.

    Args:
        df: DataFrame with columns obs, obs_variance, prior_mean, prior_spread,
            and member_X_prior / member_X_posterior columns for each ensemble member X
            (zero-indexed, no leading zeros), and optionally posterior_mean,
            posterior_spread.

    Returns:
        Tuple of (histogram, ranks, diagnostics_dict).
    """
    stage = "posterior" if use_posterior else "prior"

    member_cols = sorted(
        [c for c in df.columns if c.endswith(f"_{stage}") and c.startswith("member_")],
        key=lambda c: int(c.split("_")[1]),
    )
    if not member_cols:
        raise ValueError(f"No member columns found for stage '{stage}'.")

    ensemble = df[member_cols].values  # shape: (n_obs, n_members)
    n_members = ensemble.shape[1]

    obs = df["obs"].values
    obs_std = np.sqrt(df["obs_variance"].values)

    if use_posterior:
        ensemble_mean = df["posterior_mean"].values
        ensemble_spread = df["posterior_spread"].values
    else:
        ensemble_mean = df["prior_mean"].values
        ensemble_spread = df["prior_spread"].values

    # Rank each observation within its ensemble, with dithering to handle ties
    obs_dithered = obs + np.random.normal(0, obs_std)
    ranks = np.sum(ensemble < obs_dithered[:, np.newaxis], axis=1)

    hist, _ = np.histogram(ranks, bins=np.arange(n_members + 2) - 0.5)

    innovations = obs - ensemble_mean
    normalized_innovations = innovations / np.sqrt(ensemble_spread**2 + obs_std**2)

    diagnostics = {
        "n_obs": len(obs),
        "n_ensemble": n_members,
        "innovation_mean": np.mean(innovations),
        "innovation_std": np.std(innovations),
        "normalized_innov_mean": np.nanmean(normalized_innovations),
        "normalized_innov_std": np.nanstd(normalized_innovations),
        "correlation": np.corrcoef(obs, ensemble_mean)[0, 1],
        "rmse": np.sqrt(np.mean(innovations**2)),
    }

    return hist, ranks, diagnostics
