"""Functions related to the binning of observations stored in DataFrames"""

import json

import numpy as np
import pandas as pd

from wrf_ensembly.observations.io import QC_VALIDATION_HOLDOUT


def parse_orig_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand orig_coords struct into flat columns named after each dimension.

    For a row with orig_coords = {names: ["profile", "height_bin"], indices: [42, 7]},
    this adds columns `_coord_profile=42` and `_coord_height_bin=7`.

    Columns are prefixed with `_coord_` so native grid dimension names (which can be
    `x`, `y`, `z`, `latitude`, ... for swath instruments like HARP2) never collide with
    the existing projected-coordinate columns on the dataframe.

    Returns a copy of df with the extra columns appended.
    """

    def _extract(row) -> dict:
        oc = row["orig_coords"]
        return {
            f"_coord_{name}": int(idx)
            for name, idx in zip(oc["names"], oc["indices"])
        }

    coord_cols = df.apply(_extract, axis=1, result_type="expand")
    return pd.concat([df, coord_cols], axis=1)


def _aggregate_group(
    group: pd.DataFrame,
    new_shape: tuple[int, ...],
    bin_indices: tuple[int, ...],
    dim_names: tuple[str, ...],
) -> pd.Series | None:
    """
    Collapse one bin group into a single superob row, preserving the full schema.

    Params:
        group: Should contain all observations inside the bin
        new_shape: Dimensions of the new grid
        bin_indices: The indices (x/y/z/...) of the observation in the **new** grid.
        dim_names: Native names of the binned dimensions, aligned with `new_shape`
            and `bin_indices`. The superob's coordinate names are these with a
            `_bin` suffix.

    Uncertainty model:
        instrument_err = rms(individual errors) / sqrt(n)   [reduces with n]
        repr_err       = std(values within group)           [does not reduce]
        total_err      = sqrt(instrument_err^2 + repr_err^2)

    Return:
        One row representing the superob or None if the given group has no data that pass the QC check
    """

    # Sanity check
    assert len(bin_indices) == len(new_shape)

    # Prefer good-QC observations; fall back to all observations with qc_flag=1
    good = group[group["qc_flag"] == 0]
    if not good.empty:
        agg_group = good
        qc = 0
    else:
        agg_group = group
        qc = 1

    n = len(agg_group)
    first = agg_group.iloc[0]

    # Deal with value and uncertainty
    mean_val = agg_group["value"].mean()
    instr_err = float(
        np.sqrt((agg_group["value_uncertainty"] ** 2).mean()) / np.sqrt(n)
    )
    repr_err = (
        float(agg_group["value"].std()) if n > 1 else float(first["value_uncertainty"])
    )
    total_err = float(np.sqrt(instr_err**2 + repr_err**2))

    # Compute centroid for location
    mean_lon = agg_group["longitude"].mean()
    mean_lat = agg_group["latitude"].mean()
    mean_x = agg_group["x"].mean()
    mean_y = agg_group["y"].mean()
    mean_z = agg_group["z"].mean()
    mean_time = agg_group["time"].mean()

    # Handle coordinates: describe the *new* binned grid. indices/shape/names are all
    # aligned with the binned dimensions (group_cols), with a `_bin` suffix on names.
    orig_coords = {
        "indices": np.array(bin_indices, dtype=np.int32),
        "shape": np.array(new_shape, dtype=np.int32),
        "names": np.array([name + "_bin" for name in dim_names], dtype=object),
    }

    # Metadata: Keep existing from the first observation, add superob information
    existing_meta = first["metadata"]
    meta = existing_meta if existing_meta else {}
    meta["superob"] = {
        "n_contributing": n,
        "repr_error": repr_err,
        "instrument_error": instr_err,
    }

    return pd.Series(
        {
            "instrument": first["instrument"],
            "quantity": first["quantity"],
            "time": mean_time,
            "longitude": mean_lon,
            "latitude": mean_lat,
            "x": mean_x,
            "y": mean_y,
            "z": mean_z,
            "z_type": first["z_type"],
            "value": mean_val,
            "value_uncertainty": total_err,
            "qc_flag": qc,
            "orig_coords": orig_coords,
            "orig_filename": first["orig_filename"],
            "metadata": meta,
        }
    )


def grid_bin(
    df: pd.DataFrame,
    hoz_bins: dict[str, int],
    vert_bins: dict[str, int],
) -> pd.DataFrame:
    """
    Bin a dataframe along its native instrument grid dimensions.

    Parameters:
        df: Observations for a single (instrument, quantity) pair.
        hoz_bins: Bin size per horizontal dimension, in native grid steps.
        vert_bins: Bin size per vertical dimension, in native grid steps.

    Returns:
        pd.DataFrame with the same schema as the input, one row per superob.
    """

    if df.empty:
        return df.copy()

    df = parse_orig_coords(df.copy())

    hoz_dim_names = list(hoz_bins.keys())
    vert_dim_names = list(vert_bins.keys())

    # Build group labels: floor-divide each index by its bin size.
    group_cols: list[str] = []
    for dim in hoz_dim_names:
        label = f"_grp_{dim}"
        bin_size = hoz_bins.get(dim, 1)
        df[label] = df[f"_coord_{dim}"] // bin_size
        group_cols.append(label)

    for dim in vert_dim_names:
        label = f"_grp_{dim}"
        bin_size = vert_bins.get(dim, 1)
        df[label] = df[f"_coord_{dim}"] // bin_size
        group_cols.append(label)

    new_shape = tuple(int(df[col].max()) + 1 for col in group_cols)
    dim_names = tuple(hoz_dim_names + vert_dim_names)

    def _agg(group):
        bin_indices = tuple(int(group[col].iloc[0]) for col in group_cols)
        return _aggregate_group(group, new_shape, bin_indices, dim_names)

    result = df.groupby(group_cols, group_keys=False).apply(_agg).reset_index(drop=True)

    return result


def time_bin(df: pd.DataFrame, bin_minutes: int, offset_minutes: int = 0) -> pd.DataFrame:
    """
    Bin observations into fixed-width UTC time windows using pandas time grouping.

    By default bins are left-aligned to UTC midnight (00:00, 01:00, ...). Use
    offset_minutes to shift boundaries — e.g. offset_minutes=-30 with bin_minutes=60
    produces bins from :30 to :30, centering each window on a full hour.

    Empty bins are dropped. The same uncertainty model as grid_bin is applied:
    instrument error reduces with sqrt(n), representativeness error is the std of
    values within the bin.

    Incompatible with grid_bin (spatial superobbing) — the two operations produce
    different orig_coords structures. After time_bin, orig_coords encodes a single
    "time_bin" dimension.

    Parameters:
        df: Observations for a single (instrument, quantity) pair.
        bin_minutes: Width of each time window in minutes.
        offset_minutes: Shift bin boundaries by this many minutes (default 0).

    Returns:
        pd.DataFrame with same schema, one row per non-empty bin.
    """
    if df.empty:
        return df.copy()

    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    bin_seconds = bin_minutes * 60
    offset_td = pd.Timedelta(minutes=offset_minutes)

    grouped = df.groupby(
        pd.Grouper(key="time", freq=f"{bin_minutes}min", offset=offset_td)
    )
    non_empty = [(ts, g) for ts, g in grouped if not g.empty]
    if not non_empty:
        return df.iloc[:0].copy()

    # Shift ts back by the offset to get an epoch-aligned integer index
    bin_indices = [
        int((ts - offset_td - epoch).total_seconds() // bin_seconds)
        for ts, _ in non_empty
    ]
    n_bins = max(bin_indices) + 1

    def _agg(group: pd.DataFrame, bin_idx: int) -> pd.Series:
        good = group[group["qc_flag"] == 0]
        agg_group = good if not good.empty else group
        qc = 0 if not good.empty else 1
        n = len(agg_group)
        first = agg_group.iloc[0]

        mean_val = agg_group["value"].mean()

        unc = agg_group["value_uncertainty"].dropna()
        instr_err = (
            float(np.sqrt((unc**2).mean()) / np.sqrt(len(unc))) if len(unc) > 0 else float("nan")
        )
        unc_val = first["value_uncertainty"]
        repr_err = (
            float(agg_group["value"].std())
            if n > 1
            else (float("nan") if pd.isna(unc_val) else float(unc_val))
        )
        components = [x**2 for x in (instr_err, repr_err) if not np.isnan(x)]
        total_err = float(np.sqrt(sum(components))) if components else float("nan")

        meta = dict(first["metadata"]) if first["metadata"] else {}
        meta["superob"] = {
            "n_contributing": n,
            "repr_error": repr_err,
            "instrument_error": instr_err,
        }

        return pd.Series(
            {
                "instrument": first["instrument"],
                "quantity": first["quantity"],
                "time": agg_group["time"].mean(),
                "longitude": agg_group["longitude"].mean(),
                "latitude": agg_group["latitude"].mean(),
                "x": agg_group["x"].mean(),
                "y": agg_group["y"].mean(),
                "z": agg_group["z"].mean(),
                "z_type": first["z_type"],
                "value": mean_val,
                "value_uncertainty": total_err,
                "qc_flag": qc,
                "orig_coords": {
                    "indices": np.array([bin_idx], dtype=np.int32),
                    "shape": np.array([n_bins], dtype=np.int32),
                    "names": np.array(["time_bin"], dtype=object),
                },
                "orig_filename": first["orig_filename"],
                "metadata": meta,
            }
        )

    results = [_agg(g, idx) for (_, g), idx in zip(non_empty, bin_indices)]
    return pd.DataFrame(results).reset_index(drop=True)


def stride_thin(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Thin observations by keeping every N-th good-QC observation for DA.

    Good-QC observations (qc_flag == 0) are sorted by time, then every n-th
    is kept with qc_flag = 0 (for DA). The rest are marked
    qc_flag = QC_VALIDATION_HOLDOUT so they remain available for validation
    but are excluded from DA. Bad-QC observations are left unchanged.

    Parameters:
        df: Observations for a single (instrument, quantity) pair.
        n:  Keep every n-th good-QC observation. n=1 means keep all (no-op for DA).

    Returns:
        pd.DataFrame with same schema, same number of rows.
    """
    df = df.copy()

    good_mask = df["qc_flag"] == 0
    good_idx = df[good_mask].sort_values("time").index

    keep_idx = good_idx[::n]
    holdout_idx = good_idx.difference(keep_idx)

    df.loc[holdout_idx, "qc_flag"] = QC_VALIDATION_HOLDOUT

    return df
