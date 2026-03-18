"""Functions related to the binning of observations stored in DataFrames"""

import json

import numpy as np
import pandas as pd


def parse_orig_coords(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand orig_coords struct into flat columns named after each dimension.

    For a row with orig_coords = {names: ["profile", "height_bin"], indices: [42, 7]},
    this adds columns `profile=42` and `height_bin=7`.

    Returns a copy of df with the extra columns appended.
    """

    def _extract(row) -> dict:
        oc = row["orig_coords"]
        return {name: int(idx) for name, idx in zip(oc["names"], oc["indices"])}

    coord_cols = df.apply(_extract, axis=1, result_type="expand")
    return pd.concat([df, coord_cols], axis=1)


def _aggregate_group(
    group: pd.DataFrame, new_shape: tuple[int, ...], bin_indices: tuple[int, ...]
) -> pd.Series | None:
    """
    Collapse one bin group into a single superob row, preserving the full schema.

    Params:
        group: Should contain all observations inside the bin
        new_shape: Dimensions of the new grid
        bin_indices: The indices (x/y/z/...) of the observation in the **new** grid.

    Uncertainty model:
        instrument_err = rms(individual errors) / sqrt(n)   [reduces with n]
        repr_err       = std(values within group)           [does not reduce]
        total_err      = sqrt(instrument_err^2 + repr_err^2)

    Return:
        One row representing the superob or None if the given group has no data that pass the QC check
    """

    # Sanity check
    assert len(bin_indices) == len(new_shape)

    # Remove any observation with bad QC
    group = group[group["qc_flag"] == 0]
    if group.empty:
        return None

    n = len(group)
    first = group.iloc[0]

    # Deal with value and uncertainty
    mean_val = group["value"].mean()
    instr_err = float(np.sqrt((group["value_uncertainty"] ** 2).mean()) / np.sqrt(n))
    repr_err = (
        float(group["value"].std()) if n > 1 else float(first["value_uncertainty"])
    )
    total_err = float(np.sqrt(instr_err**2 + repr_err**2))

    # Compute centroid for location
    mean_lon = group["longitude"].mean()
    mean_lat = group["latitude"].mean()
    mean_x = group["x"].mean()
    mean_y = group["y"].mean()
    mean_z = group["z"].mean()
    mean_time = group["time"].mean()

    # QC passes by default because we only include good observations
    qc = 0

    # Handle coordinates: use the `bin_indices` and `new_shape`, append `_bin` to all names
    orig_coords = {
        "indices": np.array(bin_indices, dtype=np.int32),
        "shape": np.array(new_shape, dtype=np.int32),
        "names": np.array(
            [name + "_bin" for name in first["orig_coords"]["names"].tolist()],
            dtype=object,
        ),
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
        df[label] = df[dim] // bin_size
        group_cols.append(label)

    for dim in vert_dim_names:
        label = f"_grp_{dim}"
        bin_size = vert_bins.get(dim, 1)
        df[label] = df[dim] // bin_size
        group_cols.append(label)

    new_shape = tuple(int(df[col].max()) + 1 for col in group_cols)

    def _agg(group):
        bin_indices = tuple(int(group[col].iloc[0]) for col in group_cols)
        return _aggregate_group(group, new_shape, bin_indices)

    result = (
        df.groupby(group_cols, group_keys=False)
        .apply(_agg)
        .dropna(how="all")
        .reset_index(drop=True)
    )

    return result
