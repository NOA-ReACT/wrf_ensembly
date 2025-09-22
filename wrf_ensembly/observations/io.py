"""Handles reading and writing the WRF-Ensembly Observation data files."""

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "instrument",
    "quantity",
    "time",
    "longitude",
    "latitude",
    "z",
    "z_type",
    "value",
    "value_uncertainty",
    "qc_flag",
    "orig_coords",
    "orig_filename",
    "metadata",
]

Z_TYPES = ["surface", "pressure", "height", "model_level", "columnar"]


def validate_schema(df: pd.DataFrame):
    """
    Checks the following for the input dataframe:
    - All required columns are present
    - z_type values are valid
    - orig_coords is a dictionary with keys 'indices', 'shape', and 'names'
    - orig_coords 'indices', 'shape', and 'names' have the same length
    - orig_coords 'indices' are integers
    - orig_coords 'shape' are integers
    - orig_coords 'names' are strings
    Throws a ValueError if any checks fail.

    Args:
        df: The dataframe to check
    """

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Verify z_type values, orig_coords fields
    for i, row in df.iterrows():
        if row.z_type not in Z_TYPES:
            raise ValueError(f"Invalid z_type encountered at row {i}: {row.z_type}")

        orig_coords = row.orig_coords
        if not isinstance(orig_coords, dict):
            raise ValueError(f"orig_coords must be a dictionary at row {i}")

        if (
            "indices" not in orig_coords
            or "shape" not in orig_coords
            or "names" not in orig_coords
        ):
            raise ValueError(
                f"orig_coords must contain 'indices', 'shape', and 'names' keys at row {i}"
            )

        if (
            len(orig_coords["indices"])
            != len(orig_coords["names"])
            != len(orig_coords["shape"])
        ):
            raise ValueError(
                f"orig_coords 'indices', 'shape', and 'names' must have the same length at row {i}"
            )
        orig_coord_len = len(orig_coords["indices"])
        for j in range(orig_coord_len):
            if not isinstance(orig_coords["indices"][j], (int, np.integer)):
                print(type(orig_coords["indices"][j]))
                raise ValueError(f"orig_coords 'indices' must be integers at row {i}")
            if not isinstance(orig_coords["shape"][j], (int, np.integer)):
                raise ValueError(f"orig_coords 'shape' must be integers at row {i}")
            if not isinstance(orig_coords["names"][j], (str, np.str_)):
                raise ValueError(f"orig_coords 'names' must be strings at row {i}")


def read_obs(path: Path | str) -> pd.DataFrame:
    """Read a WRF-Ensembly Observation data file into a pandas DataFrame."""

    df = pd.read_parquet(path)
    validate_schema(df)
    return df


def write_obs(df: pd.DataFrame, path: Path | str):
    """Write a pandas DataFrame to a WRF-Ensembly Observation data file."""

    validate_schema(df)
    df.to_parquet(path, index=False)
