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

QC_VALIDATION_HOLDOUT = -1
"""
Observation is good quality but held out from DA for independent validation.
Negative value signals a system/configuration reason (not a source observation fault).
Used by stride thinning. The validation pipeline includes these; the DA pipeline does not.
"""


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

    # Both checks below are deliberately kept off `df.iterrows()`. Building a Series per
    # row costs ~30x more than the checks themselves, which matters because every
    # write_obs() pays for it - on a converted GRASP granule (~700k rows) that was 9
    # seconds of the 11 the whole conversion took.
    invalid_z_type = ~df["z_type"].isin(Z_TYPES).to_numpy()
    if invalid_z_type.any():
        pos = int(invalid_z_type.argmax())
        raise ValueError(
            f"Invalid z_type encountered at row {df.index[pos]}: {df['z_type'].iat[pos]}"
        )

    # Verify orig_coords fields
    for i, orig_coords in zip(df.index, df["orig_coords"].to_numpy()):
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

        indices = orig_coords["indices"]
        shape = orig_coords["shape"]
        names = orig_coords["names"]

        if not (len(indices) == len(names) == len(shape)):
            raise ValueError(
                f"orig_coords 'indices', 'shape', and 'names' must have the same length at row {i}"
            )
        for j in range(len(indices)):
            if not isinstance(indices[j], (int, np.integer)):
                raise ValueError(f"orig_coords 'indices' must be integers at row {i}")
            if not isinstance(shape[j], (int, np.integer)):
                raise ValueError(f"orig_coords 'shape' must be integers at row {i}")
            if not isinstance(names[j], (str, np.str_)):
                raise ValueError(f"orig_coords 'names' must be strings at row {i}")


def read_obs(path: Path | str) -> pd.DataFrame:
    """Read a WRF-Ensembly Observation data file into a pandas DataFrame."""

    df = pd.read_parquet(path)

    # Any empty metadata fields should be converted to pd.NA
    df["metadata"] = df["metadata"].apply(lambda x: pd.NA if x == "" else x)

    validate_schema(df)
    return df


def write_obs(df: pd.DataFrame, path: Path | str):
    """Write a pandas DataFrame to a WRF-Ensembly Observation data file."""

    validate_schema(df)
    df.to_parquet(path, index=False)
