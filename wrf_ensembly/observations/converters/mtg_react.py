"""Converter for MTG-REACT dust optical depth retrievals.

These are XGBoost predictions of EarthCARE ATLID dust optical depth at 355nm, driven by
MTG FCI radiances plus ERA5/ERA5-Land fields. Each parquet file is one 6-hourly snapshot,
flattened to one row per land pixel of a regular 0.05 degree lat/lon grid. Besides the
prediction, the files carry the ~90 model input features, which we don't need.
"""

import re
from pathlib import Path

import click
import numpy as np
import pandas as pd

from wrf_ensembly.observations import io as obs_io

PREDICTION_COLUMN = "EARTHCARE_DOD_a355_prediction"

QC_COLUMNS = [
    "MTCM_quality_illumination",
    "MTCM_quality_mtg_parameters",
    "MTCM_quality_nwp_parameters",
    "MTCM_quality_overall_processing",
]
"""MTG cloud mask quality flags, carried into the metadata for later filtering."""

# Every file sits on the same regular grid, but each only lists the (land, cloud-free)
# pixels it retrieved, so the extent of an individual file varies. orig_coords indexes
# into the full grid below so that all files share one shape. The extent is the union
# of all files of the B072 batch, padded out to whole degrees.
GRID_RESOLUTION = 0.05
GRID_LAT_MIN = 4.0
GRID_LAT_MAX = 42.0
GRID_LON_MIN = -18.0
GRID_LON_MAX = 60.0
GRID_SHAPE = (
    round((GRID_LAT_MAX - GRID_LAT_MIN) / GRID_RESOLUTION) + 1,
    round((GRID_LON_MAX - GRID_LON_MIN) / GRID_RESOLUTION) + 1,
)

DEFAULT_UNCERTAINTY = 0.1
"""
Fixed DOD uncertainty. Placeholder until O-B statistics are available; for scale, the two
B072 model variants differ by ~0.05 on average for the same scene.
"""

_MODEL_DIR_PATTERN = re.compile(r"_([0-9a-f]{64})$")


def model_id_from_path(path: Path) -> str | None:
    """Model directories end in a 64-char hash (`model_XGBoost_..._<sha256>`) telling the
    model variants apart. Returns its first 8 characters, or None if the file does not
    live in such a directory.
    """

    match = _MODEL_DIR_PATTERN.search(path.resolve().parent.name)
    return match.group(1)[:8] if match else None


def _nullable_int(value: float) -> int | None:
    """NaN isn't representable in JSON, so a missing flag becomes a null instead."""

    return None if np.isnan(value) else int(value)


def convert_mtg_react(
    parquet_path: Path,
    uncertainty: float = DEFAULT_UNCERTAINTY,
) -> pd.DataFrame | None:
    """Convert an MTG-REACT DOD prediction file to WRF-Ensembly Observation format.

    Args:
        parquet_path: Path to a `DOD_a355_prediction_*.parquet` file.
        uncertainty: Fixed DOD uncertainty assigned to every observation.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format, or None if no valid
        observations are found.
    """

    raw = pd.read_parquet(
        parquet_path,
        columns=["time", "latitude", "longitude", PREDICTION_COLUMN, *QC_COLUMNS],
    )

    dod = raw[PREDICTION_COLUMN].to_numpy(dtype=np.float64)
    raw = raw[~np.isnan(dod) & (dod >= 0)]
    if raw.empty:
        return None

    # float32 coordinates, so round to the nearest grid cell instead of truncating
    lat = raw["latitude"].to_numpy(dtype=np.float64)
    lon = raw["longitude"].to_numpy(dtype=np.float64)
    y_idx = np.rint((lat - GRID_LAT_MIN) / GRID_RESOLUTION).astype(int)
    x_idx = np.rint((lon - GRID_LON_MIN) / GRID_RESOLUTION).astype(int)
    outside = (y_idx < 0) | (y_idx >= GRID_SHAPE[0]) | (x_idx < 0) | (x_idx >= GRID_SHAPE[1])
    if outside.any():
        raise ValueError(
            f"{parquet_path.name} has {outside.sum()} pixels outside the expected grid "
            f"(lat {GRID_LAT_MIN}..{GRID_LAT_MAX}, lon {GRID_LON_MIN}..{GRID_LON_MAX}). "
            "Extend the GRID_* constants in mtg_react.py."
        )

    orig_coords = [
        {"indices": (int(y), int(x)), "shape": GRID_SHAPE, "names": ("y", "x")}
        for y, x in zip(y_idx, x_idx)
    ]

    model_id = model_id_from_path(parquet_path)
    qc_values = {col: raw[col].to_numpy(dtype=np.float64) for col in QC_COLUMNS}
    metadata = [
        {
            "model_id": model_id,
            **{col: _nullable_int(qc_values[col][i]) for col in QC_COLUMNS},
        }
        for i in range(len(raw))
    ]

    df = pd.DataFrame(
        {
            "instrument": "MTG_REACT",
            "quantity": "DOD_355nm",
            "time": pd.to_datetime(raw["time"].to_numpy()).tz_localize("UTC"),
            "latitude": lat,
            "longitude": lon,
            "z": 0.0,
            "z_type": "columnar",
            "value": raw[PREDICTION_COLUMN].to_numpy(dtype=np.float64),
            "value_uncertainty": uncertainty,
            "qc_flag": 0,
            "orig_filename": parquet_path.name,
            "metadata": metadata,
        }
    )
    df["orig_coords"] = orig_coords

    return df[obs_io.REQUIRED_COLUMNS]


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--uncertainty",
    default=DEFAULT_UNCERTAINTY,
    type=float,
    show_default=True,
    help="Fixed DOD uncertainty assigned to every observation.",
)
def mtg_react(input_path: Path, output_path: Path, uncertainty: float):
    """Convert an MTG-REACT DOD prediction file to WRF-Ensembly observation format.

    INPUT_PATH: Path to a DOD_a355_prediction_*.parquet file.
    OUTPUT_PATH: Path to save the converted parquet file.

    Only the dust optical depth prediction is converted, as DOD_355nm. The four
    MTCM_quality_* flags and the model variant (first 8 chars of the hash in the model
    directory name) are kept in the metadata. Night-time retrievals are included.
    """

    print(f"Converting MTG-REACT file: {input_path}")
    print(f"Model id: {model_id_from_path(input_path)}")
    print(f"Output path: {output_path}")

    converted_df = convert_mtg_react(input_path, uncertainty)

    if converted_df is None or converted_df.empty:
        print("No valid observations found in the input file, aborting")
        return

    obs_io.write_obs(converted_df, output_path)

    print(f"Successfully converted {len(converted_df)} observations")
    print(f"Saved to: {output_path}")
