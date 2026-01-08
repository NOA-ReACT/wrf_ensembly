"""Converter for VIIRS AOD observations"""

from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr

from wrf_ensembly.observations import io as obs_io


def tai93_to_datetime(tai93_seconds: np.ndarray) -> np.ndarray:
    """
    Convert TAI93 time (seconds since 1993-01-01 00:00:00 TAI) to pandas timestamps.

    TAI (International Atomic Time) is ahead of UTC by a certain number of leap seconds.
    As of 2024, TAI is 37 seconds ahead of UTC.

    Args:
        tai93_seconds: Array of seconds since 1993-01-01 00:00:00 TAI

    Returns:
        Array of pandas timestamps in UTC
    """
    # TAI epoch is 1993-01-01 00:00:00
    # As of 2024, TAI is 37 seconds ahead of UTC
    # So TAI93 = UTC93 + 37 seconds
    tai_epoch = pd.Timestamp("1993-01-01 00:00:00", tz="UTC")
    leap_seconds = 37  # Current leap second offset as of 2024

    # Convert TAI93 to UTC by subtracting leap seconds
    utc_timestamps = tai_epoch + pd.to_timedelta(tai93_seconds - leap_seconds, unit="s")

    return utc_timestamps


def get_index_tuples(arr):
    """
    Get array of index tuples for flattened N-dimensional array.

    Parameters:
        arr: numpy array of any dimension

    Returns:
        list of tuples, where each tuple contains the original N-D indices
    """
    flat_indices = np.arange(arr.size)
    return [np.unravel_index(i, arr.shape) for i in flat_indices]


def convert_viirs(
    ds: xr.Dataset,
    original_filename: str,
) -> None | pd.DataFrame:
    """Convert a VIIRS AOD file to WRF-Ensembly Observation format.

    Args:
        ds: xarray Dataset containing VIIRS data.
        original_filename: Name of the original file.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format (correct columns etc).
    """

    # Extract coordinates
    latitude = ds["Latitude"].values.flatten()
    longitude = ds["Longitude"].values.flatten()
    scan_time = ds["Scan_Start_Time"].values.flatten()

    # Convert time from TAI93 to UTC
    timestamp = tai93_to_datetime(scan_time)

    # Extract AOD data - use the best estimate (QA-filtered)
    aod550 = ds["Aerosol_Optical_Thickness_550_Land_Ocean_Best_Estimate"].values.flatten()

    # Get uncertainties - combine land and ocean uncertainties
    # Use the appropriate uncertainty based on where data exists
    uncertainty_land = ds["Aerosol_Optical_Thickness_550_Expected_Uncertainty_Land"].values.flatten()
    uncertainty_ocean = ds["Aerosol_Optical_Thickness_550_Expected_Uncertainty_Ocean"].values.flatten()

    # Use land uncertainty where available, otherwise ocean uncertainty
    aod550_uncertainty = np.where(
        ~np.isnan(uncertainty_land) & (uncertainty_land != -999.0),
        uncertainty_land,
        uncertainty_ocean
    )

    # Create valid mask: filter out fill values and NaNs
    # The fill value is -999.0 for AOD
    valid_mask = (
        (aod550 != -999.0) &
        ~np.isnan(aod550) &
        (aod550_uncertainty != -999.0) &
        ~np.isnan(aod550_uncertainty)
    )

    if not np.any(valid_mask):
        return None

    # Get original coordinate information
    indices = get_index_tuples(ds["Aerosol_Optical_Thickness_550_Land_Ocean_Best_Estimate"])
    coord_names = ds["Aerosol_Optical_Thickness_550_Land_Ocean_Best_Estimate"].dims
    coord_shape = ds["Aerosol_Optical_Thickness_550_Land_Ocean_Best_Estimate"].shape

    # Create dataframe
    df = pd.DataFrame(
        {
            "instrument": "VIIRS",
            "quantity": "AOD_550",
            "time": timestamp,
            "latitude": latitude,
            "longitude": longitude,
            "z": 0.0,
            "z_type": "columnar",
            "value": aod550,
            "value_uncertainty": aod550_uncertainty,
            "qc_flag": valid_mask.astype(int),
            "orig_filename": original_filename,
            "metadata": pd.NA,
        }
    )

    df["orig_coords"] = [
        {"indices": indices[i], "names": coord_names, "shape": coord_shape}
        for i in range(df.shape[0])
    ]

    # Filter to only valid observations
    df = df[df["qc_flag"] == 1]

    # Set qc_flag to 0 for valid data (0 = good in our system)
    df["qc_flag"] = 0

    df = df[obs_io.REQUIRED_COLUMNS]
    return df


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
def viirs(input_path: Path, output_path: Path):
    """Convert VIIRS AOD netCDF file to WRF-Ensembly observation format.

    Args:
        INPUT_PATH: Path to the input VIIRS netCDF file.
        OUTPUT_PATH: Path to save the converted parquet file.

    This converter extracts the Aerosol_Optical_Thickness_550_Land_Ocean_Best_Estimate
    variable which contains QA-filtered AOD retrievals at 550 nm over both land and ocean.
    """

    print(f"Converting VIIRS AOD file: {input_path}")
    print(f"Output path: {output_path}")

    # Open the dataset
    # Note: decode_timedelta=False to avoid FutureWarning about TAI93 time format
    ds = xr.open_dataset(input_path, decode_timedelta=False)

    # Convert the data
    converted_df = convert_viirs(ds, original_filename=input_path.name)

    # Close the dataset
    ds.close()

    if converted_df is None or converted_df.empty:
        print("No valid observations found in the input file, aborting")
        return

    # Save to output path as parquet
    obs_io.write_obs(converted_df, output_path)

    print(f"Successfully converted {len(converted_df)} observations")
    print(f"Saved to: {output_path}")
