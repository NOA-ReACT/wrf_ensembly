"""Converter for MODIS AOD observations"""

from pathlib import Path

import click
import numpy as np
import pandas as pd

from wrf_ensembly.observations import io as obs_io

try:
    from pyhdf.SD import SD, SDC

    HAS_PYHDF = True
except ImportError:
    HAS_PYHDF = False


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


def convert_modis(
    hdf_path: Path,
) -> None | pd.DataFrame:
    """Convert a MODIS AOD HDF4 file to WRF-Ensembly Observation format.

    Args:
        hdf_path: Path to the MODIS HDF4 file.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format (correct columns etc).
    """

    if not HAS_PYHDF:
        raise ImportError(
            "pyhdf is required to read MODIS HDF4 files. "
            "Install it with: pip install pyhdf"
        )

    # Open the HDF4 file
    hdf = SD(str(hdf_path), SDC.READ)

    # Read coordinate data
    lat_ds = hdf.select("Latitude")
    lon_ds = hdf.select("Longitude")
    time_ds = hdf.select("Scan_Start_Time")

    latitude = lat_ds[:].flatten()
    longitude = lon_ds[:].flatten()
    scan_time = time_ds[:].flatten()

    lat_ds.endaccess()
    lon_ds.endaccess()
    time_ds.endaccess()

    # Read AOD data
    aod_ds = hdf.select("AOD_550_Dark_Target_Deep_Blue_Combined")
    aod_shape = aod_ds.info()[2]
    aod_raw = aod_ds[:].flatten()
    aod_attrs = aod_ds.attributes()
    aod_scale = aod_attrs["scale_factor"]
    aod_fill = aod_attrs["_FillValue"]
    coord_names = tuple(aod_ds.dimensions().keys())
    aod_ds.endaccess()

    # Read QA flag
    qa_ds = hdf.select("AOD_550_Dark_Target_Deep_Blue_Combined_QA_Flag")
    qa_flag = qa_ds[:].flatten()
    qa_fill = qa_ds.attributes()["_FillValue"]
    qa_ds.endaccess()

    # Close the file
    hdf.end()

    # Apply scale factor to AOD
    aod550 = aod_raw * aod_scale

    # Convert time from TAI93 to UTC
    timestamp = tai93_to_datetime(scan_time)

    # Create valid mask: filter out fill values and require good QA (2 or 3)
    # QA values: 0 = No Confidence, 1 = Marginal, 2 = Good, 3 = Very Good
    valid_mask = (
        (aod_raw != aod_fill)
        & ~np.isnan(aod550)
        & (qa_flag != qa_fill)
        & (qa_flag >= 2)  # Only accept Good or Very Good quality
        & (latitude != -999.0)
        & (longitude != -999.0)
        & (scan_time != -999.0)
    )

    if not np.any(valid_mask):
        return None

    # Get indices for each flattened element
    indices = get_index_tuples(aod_raw.reshape(aod_shape))

    # Create dataframe
    df = pd.DataFrame(
        {
            "instrument": "MODIS",
            "quantity": "AOD_550",
            "time": timestamp,
            "latitude": latitude,
            "longitude": longitude,
            "z": 0.0,
            "z_type": "columnar",
            "value": aod550,
            "value_uncertainty": pd.NA,  # MODIS doesn't provide uncertainty in this product
            "qc_flag": valid_mask.astype(int),
            "orig_filename": hdf_path.name,
            "metadata": pd.NA,
        }
    )

    df["orig_coords"] = [
        {"indices": indices[i], "names": coord_names, "shape": aod_shape}
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
def modis(input_path: Path, output_path: Path):
    """Convert MODIS AOD HDF4 file to WRF-Ensembly observation format.

    Args:
        INPUT_PATH: Path to the input MODIS HDF4 file.
        OUTPUT_PATH: Path to save the converted parquet file.

    This converter extracts the AOD_550_Dark_Target_Deep_Blue_Combined variable
    which contains the combined Dark Target and Deep Blue AOT at 0.55 micron for
    land and ocean. Only observations with QA flag >= 2 (Good or Very Good) are kept.
    """

    print(f"Converting MODIS AOD file: {input_path}")
    print(f"Output path: {output_path}")

    # Convert the data
    converted_df = convert_modis(input_path)

    if converted_df is None or converted_df.empty:
        print("No valid observations found in the input file, aborting")
        return

    # Save to output path as parquet
    obs_io.write_obs(converted_df, output_path)

    print(f"Successfully converted {len(converted_df)} observations")
    print(f"Saved to: {output_path}")
