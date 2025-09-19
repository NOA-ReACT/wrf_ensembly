"""Converter for RemoTAP-spexone AOD observations"""

from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr

from wrf_ensembly.observations import io as obs_io


def parse_spex_date(
    utc_date: np.ndarray, fraction_of_day: np.ndarray
) -> pd.DatetimeIndex:
    """
    SPEX dates are provided in a UTC date as an integer and a float representing the fraction of day.
    This function converts them to a pandas timestamp.
    """

    utcdate_str = np.array([str(int(date)) for date in utc_date])
    utcdate_str[utcdate_str == "29990101"] = np.nan
    dates = pd.to_datetime(utcdate_str, format="%Y%m%d")
    dates += pd.to_timedelta(fraction_of_day, unit="D")

    return dates


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


def convert_remotap_spexone(path: Path) -> None | pd.DataFrame:
    """Convert a RemoTAP-spexone file to WRF-Ensembly Observation format.

    Args:
        path: The path to the RemoTAP-spexone netCDF file.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format (correct columns etc).
    """

    # geolocation arrays & date parsing
    ds_geolocation = xr.open_dataset(path, group="geolocation_data")

    latitude = ds_geolocation["latitude"].values.flatten()
    longitude = ds_geolocation["longitude"].values.flatten()
    timestamp = parse_spex_date(
        ds_geolocation["utc_date"].values.flatten(),
        ds_geolocation["fracday"].values.flatten(),
    )

    # AOD & uncertainty
    ds_geophysical = xr.open_dataset(path, group="geophysical_data")
    aod550 = ds_geophysical["aot550"].values.flatten()
    aod550_uncertainty = ds_geophysical["aot550_uncertainty"].values.flatten()

    # We need to preserve the original indices of the valid data points
    indices = get_index_tuples(ds_geophysical["aot550"].values)
    coord_names = ds_geophysical["aot550"].dims
    coord_shape = ds_geophysical["aot550"].shape

    # Create dataframe
    df = pd.DataFrame(
        {
            "instrument": "RemoTAP-spexone",
            "quantity": "AOD_550nm",
            "time": timestamp,
            "latitude": latitude,
            "longitude": longitude,
            "z": 0.0,
            "z_type": "columnar",
            "value": aod550,
            "value_uncertainty": aod550_uncertainty,
            "qc_flag": 0,
            # "orig_coords": ...,
            "orig_filename": path.name,
            "metadata": "",
        }
    )
    df["orig_coords"] = [
        {"indices": indices[i], "names": coord_names, "shape": coord_shape}
        for i in range(df.shape[0])
    ]

    df = df[obs_io.REQUIRED_COLUMNS]
    return df


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
def remotap_spexone(input_path: Path, output_path: Path):
    """Convert RemoTAP-spexone netCDF file to WRF-Ensembly observation format.

    Args:
        INPUT_PATH: Path to the input RemoTAP-spexone netCDF file.
        OUTPUT_PATH: Path to save the converted parquet file.
    """

    print(f"Converting RemoTAP-SPEX file: {input_path}")
    print(f"Output path: {output_path}")

    # Convert the data
    converted_df = convert_remotap_spexone(input_path)
    if converted_df is None or converted_df.empty:
        print("No observations found in the input file, aborting")
        return

    # Save to output path as parquet
    obs_io.write_obs(converted_df, output_path)

    print(f"Successfully converted {len(converted_df)} observations")
    print(f"Saved to: {output_path}")
