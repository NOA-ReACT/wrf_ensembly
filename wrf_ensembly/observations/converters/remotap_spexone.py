"""Converter for RemoTAP-spexone AOD observations"""

import re
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

    utcdate_str = utc_date.astype(int).astype(str)
    utcdate_str[utcdate_str == "29990101"] = np.nan
    dates = pd.to_datetime(utcdate_str.flatten(), format="%Y%m%d")
    dates += pd.to_timedelta(fraction_of_day.flatten(), unit="D")
    dates = dates.tz_localize("UTC")
    dates = dates.values.reshape(utc_date.shape)

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


def convert_remotap_spexone(
    path: Path,
    across_track_bin_size: int | None = None,
    along_track_bin_size: int | None = None,
) -> None | pd.DataFrame:
    """Convert a RemoTAP-spexone file to WRF-Ensembly Observation format.

    Args:
        path: The path to the RemoTAP-spexone netCDF file.
        across_track_bin_size: Number of across-track bins to aggregate into one using mean.
        along_track_bin_size: Number of along-track bins to aggregate into one using mean.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format (correct columns etc).
    """

    # geolocation arrays & date parsing
    ds_geolocation = xr.open_dataset(path, group="geolocation_data")
    ds_geolocation["timestamp"] = (
        ("bins_along_track", "bins_across_track"),
        parse_spex_date(
            ds_geolocation["utc_date"].to_numpy(), ds_geolocation["fracday"].to_numpy()
        ),
    )

    # Apply binning if requested
    bin_sizes = {}
    if across_track_bin_size is not None:
        bin_sizes["bins_across_track"] = across_track_bin_size
    if along_track_bin_size is not None:
        bin_sizes["bins_along_track"] = along_track_bin_size
    if bin_sizes:
        orig_size = (
            ds_geolocation.sizes["bins_across_track"],
            ds_geolocation.sizes["bins_along_track"],
        )
        ds_geolocation = ds_geolocation.coarsen(**bin_sizes, boundary="trim").mean()  # type: ignore
        new_size = (
            ds_geolocation.sizes["bins_across_track"],
            ds_geolocation.sizes["bins_along_track"],
        )

        print(
            f"Binned data from size {orig_size} to {new_size} using bin sizes {bin_sizes}"
        )

    latitude = ds_geolocation["latitude"].values.flatten()
    longitude = ds_geolocation["longitude"].values.flatten()
    timestamp = ds_geolocation["timestamp"].values.flatten()

    # AOD & uncertainty
    ds_geophysical = xr.open_dataset(path, group="geophysical_data")
    if bin_sizes:
        ds_geophysical = ds_geophysical.coarsen(**bin_sizes, boundary="trim").mean()  # type: ignore

    aod550 = ds_geophysical["aot550"].to_numpy().flatten()
    aod550_uncertainty = ds_geophysical["aot550_uncertainty"].to_numpy().flatten()

    # We need to preserve the original indices of the valid data points
    indices = get_index_tuples(ds_geophysical["aot550"])
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
            "metadata": pd.NA,
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
@click.option(
    "--imerg-path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to IMERG file to mark ocean/land points as invalid",
)
@click.option(
    "--across-track-bin-size",
    type=int,
    default=None,
    help="Number of across-track bins to aggregate into one using mean.",
)
@click.option(
    "--along-track-bin-size",
    type=int,
    default=None,
    help="Number of along-track bins to aggregate into one using mean.",
)
def remotap_spexone(
    input_path: Path,
    output_path: Path,
    imerg_path: Path | None = None,
    across_track_bin_size: int | None = None,
    along_track_bin_size: int | None = None,
):
    """Convert RemoTAP-spexone netCDF file to WRF-Ensembly observation format.

    Args:
        INPUT_PATH: Path to the input RemoTAP-spexone netCDF file.
        OUTPUT_PATH: Path to save the converted parquet file.

    If `--imerg-path` is given, the script will try to determine if this is a land or
    an ocean file based on the input filename. Then it will mark all mismatching points
    as invalid QC (qc_flag = 1). So if the input file is a land file, all ocean points
    will be marked as invalid, and vice versa. A threshold of >60% is used to determine
    which points are ocean based on the IMERG landseamask variable.

    You can use the `--across-track-bin-size` and `--along-track-bin-size` options to
    coarsen the data into larger bins using mean as the aggregation function. If the bin sizes
    don't evenly divide the data dimensions, the remaining data at the edges will be discarded.
    """

    print(f"Converting RemoTAP-SPEX file: {input_path}")
    print(f"Output path: {output_path}")

    # Sanity check product mane
    with xr.open_dataset(input_path) as root_ds:
        product_name = root_ds.attrs.get("product_name", "unknown")
    if not re.match(r"PACE_SPEXONE.*AER_(?:LAND|OCEAN)_REMOTAP\.nc", product_name):
        raise ValueError(
            f"File {input_path} is not a valid SPEX file. "
            r"Expected product name to match 'PACE_SPEXONE.*AER_(?:LAND|OCEAN)_REMOTAP\.nc', "
            f"but got '{product_name}'"
        )

    if imerg_path is not None:
        imerg = xr.open_dataset(imerg_path)
        imerg["is_ocean"] = imerg["landseamask"] > 60
        is_ocean_retrieval = "OCEAN" in product_name
        print(
            f"Using IMERG file {imerg_path} to mark {'ocean' if is_ocean_retrieval else 'land'} points as invalid"
        )

    # Convert the data
    converted_df = convert_remotap_spexone(
        input_path,
        across_track_bin_size=across_track_bin_size,
        along_track_bin_size=along_track_bin_size,
    )
    if converted_df is None or converted_df.empty:
        print("No observations found in the input file, aborting")
        return

    # If imerg_path is given, mark ocean/land points as invalid
    if imerg_path is not None:
        is_ocean = converted_df.apply(
            lambda row: imerg["is_ocean"]
            .sel(lat=row["latitude"], lon=row["longitude"], method="nearest")
            .item(),
            axis=1,
        )
        if is_ocean_retrieval:
            converted_df.loc[~is_ocean, "qc_flag"] = 1
            print(f"Marked {sum(~is_ocean)} land points as invalid")
        else:
            converted_df.loc[is_ocean, "qc_flag"] = 1
            print(f"Marked {sum(is_ocean)} ocean points as invalid")

    # Save to output path as parquet
    obs_io.write_obs(converted_df, output_path)

    print(f"Successfully converted {len(converted_df)} observations")
    print(f"Saved to: {output_path}")
