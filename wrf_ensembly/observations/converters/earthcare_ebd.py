"""Converter for EarthCARE ATLID EBD files"""

from pathlib import Path

import click
import pandas as pd
import xarray as xr
import numpy as np

from wrf_ensembly.observations import io as obs_io


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


def convert_earthcare_ebd(input_path: Path) -> pd.DataFrame:
    """Convert EarthCARE ATLID EBD file to WRF-Ensembly Observation format.

    ! TODO Written with the assumption we are using Thanasis' downsampling script
           which causes the bad bins to be NaN. We should add downsampling to this
           script I think.

    Args:
        input_path: Path to the EarthCARE ATLID EBD netCDF file.
    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format (correct columns etc).
    """

    ec = xr.open_dataset(input_path, group="ScienceData")
    ec = ec.set_coords(["time", "latitude", "longitude", "height"])

    # Flatten the data arrays
    particle_extinction = ec["particle_extinction_coefficient_355nm"].to_numpy()
    value = particle_extinction.flatten()
    value_uncertainty = (
        ec["particle_extinction_coefficient_355nm_error"].to_numpy().flatten()
    )
    height = ec["height"].to_numpy().flatten()

    # The coordinates are 1D arrays that need to be broadcast to the shape of the data
    # The data shape is (time, height), so we need to repeat the coordinates accordingly
    # time, latitude, longitude: (N,) -> (N, 1) -> (N, M)
    # where N is number of time steps and M is number of height bins
    time = (
        ec["time"]
        .to_numpy()[:, np.newaxis]
        .repeat(particle_extinction.shape[1], axis=1)
        .flatten()
    )
    latitude = (
        ec["latitude"]
        .to_numpy()[:, np.newaxis]
        .repeat(particle_extinction.shape[1], axis=1)
        .flatten()
    )
    longitude = (
        ec["longitude"]
        .to_numpy()[:, np.newaxis]
        .repeat(particle_extinction.shape[1], axis=1)
        .flatten()
    )

    # Any bin with NaN extinction value is invalid
    qc_flag = ~np.isnan(value)

    # Preserve the original indices
    indices = get_index_tuples(particle_extinction)
    coord_names = ec["particle_extinction_coefficient_355nm"].dims
    coord_shape = ec["particle_extinction_coefficient_355nm"].shape

    # Create dataframe in WRF-Ensembly Observation format
    df = pd.DataFrame(
        {
            "instrument": "EarthCARE_ATLID_EBD",
            "quantity": "EC_ATL_EXT355",
            "time": time,
            "latitude": latitude,
            "longitude": longitude,
            "z": height,
            "z_type": "height",
            "value": value,
            "value_uncertainty": value_uncertainty,
            "qc_flag": qc_flag.astype(int),
            # "orig_coords": ...,
            "orig_filename": input_path.name,
            "metadata": "",
        }
    )
    df["orig_coords"] = [
        {"indices": indices[i], "names": coord_names, "shape": coord_shape}
        for i in range(df.shape[0])
    ]
    df["time"] = df["time"].dt.tz_localize("UTC")
    df = df[obs_io.REQUIRED_COLUMNS]
    return df


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
def earthcare_atl_ebd(input_path: Path, output_path: Path):
    """Convert EarthCARE ATLID EBD file to WRF-Ensembly Observation format.

    Args:
        input_path: Path to the EarthCARE ATLID EBD netCDF file.
        output_path: Path to save the converted WRF-Ensembly Observation CSV file.
    """

    df = convert_earthcare_ebd(input_path)

    if output_path.is_dir():
        output_path = output_path / f"{input_path.stem}.parquet"
    obs_io.write_obs(df, output_path)

    print(f"Successfully wrote {len(df)} observations to {output_path}")
