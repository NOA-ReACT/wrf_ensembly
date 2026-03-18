"""Converter for EarthCARE ATLID EBD files"""

from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr

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


def convert_earthcare_ebd(
    input_path: Path,
) -> pd.DataFrame:
    """Convert EarthCARE ATLID EBD file to WRF-Ensembly Observation format.

    Args:
        input_path: Path to the EarthCARE ATLID EBD netCDF file.
    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format (correct columns etc).
    """

    ec = xr.open_dataset(input_path, group="ScienceData")
    ec = ec.set_coords(["time", "latitude", "longitude", "height"])

    ec = ec[
        [
            "particle_extinction_coefficient_355nm_low_resolution",
            "particle_extinction_coefficient_355nm_low_resolution_error",
            "simple_classification",
        ]
    ]
    # Prepare qc flag, 0 = valid, 1 = invalid:
    # 1) No NaN extinction
    # 2) Only aerosol (3) or clear air (0)
    # 3) Extinction in larger than 0.0003
    qc_flag = ec["particle_extinction_coefficient_355nm_low_resolution"].isnull()
    qc_flag |= ~(
        (ec["simple_classification"] == 0) | (ec["simple_classification"] == 3)
    )
    qc_flag |= ec["particle_extinction_coefficient_355nm_low_resolution"] > 0.0003

    # Clamp extinction to zero
    ec["particle_extinction_coefficient_355nm_low_resolution"] = ec[
        "particle_extinction_coefficient_355nm_low_resolution"
    ].where(ec["particle_extinction_coefficient_355nm_low_resolution"] >= 0, 0.0)

    # Assign qc_flag back to dataset
    ec["qc_flag"] = qc_flag

    # Flatten the data arrays, extract from `ec` again after possible binning and filtering
    particle_extinction = ec[
        "particle_extinction_coefficient_355nm_low_resolution"
    ].to_numpy()
    value = particle_extinction.flatten()
    value_uncertainty = (
        ec["particle_extinction_coefficient_355nm_low_resolution_error"]
        .to_numpy()
        .flatten()
    )
    uncertainty_is_zero = value_uncertainty == 0
    height = ec["height"].to_numpy().flatten()
    simple_classification = ec["simple_classification"].to_numpy().flatten()
    qc_flag = ec["qc_flag"].to_numpy().flatten()

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

    # Preserve the original indices
    indices = get_index_tuples(particle_extinction)
    coord_names = ec["particle_extinction_coefficient_355nm_low_resolution"].dims
    coord_shape = ec["particle_extinction_coefficient_355nm_low_resolution"].shape

    # Uncertainty:
    # - 0.0025 1/km for clear air fixed
    # - 20% of extinction value for aerosol, with a minimum value of 0.0025 1/km
    # Remember that everything needs to be in 1/m
    value_uncertainty = np.full_like(value_uncertainty, 0.0025)
    clear_mask = simple_classification != 3
    value_uncertainty[clear_mask] = 0.0015
    aerosol_mask = simple_classification == 3
    value_uncertainty[aerosol_mask] = np.maximum(
        0.20 * value[aerosol_mask],  # 20% relative error
        0.005,  # Higher floor for aerosol regions
    )
    value_uncertainty = value_uncertainty / 1000.0  # Convert to 1/m

    # Shapes of everything
    shapes = {
        "time": time.shape,
        "latitude": latitude.shape,
        "longitude": longitude.shape,
        "height": height.shape,
        "value": value.shape,
        "value_uncertainty": value_uncertainty.shape,
        "qc_flag": qc_flag.shape,
    }

    # Create dataframe in WRF-Ensembly Observation format
    df = pd.DataFrame(
        {
            "instrument": "EarthCARE_ATLID_EBD",
            "quantity": "LIDAR_EXTINCTION_355nm",
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
            "metadata": pd.NA,
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
def earthcare_atl_ebd(
    input_path: Path,
    output_path: Path,
):
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
