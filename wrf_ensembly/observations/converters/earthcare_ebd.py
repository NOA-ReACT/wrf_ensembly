"""Converter for EarthCARE ATLID EBD files"""

from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_dilation

from wrf_ensembly.observations import io as obs_io

# QC parameters (calibrated on the 2024-08 golden-week dust_alpha free run,
# see certainty/analysis/dust_alpha_bias/ebd_alpha_obs_error.py):
# Physical ceiling for dust extinction at 355 nm. The previous blanket cap of
# 3e-4 1/m amputated real plume cores (~17% of aerosol-classified cells);
# values above ~2e-3 1/m (2/km) are cloud contamination, not dust.
DUST_EXT_CEILING = 2e-3
# Cloud-adjacency buffer: aerosol/clear cells within this many along-track
# profiles / height cells of any cloud/attenuated/invalid cell are rejected.
# We do not re-classify anything — we use the product's own classification and
# stay away from its edges. 98.5% of the physically implausible > 2e-3 cells
# sit inside this buffer.
CLOUD_BUFFER_PROFILES = 2
CLOUD_BUFFER_HEIGHT = 3
# Observation error sigma_o = INTERCEPT + SLOPE * extinction, N-weighted fit of
# binned sigma(O-B) in the screened SAL envelope at dust_alpha = 0.5. An upper
# bound (still contains model error); to be iterated with Desroziers once a DA
# run exists. Replaces the previous ad-hoc values (the 20% relative term was
# kept; the old 5e-6 floor was ~14x too small).
SIGMA_O_INTERCEPT = 6.8e-5  # 1/m
SIGMA_O_SLOPE = 0.20


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
    # 3) Aerosol extinction above the physical dust ceiling (cloud leakage)
    # 4) Cells adjacent to cloud/attenuated/invalid cells (cloud buffer)
    ext = ec["particle_extinction_coefficient_355nm_low_resolution"]
    cloudish = (
        ~((ec["simple_classification"] == 0) | (ec["simple_classification"] == 3))
        | ext.isnull()
    )
    # Dilate the cloud/invalid mask; data dims are (along_track, height).
    near_cloud = binary_dilation(
        cloudish.to_numpy(),
        structure=np.ones(
            (2 * CLOUD_BUFFER_PROFILES + 1, 2 * CLOUD_BUFFER_HEIGHT + 1), dtype=bool
        ),
    )
    qc_flag = cloudish.copy()
    qc_flag |= ext > DUST_EXT_CEILING
    qc_flag |= xr.DataArray(near_cloud, dims=cloudish.dims, coords=cloudish.coords)

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
    height = ec["height"].to_numpy().flatten()
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

    # Uncertainty: sigma_o = intercept + slope * extinction (1/m), fitted to the
    # binned O-B spread of the golden-week dust_alpha free run (see constants
    # at the top). Clear-air cells (value clamped to 0) get the intercept.
    value_uncertainty = SIGMA_O_INTERCEPT + SIGMA_O_SLOPE * np.nan_to_num(
        np.maximum(value, 0.0)
    )

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
