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
    vertical_bins: list[int] | None = None,
    horizontal_bin_size: int | None = None,
    filter_before_binning: bool = False,
) -> pd.DataFrame:
    """Convert EarthCARE ATLID EBD file to WRF-Ensembly Observation format.

    Args:
        input_path: Path to the EarthCARE ATLID EBD netCDF file.
        vertical_bin_size: Size of vertical bins for aggregation, group by this many height bins and use mean.
        horizontal_bin_size: Size of horizontal bins for aggregation, group by this many profiles and use mean.
        filter_before_binning: If True, apply QC filtering before binning.
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
    # 3) Extinction in [0, 0.0003]
    qc_flag = ec["particle_extinction_coefficient_355nm_low_resolution"].isnull()
    qc_flag |= ~(
        (ec["simple_classification"] == 0) | (ec["simple_classification"] == 3)
    )
    qc_flag |= ec["particle_extinction_coefficient_355nm_low_resolution"] <= 0
    qc_flag |= ec["particle_extinction_coefficient_355nm_low_resolution"] > 0.0003

    # Filter data before binning if requested
    if filter_before_binning:
        print("Applying QC filtering before binning.")
        ec = ec.where(qc_flag, drop=False)

    # Assign qc_flag back to dataset so it takes up the correct shape after possible binning
    ec["qc_flag"] = qc_flag

    # Apply binning if requested
    if vertical_bins is not None:
        # We can't use groupby_bins because height is 2D in the file. So we will do it manually for each vertical profile.
        ec_downsampled = []
        for i in range(ec.sizes["along_track"]):
            ec_profile = ec.isel(along_track=i)
            ec_binned = (
                ec_profile.groupby_bins("height", vertical_bins)
                .mean(skipna=True)
                .rename_dims({"height_bins": "height"})
            )
            ec_downsampled.append(ec_binned)
        ec = xr.concat(ec_downsampled, dim="along_track")

        # For each height bin, compute the mid point, and assign as the new height coordinate
        # Remember that this is an array of pd.Interval objects
        bins = ec["height_bins"].values
        bin_mids = np.array([interval.mid for interval in bins])

        # The original height coordinate is 2D (along_track, height), so we need to broadcast the bin mids accordingly
        bin_mids = np.tile(bin_mids, (ec.sizes["along_track"], 1))

        # Assign the new height coordinate
        ec = ec.drop_vars("height_bins")
        ec["height"] = (("along_track", "height"), bin_mids)

    if horizontal_bin_size is not None:
        print(f"Applying horizontal binning with size {horizontal_bin_size}.")
        ec = ec.coarsen(along_track=horizontal_bin_size, boundary="trim").mean()

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

    # Uncertainty: 20% when there are aerosols, 5% when there are not
    # This is a bit arbitrary, but we need something reasonable
    # TODO Better uncertainty estimation
    value_uncertainty = np.where(simple_classification == 3, 0.2 * value, 0.05 * value)

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
@click.option(
    "--vertical-bins",
    type=str,
    help="Which vertical bins to use for aggregation, in meters, comma-separated.",
)
@click.option(
    "--horizontal-bin-size",
    type=int,
    help="How many horizontal bins/profiles to aggregate.",
)
@click.option("--filter-before-binning", is_flag=True, help="Filter before binning.")
def earthcare_atl_ebd(
    input_path: Path,
    output_path: Path,
    vertical_bins: str | None = None,
    horizontal_bin_size: int | None = None,
    filter_before_binning: bool = False,
):
    """
    Convert EarthCARE ATLID EBD file to WRF-Ensembly Observation format.
    If you specify either (or both) of the bin sizes, the data will be aggregated accordingly, using `mean` as the aggregation function, ignoring NaNs.
    If you do decide to downsample, it is recommended to also use the `--filter-before-binning` flag to apply QC filtering before the binning operation.

    Args:
        input_path: Path to the EarthCARE ATLID EBD netCDF file.
        output_path: Path to save the converted WRF-Ensembly Observation CSV file.
    """

    vertical_bins_parsed: list[int] | None = None
    if vertical_bins is not None:
        vertical_bins_parsed = [int(v) for v in vertical_bins.split(",")]
        print(f"Using vertical bins: {vertical_bins_parsed}")

    df = convert_earthcare_ebd(
        input_path, vertical_bins_parsed, horizontal_bin_size, filter_before_binning
    )

    if output_path.is_dir():
        output_path = output_path / f"{input_path.stem}.parquet"
    obs_io.write_obs(df, output_path)

    print(f"Successfully wrote {len(df)} observations to {output_path}")
