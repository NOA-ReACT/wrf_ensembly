"""Converter for AEOLUS L2A files to WRF-Ensembly Observation format."""

from pathlib import Path

import click
import numpy as np
import pandas as pd

from wrf_ensembly.observations import io as obs_io

try:
    import coda

    HAS_CODA = True
except (ImportError, OSError):
    HAS_CODA = False
    coda = None


def convert_aeolus_l2a(path: Path) -> pd.DataFrame | None:
    """Convert an AEOLUS L2A file to WRF-Ensembly Observation format.

    Args:
        path: Path to the AEOLUS L2A file.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format, or None if no valid data.

    Raises:
        ImportError: If the coda library is not installed.
    """
    if not HAS_CODA:
        raise ImportError(
            "The 'coda' library is required for reading AEOLUS L2A files. "
            "Please install it using: pip install coda"
        )

    # Open the CODA file
    cf = coda.open(str(path))  # type: ignore[union-attr]
    if cf.product_type != "ALD_U_N_2A":
        cf.close()
        raise ValueError("Input file is not an AEOLUS DBL L2A file.")

    # How many MLE profiles are in the file?
    mle_count = int(coda.fetch(cf, "sph", "num_prof_mle"))  # type: ignore
    # How many measurements in each BRC?
    meas_in_brc_count = int(coda.fetch(cf, "sph", "num_meas_max_brc"))  # type: ignore
    # How many height bins in each measurement?
    height_bin_count = int(coda.fetch(cf, "sph", "num_bins_per_meas"))  # type: ignore

    # Grab geolocation
    latitude = np.zeros((mle_count, height_bin_count + 1))
    longitude = np.zeros((mle_count, height_bin_count + 1))
    altitude = np.zeros((mle_count, height_bin_count + 1))

    for i in range(mle_count):
        path_tuple = (
            "geolocation",
            i,
            "measurement_geolocation",
            15,  # 30 measurements in each BRC, pick the middle one
            "rayleigh_geolocation_height_bin",
            -1,
        )

        latitude[i, :] = coda.fetch(cf, *path_tuple, "latitude_of_height_bin")  # type: ignore
        longitude[i, :] = coda.fetch(cf, *path_tuple, "longitude_of_height_bin")  # type: ignore
        altitude[i, :] = coda.fetch(cf, *path_tuple, "altitude_of_height_bin")  # type: ignore

    # Destagger geolocation arrays (since they refer to the vertical edges of the bin)
    altitude_top = altitude[:, 0:-1]
    altitude_bottom = altitude[:, 1:]
    altitude = (altitude_top + altitude_bottom) / 2

    latitude = (latitude[:, 0:-1] + latitude[:, 1:]) / 2
    longitude = (longitude[:, 0:-1] + longitude[:, 1:]) / 2

    # Grab time, repeat for all height bins
    timestamp = coda.fetch(cf, "sca_mle_opt_properties", -1, "starttime")  # type: ignore
    timestamp = np.repeat(timestamp, height_bin_count).reshape(  # type: ignore
        (mle_count, height_bin_count)
    )  # Same for all height bins

    # Do the same for a profile ID
    profile_id = np.arange(mle_count)
    profile_id = np.repeat(profile_id, height_bin_count).reshape(
        (mle_count, height_bin_count)
    )

    # Grab featuremask
    feature_mask = np.zeros((mle_count, meas_in_brc_count, height_bin_count))
    for i, m, h in np.ndindex(feature_mask.shape):
        # For some reason the expanded argument list didn't work?
        feature_mask[i, m, h] = coda.fetch(  # type: ignore
            cf,
            f"feature_mask[{i}]/feature_mask_indices[{m * 24 + h}]/feature_mask_index",
        )

    # Feature mask values are:
    # - Negative values are surface, attenuated, ...
    # - 0 is clear sky
    # - 0-4 are mostly molecular
    # - 10 is certainy cloud
    # We reject anything outside the [0, 6]
    feature_mask = ~((feature_mask < 0) | (feature_mask >= 6))
    # Count of valid measurements in each BRC
    valid_meas_in_brc = np.sum(feature_mask, axis=1)

    # Read extinction & variance
    extinction = np.zeros((mle_count, height_bin_count))
    extinction_variance = np.zeros((mle_count, height_bin_count))
    for i in range(mle_count):
        extinction[i, :] = coda.fetch(  # type: ignore
            cf,
            "sca_mle_opt_properties",
            i,
            "sca_mle_optical_properties",
            -1,
            "extinction",
        )
        extinction_variance[i, :] = coda.fetch(  # type: ignore
            cf,
            "sca_mle_pcd",
            i,
            "profile_pcd_bins",
            -1,
            "extinction_variance",
        )

    cf.close()

    # For QC to pass we need a non-fill value extinction and at least 15 valid measurements
    # in the BRC
    qc_pass = (valid_meas_in_brc > 15) & (extinction != -1)

    # Flatten all arrays for the final dataframe
    profile_id_flat = profile_id.ravel(order="C")
    longitude_flat = longitude.ravel(order="C")
    latitude_flat = latitude.ravel(order="C")
    altitude_flat = altitude.ravel(order="C")
    timestamp_flat = timestamp.ravel(order="C")
    extinction_flat = extinction.ravel(order="C") * 1e-6  # Convert from 1/Mm to 1/m
    extinction_variance_flat = extinction_variance.ravel(order="C")
    qc_pass_flat = qc_pass.ravel(order="C")
    altitude_top_flat = altitude_top.ravel(order="C")
    altitude_bottom_flat = altitude_bottom.ravel(order="C")
    valid_meas_in_brc_flat = valid_meas_in_brc.ravel(order="C")

    # Check if we have any data
    if len(extinction_flat) == 0:
        return None

    # Create QC flags based on data quality:
    # QC = 0: Good quality (pass QC checks)
    # QC = 1: Failed QC checks (insufficient valid measurements or fill value)
    qc_flags = np.where(qc_pass_flat, 0, 1)

    # Convert date from 'seconds since 2000' to pd.Timestamp
    time_utc = pd.to_datetime("2000-01-01") + pd.to_timedelta(
        timestamp_flat, unit="seconds"
    )
    # Ensure UTC timezone
    time_utc = time_utc.tz_localize("UTC")

    # Convert longitude from [0, 360] to [-180, 180]
    longitude_180 = ((longitude_flat + 180) % 360) - 180

    # Create the WRF-Ensembly observation format DataFrame directly
    obs_df = pd.DataFrame(
        {
            "time": time_utc,
            "longitude": longitude_180,
            "latitude": latitude_flat,
            "z": altitude_flat,
            "z_type": "height",  # Altitude is in meters above geoid
            "value": extinction_flat,
            # "value_uncertainty": extinction_variance_flat,
            "value_uncertainty": extinction_flat * 0.2,  # Assume 20% uncertainty
            "qc_flag": qc_flags,
            "instrument": "AEOLUS_L2A",
            "quantity": "LIDAR_EXTINCTION_355nm",  # LIDAR extinction at 355nm
        }
    )

    # Create orig_coords for traceability
    # Convert flattened index to 2D indices (mle_count, height_bin_count)
    obs_df["orig_coords"] = obs_df.apply(
        lambda row: {
            "indices": np.array(
                (row.name // height_bin_count, row.name % height_bin_count), dtype=int
            ),
            "shape": np.array((mle_count, height_bin_count), dtype=int),
            "names": np.array(("profile", "height_bin"), dtype=object),
        },
        axis=1,
    )

    obs_df["orig_filename"] = path.name

    # Add metadata with profile ID and bin information
    obs_df["metadata"] = [
        {
            "profile_id": int(profile_id_flat[i]),
            "altitude_top": float(altitude_top_flat[i]),
            "altitude_bottom": float(altitude_bottom_flat[i]),
            "valid_measurements_in_brc": int(valid_meas_in_brc_flat[i]),
        }
        for i in range(len(obs_df))
    ]

    # Sort columns as defined in the schema and validate
    obs_df = obs_df[obs_io.REQUIRED_COLUMNS]
    obs_io.validate_schema(obs_df)

    return obs_df


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
def aeolus_l2a(input_path: Path, output_path: Path):
    """Convert AEOLUS L2A file to WRF-Ensembly observation format.

    INPUT_PATH: Path to the AEOLUS L2A file
    OUTPUT_PATH: Path where to save the converted observations (will be saved as parquet)
    """
    if not HAS_CODA:
        raise click.ClickException(
            "The 'coda' library is required for reading AEOLUS L2A files. "
            "Please install it using: pip install coda"
        )

    print(f"Converting AEOLUS L2A file: {input_path}")
    print(f"Output path: {output_path}")

    # Convert the data
    converted_df = convert_aeolus_l2a(input_path)
    if converted_df is None or converted_df.empty:
        print("No observations found in the input file, aborting")
        return

    # Save to output path as parquet
    obs_io.write_obs(converted_df, output_path)

    # Report statistics
    total_obs = len(converted_df)
    good_obs = len(converted_df[converted_df["qc_flag"] == 0])
    bad_obs = total_obs - good_obs

    print(f"Successfully converted {total_obs} observations:")
    print(f"  - {good_obs} good quality observations (QC=0)")
    print(f"  - {bad_obs} flagged observations (QC>0)")
    print(f"Saved to: {output_path}")
