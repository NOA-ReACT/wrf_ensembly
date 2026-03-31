"""Converter for AEOLUS L2A files to WRF-Ensembly Observation format."""

from pathlib import Path

import click
import coda
import numpy as np
import pandas as pd

from wrf_ensembly.observations import io as obs_io


def _read_brc_geolocation(cf, brc_idx: int, meas_idx: int, height_bin_count: int):
    """Read and destagger rayleigh geolocation for one BRC/measurement.

    The geolocation fields contain height_bin_count+1 values at vertical edges.
    Returns five arrays of shape (height_bin_count,): lat, lon, alt (midpoints),
    alt_top, alt_bottom.
    """
    path_tuple = (
        "geolocation",
        brc_idx,
        "measurement_geolocation",
        meas_idx,
        "rayleigh_geolocation_height_bin",
        -1,
    )
    lat = coda.fetch(cf, *path_tuple, "latitude_of_height_bin")
    lon = coda.fetch(cf, *path_tuple, "longitude_of_height_bin")
    alt = coda.fetch(cf, *path_tuple, "altitude_of_height_bin")

    alt_top = alt[:-1]
    alt_bottom = alt[1:]
    return (
        (lat[:-1] + lat[1:]) / 2,
        (lon[:-1] + lon[1:]) / 2,
        (alt_top + alt_bottom) / 2,
        alt_top,
        alt_bottom,
    )


def _read_feature_mask(
    cf, brc_count: int, meas_in_brc_count: int, height_bin_count: int
) -> np.ndarray:
    """Read and threshold the feature mask.

    Feature mask values:
      <0 : surface / attenuated / invalid
      0-4: mostly molecular (clear sky range)
      >=6: cloud / dense aerosol
    We accept values in [0, 6).

    Returns a boolean array of shape (brc_count, meas_in_brc_count, height_bin_count)
    where True means the measurement/bin passed the cloud-free quality filter.
    Callers that need a BRC-level count should sum over axis=1.
    """
    feature_mask = np.zeros((brc_count, meas_in_brc_count, height_bin_count))
    for i, m, h in np.ndindex(feature_mask.shape):
        feature_mask[i, m, h] = coda.fetch(
            cf,
            f"feature_mask[{i}]/feature_mask_indices[{m * height_bin_count + h}]/feature_mask_index",
        )
    return ~(
        (feature_mask < 0) | (feature_mask >= 6)
    )  # (brc_count, meas_count, height_bin_count)


def _make_obs_df(
    timestamp_flat,
    lon_flat,
    lat_flat,
    alt_flat,
    extinction_flat,
    qc_flags,
    instrument: str,
    orig_coords: list,
    orig_filename: str,
    metadata: list,
) -> pd.DataFrame:
    """Assemble a WRF-Ensembly observation DataFrame from pre-flattened arrays."""
    time_utc = pd.to_datetime("2000-01-01") + pd.to_timedelta(
        timestamp_flat, unit="seconds"
    )
    time_utc = time_utc.tz_localize("UTC")
    lon_180 = ((lon_flat + 180) % 360) - 180

    obs_df = pd.DataFrame(
        {
            "time": time_utc,
            "longitude": lon_180,
            "latitude": lat_flat,
            "z": alt_flat,
            "z_type": "height",
            "value": extinction_flat,
            "value_uncertainty": extinction_flat * 0.2,  # Assume 20% uncertainty
            "qc_flag": qc_flags,
            "instrument": instrument,
            "quantity": "LIDAR_EXTINCTION_355nm",
        }
    )
    obs_df["orig_coords"] = orig_coords
    obs_df["orig_filename"] = orig_filename
    obs_df["metadata"] = metadata

    obs_df = obs_df[obs_io.REQUIRED_COLUMNS]
    obs_io.validate_schema(obs_df)
    return obs_df


def _convert_mle(
    cf,
    brc_count: int,
    height_bin_count: int,
    feature_mask: np.ndarray,
    path: Path,
) -> pd.DataFrame | None:
    """Convert the SCA-MLE product."""
    lat = np.zeros((brc_count, height_bin_count))
    lon = np.zeros((brc_count, height_bin_count))
    alt = np.zeros((brc_count, height_bin_count))
    alt_top = np.zeros((brc_count, height_bin_count))
    alt_bottom = np.zeros((brc_count, height_bin_count))
    for i in range(brc_count):
        lat[i], lon[i], alt[i], alt_top[i], alt_bottom[i] = _read_brc_geolocation(
            cf, i, 15, height_bin_count
        )

    timestamp = coda.fetch(cf, "sca_mle_opt_properties", -1, "starttime")
    timestamp = np.repeat(timestamp, height_bin_count).reshape(
        (brc_count, height_bin_count)
    )

    # Count cloud-free measurements per BRC per bin
    valid_meas = np.sum(feature_mask, axis=1)  # (brc_count, height_bin_count)

    extinction = np.zeros((brc_count, height_bin_count))
    extinction_variance = np.zeros((brc_count, height_bin_count))
    for i in range(brc_count):
        extinction[i] = coda.fetch(
            cf,
            "sca_mle_opt_properties",
            i,
            "sca_mle_optical_properties",
            -1,
            "extinction",
        )
        extinction_variance[i] = coda.fetch(
            cf, "sca_mle_pcd", i, "profile_pcd_bins", -1, "extinction_variance"
        )

    # Replace -1 sentinels with NaN
    extinction = np.where(extinction == -1, np.nan, extinction)

    qc_pass = (valid_meas > 15) & ~np.isnan(extinction)

    profile_idx = np.repeat(np.arange(brc_count), height_bin_count)
    bin_idx = np.tile(np.arange(height_bin_count), brc_count)

    n = brc_count * height_bin_count
    orig_coords = [
        {
            "indices": np.array((profile_idx[i], bin_idx[i]), dtype=int),
            "shape": np.array((brc_count, height_bin_count), dtype=int),
            "names": np.array(("profile", "height_bin"), dtype=object),
        }
        for i in range(n)
    ]

    alt_top_flat = alt_top.ravel()
    alt_bottom_flat = alt_bottom.ravel()
    valid_meas_flat = valid_meas.ravel()
    metadata = [
        {
            "profile_id": int(profile_idx[i]),
            "altitude_top": float(alt_top_flat[i])
            if not np.isnan(alt_top_flat[i])
            else None,
            "altitude_bottom": float(alt_bottom_flat[i])
            if not np.isnan(alt_bottom_flat[i])
            else None,
            "valid_measurements_in_brc": float(valid_meas_flat[i])
            if not np.isnan(valid_meas_flat[i])
            else None,
        }
        for i in range(n)
    ]

    extinction_flat = extinction.ravel() * 1e-6  # 1/Mm → 1/m
    if len(extinction_flat) == 0:
        return None

    return _make_obs_df(
        timestamp_flat=timestamp.ravel(),
        lon_flat=lon.ravel(),
        lat_flat=lat.ravel(),
        alt_flat=alt.ravel(),
        extinction_flat=extinction_flat,
        qc_flags=np.where(qc_pass.ravel(), 0, 1),
        instrument="AEOLUS_L2A_MLE",
        orig_coords=orig_coords,
        orig_filename=path.name,
        metadata=metadata,
    )


def _convert_sca(
    cf,
    sca_count: int,
    height_bin_count: int,
    feature_mask: np.ndarray,
    path: Path,
) -> pd.DataFrame | None:
    """Convert the SCA product.

    SCA optical properties include their own per-bin geolocation (already at midpoints).
    """
    lat = np.zeros((sca_count, height_bin_count))
    lon = np.zeros((sca_count, height_bin_count))
    alt = np.zeros((sca_count, height_bin_count))
    extinction = np.zeros((sca_count, height_bin_count))
    extinction_variance = np.zeros((sca_count, height_bin_count))
    processing_qc = np.zeros((sca_count, height_bin_count))

    for i in range(sca_count):
        lat[i] = coda.fetch(
            cf, "sca_optical_properties", i, "geolocation_middle_bins", -1, "latitude"
        )
        lon[i] = coda.fetch(
            cf, "sca_optical_properties", i, "geolocation_middle_bins", -1, "longitude"
        )
        alt[i] = coda.fetch(
            cf, "sca_optical_properties", i, "geolocation_middle_bins", -1, "altitude"
        )
        extinction[i] = coda.fetch(
            cf, "sca_optical_properties", i, "sca_optical_properties", -1, "extinction"
        )
        extinction_variance[i] = coda.fetch(
            cf, "sca_pcd", i, "profile_pcd_bins", -1, "extinction_variance"
        )
        processing_qc[i] = coda.fetch(
            cf, "sca_pcd", i, "profile_pcd_bins", -1, "processing_qc_flag"
        )

    timestamp = coda.fetch(cf, "sca_optical_properties", -1, "starttime")
    timestamp = np.repeat(timestamp, height_bin_count).reshape(
        (sca_count, height_bin_count)
    )

    # Replace -1 sentinels with NaN
    extinction = np.where(extinction == -1, np.nan, extinction)

    # processing_qc_flag is a bit-packed valid-if-set field (1-indexed):
    #   Bit 1 (0x01): extinction valid
    #   Bit 2 (0x02): backscatter valid
    #   Bit 3 (0x04): Mie SNR valid
    #   Bit 4 (0x08): Rayleigh SNR valid
    #   Bit 5 (0x10): extinction error bar valid
    #   Bit 6 (0x20): backscatter error bar valid
    #   Bit 7 (0x40): cumulative LOD valid
    # We require at least the extinction bit to be set.
    # Also require at least 15 cloud-free measurements per BRC bin (same as MLE).
    valid_meas = np.sum(feature_mask, axis=1)  # (sca_count, height_bin_count)
    qc_pass = (
        ((processing_qc.astype(int) & 0x01) == 1)
        & ~np.isnan(extinction)
        & (valid_meas > 15)
    )

    sca_idx = np.repeat(np.arange(sca_count), height_bin_count)
    bin_idx = np.tile(np.arange(height_bin_count), sca_count)

    n = sca_count * height_bin_count
    orig_coords = [
        {
            "indices": np.array((sca_idx[i], bin_idx[i]), dtype=int),
            "shape": np.array((sca_count, height_bin_count), dtype=int),
            "names": np.array(("profile", "height_bin"), dtype=object),
        }
        for i in range(n)
    ]
    metadata = [
        {
            "profile_id": int(sca_idx[i]),
        }
        for i in range(n)
    ]

    extinction_flat = extinction.ravel() * 1e-6  # 1/Mm → 1/m
    if len(extinction_flat) == 0:
        return None

    return _make_obs_df(
        timestamp_flat=timestamp.ravel(),
        lon_flat=lon.ravel(),
        lat_flat=lat.ravel(),
        alt_flat=alt.ravel(),
        extinction_flat=extinction_flat,
        qc_flags=np.where(qc_pass.ravel(), 0, 1),
        instrument="AEOLUS_L2A_SCA",
        orig_coords=orig_coords,
        orig_filename=path.name,
        metadata=metadata,
    )


def _convert_ael_pro(
    cf,
    brc_count: int,
    meas_count: int,
    height_bin_count: int,
    feature_mask: np.ndarray,
    path: Path,
) -> pd.DataFrame | None:
    """Convert the AEL-PRO product.

    AEL-PRO is a per-measurement (finer resolution) product with brc_count × meas_count
    profiles, each with height_bin_count altitude bins. The BRC and measurement dimensions
    are flattened into a single profile dimension before any binning is applied.
    """
    lat = np.zeros((brc_count, meas_count, height_bin_count))
    lon = np.zeros((brc_count, meas_count, height_bin_count))
    alt = np.zeros((brc_count, meas_count, height_bin_count))
    alt_top = np.zeros((brc_count, meas_count, height_bin_count))
    alt_bottom = np.zeros((brc_count, meas_count, height_bin_count))
    extinction = np.zeros((brc_count, meas_count, height_bin_count))
    error_extinction = np.zeros((brc_count, meas_count, height_bin_count))
    classification = np.zeros((brc_count, meas_count, height_bin_count))
    # quality_index is per measurement (scalar), broadcast to all bins later
    quality_index = np.zeros((brc_count, meas_count))

    for i in range(brc_count):
        for m in range(meas_count):
            lat[i, m], lon[i, m], alt[i, m], alt_top[i, m], alt_bottom[i, m] = (
                _read_brc_geolocation(cf, i, m, height_bin_count)
            )
            extinction[i, m] = coda.fetch(
                cf,
                "ael_pro_opt_properties",
                i,
                "measurement_ael_pro_optical_properties",
                m,
                "height_bin_ael_pro_optical_properties",
                -1,
                "extinction",
            )
            error_extinction[i, m] = coda.fetch(
                cf,
                "ael_pro_pcd",
                i,
                "measurement_ael_pro_pcd",
                m,
                "height_bin_ael_pro_pcd",
                -1,
                "error_extinction",
            )
            classification[i, m] = coda.fetch(
                cf,
                "ael_pro_opt_properties",
                i,
                "measurement_ael_pro_optical_properties",
                m,
                "height_bin_ael_pro_optical_properties",
                -1,
                "classification",
            )
            quality_index[i, m] = coda.fetch(
                cf, "ael_pro_pcd", i, "measurement_ael_pro_pcd", m, "quality_index"
            )

    # In classification, aerosol-related keys are:
    # 3 and 103: Tropospheric Aerosol
    # 13 and 213: Stratospheric aerosol
    is_aerosol = (
        (classification == 3)
        | (classification == 103)
        | (classification == 13)
        | (classification == 213)
    )

    # Broadcast quality_index to all height bins
    quality_index_bins = quality_index[:, :, np.newaxis] * np.ones(
        (1, 1, height_bin_count)
    )
    # quality_index is a bit-packed problem-if-set field (0-indexed):
    #   Bit 0 (0x01): no retrieval
    #   Bit 1 (0x02): high chi_sq in optimal estimation (obs level)
    #   Bit 2 (0x04): high chi_sq in optimal estimation (measurement level)
    #   Bit 3 (0x08): maximum iterations reached (obs level)
    #   Bit 4 (0x10): maximum iterations reached (measurement level)
    #   Bit 5 (0x20): low SNR in retrieved extinction
    # Reject if any of bits 0–5 are set.
    # Also apply the per-measurement feature mask directly (AEL-PRO resolution matches).
    _AEL_PRO_PROBLEM_BITS = 0x3F
    qc_pass_3d = (
        ((quality_index_bins.astype(int) & _AEL_PRO_PROBLEM_BITS) == 0)
        & (extinction != -1)
        & is_aerosol
        # & feature_mask
    )

    # Replace -1 sentinels with NaN
    extinction = np.where(extinction == -1, np.nan, extinction)

    # BRC-level timestamps, repeated for each measurement and height bin
    timestamp = coda.fetch(cf, "ael_pro_opt_properties", -1, "starttime")
    timestamp = np.repeat(timestamp, meas_count * height_bin_count).reshape(
        (brc_count, meas_count, height_bin_count)
    )

    # Build brc_id array before flattening (needed for metadata)
    brc_ids = np.repeat(np.arange(brc_count), meas_count).reshape(
        (brc_count, meas_count)
    )

    # Flatten (brc, meas, H) → (brc*meas, H)
    profile_count = brc_count * meas_count
    lat = lat.reshape(profile_count, height_bin_count)
    lon = lon.reshape(profile_count, height_bin_count)
    alt = alt.reshape(profile_count, height_bin_count)
    alt_top = alt_top.reshape(profile_count, height_bin_count)
    alt_bottom = alt_bottom.reshape(profile_count, height_bin_count)
    extinction = extinction.reshape(profile_count, height_bin_count)
    timestamp = timestamp.reshape(profile_count, height_bin_count)
    qc_pass_2d = qc_pass_3d.reshape(profile_count, height_bin_count)
    brc_ids_flat = (
        brc_ids.ravel()
    )  # (profile_count,) — one BRC id per flattened profile
    meas_ids_flat = np.tile(np.arange(meas_count), brc_count)

    qc_pass = qc_pass_2d

    profile_idx = np.repeat(np.arange(profile_count), height_bin_count)
    bin_idx = np.tile(np.arange(height_bin_count), profile_count)

    n = profile_count * height_bin_count
    orig_coords = [
        {
            "indices": np.array((profile_idx[i], bin_idx[i]), dtype=int),
            "shape": np.array((profile_count, height_bin_count), dtype=int),
            "names": np.array(("profile", "height_bin"), dtype=object),
        }
        for i in range(n)
    ]

    alt_top_flat = alt_top.ravel()
    alt_bottom_flat = alt_bottom.ravel()
    # Expand brc/meas ids to all height bins for metadata
    brc_ids_per_obs = np.repeat(brc_ids_flat, height_bin_count)
    meas_ids_per_obs = np.repeat(meas_ids_flat, height_bin_count)
    metadata = [
        {
            "profile_id": int(profile_idx[i]),
            "brc_id": int(brc_ids_per_obs[i]),
            "measurement_id": int(meas_ids_per_obs[i]),
            "altitude_top": float(alt_top_flat[i])
            if not np.isnan(alt_top_flat[i])
            else None,
            "altitude_bottom": float(alt_bottom_flat[i])
            if not np.isnan(alt_bottom_flat[i])
            else None,
        }
        for i in range(n)
    ]

    extinction_flat = extinction.ravel() * 1e-6  # 1/Mm → 1/m
    if len(extinction_flat) == 0:
        return None

    return _make_obs_df(
        timestamp_flat=timestamp.ravel(),
        lon_flat=lon.ravel(),
        lat_flat=lat.ravel(),
        alt_flat=alt.ravel(),
        extinction_flat=extinction_flat,
        qc_flags=np.where(qc_pass.ravel(), 0, 1),
        instrument="AEOLUS_L2A_AEL_PRO",
        orig_coords=orig_coords,
        orig_filename=path.name,
        metadata=metadata,
    )


def convert_aeolus_l2a(
    path: Path,
    include_mle: bool = True,
    include_sca: bool = True,
    include_ael_pro: bool = True,
) -> pd.DataFrame | None:
    """Convert an AEOLUS L2A file to WRF-Ensembly Observation format.

    Args:
        path: Path to the AEOLUS L2A file.
        include_mle: Include the SCA-MLE product (instrument: AEOLUS_L2A_MLE).
        include_sca: Include the SCA product (instrument: AEOLUS_L2A_SCA).
        include_ael_pro: Include the AEL-PRO product (instrument: AEOLUS_L2A_AEL_PRO).

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format, or None if no valid data.
    """
    cf = coda.open(str(path))
    if cf.product_type != "ALD_U_N_2A":
        cf.close()
        raise ValueError("Input file is not an AEOLUS DBL L2A file.")

    num_brc = int(coda.fetch(cf, "sph", "num_brc"))
    mle_count = int(coda.fetch(cf, "sph", "num_prof_mle"))
    sca_count = int(coda.fetch(cf, "sph", "num_prof_sca"))
    meas_in_brc_count = int(coda.fetch(cf, "sph", "num_meas_max_brc"))
    height_bin_count = int(coda.fetch(cf, "sph", "num_bins_per_meas"))

    # Read the feature mask once; shape (num_brc, meas_in_brc_count, height_bin_count).
    # All three products use the same underlying mask — BRC-level products sum over
    # the measurement axis, AEL-PRO uses it per-measurement directly.
    # The feature mask always covers all num_brc BRCs, which may differ from
    # num_prof_mle/num_prof_sca in some files.
    print("  Reading feature mask...")
    feature_mask = _read_feature_mask(cf, num_brc, meas_in_brc_count, height_bin_count)

    parts = []

    if include_mle:
        print(
            f"  Converting MLE product ({mle_count} BRCs × {height_bin_count} bins)..."
        )
        df = _convert_mle(
            cf,
            mle_count,
            height_bin_count,
            feature_mask[:mle_count],
            path,
        )
        if df is not None:
            parts.append(df)

    if include_sca:
        print(
            f"  Converting SCA product ({sca_count} profiles × {height_bin_count} bins)..."
        )
        df = _convert_sca(
            cf,
            sca_count,
            height_bin_count,
            feature_mask[:sca_count],
            path,
        )
        if df is not None:
            parts.append(df)

    if include_ael_pro:
        print(
            f"  Converting AEL-PRO product "
            f"({num_brc} BRCs × {meas_in_brc_count} measurements × {height_bin_count} bins)..."
        )
        df = _convert_ael_pro(
            cf,
            num_brc,
            meas_in_brc_count,
            height_bin_count,
            feature_mask,
            path,
        )
        if df is not None:
            parts.append(df)

    cf.close()

    if not parts:
        return None

    result = pd.concat(parts, ignore_index=True)
    return result


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--mle/--no-mle", default=True, help="Include MLE product (AEOLUS_L2A_MLE)."
)
@click.option(
    "--sca/--no-sca", default=True, help="Include SCA product (AEOLUS_L2A_SCA)."
)
@click.option(
    "--ael-pro/--no-ael-pro",
    default=True,
    help="Include AEL-PRO product (AEOLUS_L2A_AEL_PRO).",
)
def aeolus_l2a(
    input_path: Path,
    output_path: Path,
    mle: bool,
    sca: bool,
    ael_pro: bool,
):
    """Convert AEOLUS L2A file to WRF-Ensembly observation format.

    INPUT_PATH: Path to the AEOLUS L2A file
    OUTPUT_PATH: Path where to save the converted observations (will be saved as parquet)
    """
    print(f"Converting AEOLUS L2A file: {input_path}")
    print(f"Output path: {output_path}")

    converted_df = convert_aeolus_l2a(
        input_path,
        include_mle=mle,
        include_sca=sca,
        include_ael_pro=ael_pro,
    )
    if converted_df is None or converted_df.empty:
        print("No observations found in the input file, aborting")
        return

    obs_io.write_obs(converted_df, output_path)

    total_obs = len(converted_df)
    good_obs = len(converted_df[converted_df["qc_flag"] == 0])
    bad_obs = total_obs - good_obs

    print(f"Successfully converted {total_obs} observations:")
    print(f"  - {good_obs} good quality observations (QC=0)")
    print(f"  - {bad_obs} flagged observations (QC>0)")
    print("Observations per instrument:")
    for instrument, count in converted_df["instrument"].value_counts().items():
        good = len(
            converted_df[
                (converted_df["instrument"] == instrument)
                & (converted_df["qc_flag"] == 0)
            ]
        )
        print(f"  - {instrument}: {count} total, {good} good quality")
    print(f"Saved to: {output_path}")
