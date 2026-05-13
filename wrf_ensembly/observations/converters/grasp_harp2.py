"""Converter for GRASP-HARP2 AOD observations."""

from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr

from wrf_ensembly.observations import io as obs_io

BAND_TO_QUANTITY: dict[str, str] = {
    "440.16": "AOD_440nm",
    "549.57": "AOD_550nm",
    "664.57": "AOD_665nm",
    "865.63": "AOD_870nm",
}


def convert_grasp_harp2(
    nc_path: Path,
    disabled_bands: tuple[str, ...] = (),
) -> pd.DataFrame | None:
    """Convert a GRASP HARP2 netCDF file to WRF-Ensembly Observation format.

    Args:
        nc_path: Path to the GRASP HARP2 netCDF file.
        disabled_bands: Band wavelength strings to exclude (e.g. ("440.16",)).

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format, or None if no valid
        observations remain after filtering.
    """

    ds = xr.open_dataset(nc_path, decode_times=True)

    latitude = ds["latitude"].values  # (y, x)
    longitude = ds["longitude"].values  # (y, x)
    aod_total = ds["aerosol_optical_depth_total"].values  # (y, x, band)
    bands = ds["band"].values.tolist()  # list of str

    # Scalar time — broadcast to all pixels
    t_decoded = pd.Timestamp(ds["t"].values.item()).tz_localize("UTC")

    y_size, x_size, n_bands = aod_total.shape
    orig_shape = (y_size, x_size, n_bands)
    orig_names = ("y", "x", "band")

    all_dfs: list[pd.DataFrame] = []

    for band_idx, band_str in enumerate(bands):
        if band_str in disabled_bands:
            continue
        if band_str not in BAND_TO_QUANTITY:
            raise ValueError(
                f"Unmapped HARP2 band '{band_str}'. Known bands: {list(BAND_TO_QUANTITY.keys())}"
            )
        quantity = BAND_TO_QUANTITY[band_str]

        aod = aod_total[:, :, band_idx]  # (y, x)
        valid_mask = ~np.isnan(aod) & ~np.isnan(latitude) & ~np.isnan(longitude)

        if not np.any(valid_mask):
            continue

        aod_flat = aod.flatten()
        lat_flat = latitude.flatten()
        lon_flat = longitude.flatten()
        valid_flat = valid_mask.flatten()

        y_indices, x_indices = np.meshgrid(
            np.arange(y_size), np.arange(x_size), indexing="ij"
        )
        y_flat = y_indices.flatten()
        x_flat = x_indices.flatten()

        n = len(aod_flat)
        orig_coords = [
            {
                "indices": (int(y_flat[i]), int(x_flat[i]), band_idx),
                "shape": orig_shape,
                "names": orig_names,
            }
            for i in range(n)
        ]

        # Come up with some uncertainty based on AOD because there is nothing in the file
        uncertainty = np.maximum(0.05 + 0.15 * aod_flat, 0.05)

        df = pd.DataFrame(
            {
                "instrument": "GRASP_HARP2",
                "quantity": quantity,
                "time": t_decoded,
                "latitude": lat_flat,
                "longitude": lon_flat,
                "z": 0.0,
                "z_type": "columnar",
                "value": aod_flat,
                "value_uncertainty": uncertainty,
                "qc_flag": (~valid_flat).astype(int),
                "orig_filename": nc_path.name,
                "metadata": pd.NA,
            }
        )
        df["orig_coords"] = orig_coords

        all_dfs.append(df)

    if not all_dfs:
        return None

    result = pd.concat(all_dfs, ignore_index=True)
    result = result[result["qc_flag"] == 0].copy()
    result["qc_flag"] = 0

    if result.empty:
        return None

    result = result[obs_io.REQUIRED_COLUMNS]
    return result


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--disable-band",
    "disabled_bands",
    multiple=True,
    type=click.Choice(list(BAND_TO_QUANTITY.keys())),
    help="Band to exclude (repeatable). E.g. --disable-band 440.16",
)
def grasp_harp2(
    input_path: Path,
    output_path: Path,
    disabled_bands: tuple[str, ...],
):
    """Convert a GRASP HARP2 netCDF file to WRF-Ensembly observation format.

    INPUT_PATH: Path to the GRASP HARP2 netCDF file.
    OUTPUT_PATH: Path to save the converted parquet file.

    Converts aerosol_optical_depth_total from all four HARP2 bands (440, 550, 665, 870 nm)
    into multi-quantity observations. Use --disable-band to exclude specific bands.
    """

    print(f"Converting GRASP HARP2 file: {input_path}")
    if disabled_bands:
        print(f"Disabled bands: {', '.join(disabled_bands)}")
    print(f"Output path: {output_path}")

    converted_df = convert_grasp_harp2(input_path, disabled_bands=disabled_bands)

    if converted_df is None or converted_df.empty:
        print("No valid observations found in the input file, aborting")
        return

    obs_io.write_obs(converted_df, output_path)

    counts = converted_df["quantity"].value_counts()
    for qty, n in counts.items():
        print(f"  {qty}: {n} observations")
    print(f"Total: {len(converted_df)} observations saved to {output_path}")
