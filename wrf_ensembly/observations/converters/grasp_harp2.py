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

BAND_TO_FINE_QUANTITY: dict[str, str] = {
    "440.16": "AOD_Fine_440nm",
    "549.57": "AOD_Fine_550nm",
    "664.57": "AOD_Fine_665nm",
    "865.63": "AOD_Fine_870nm",
}

BAND_TO_COARSE_QUANTITY: dict[str, str] = {
    "440.16": "AOD_Coarse_440nm",
    "549.57": "AOD_Coarse_550nm",
    "664.57": "AOD_Coarse_665nm",
    "865.63": "AOD_Coarse_870nm",
}


def convert_grasp_harp2(
    nc_path: Path,
    disabled_bands: tuple[str, ...] = (),
    imerg_path: Path | None = None,
    disable_fine_mode: bool = False,
    disable_coarse_mode: bool = False,
) -> pd.DataFrame | None:
    """Convert a GRASP HARP2 netCDF file to WRF-Ensembly Observation format.

    Args:
        nc_path: Path to the GRASP HARP2 netCDF file.
        disabled_bands: Band wavelength strings to exclude (e.g. ("440.16",)).
        imerg_path: Optional path to IMERG Land/Sea mask netCDF. When provided, each
            observation is tagged with ``is_over_land`` in its metadata (0 = land, 1 = sea).
        disable_fine_mode: Skip aerosol_fine_mode_optical_depth conversion.
        disable_coarse_mode: Skip aerosol_coarse_mode_optical_depth conversion.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format, or None if no valid
        observations remain after filtering.
    """

    ds = xr.open_dataset(nc_path, decode_times=True)

    # Time has some extra dimensions (instrument, measurement_type, observation) but
    # the data is the same in all of these, so we just pick the minimum value. Using
    # the maximum would produce the same value.
    time = (
        ds["measurement_time_unix"]
        .min(dim=("instrument", "measurement_type", "observation"))
        .astype("datetime64[s]")
    ).values  # (y, x, band)
    latitude = ds["latitude"].values  # (y, x)
    longitude = ds["longitude"].values  # (y, x)
    bands = ds["band"].values.tolist()  # list of str

    aod_total = ds["aerosol_optical_depth_total"].values  # (y, x, band)
    y_size, x_size, _ = aod_total.shape
    orig_shape = (y_size, x_size)
    orig_names = ("y", "x")

    datasets: list[tuple[np.ndarray, dict[str, str]]] = [
        (aod_total, BAND_TO_QUANTITY),
    ]
    if not disable_fine_mode:
        datasets.append(
            (ds["aerosol_fine_mode_optical_depth"].values, BAND_TO_FINE_QUANTITY)
        )
    if not disable_coarse_mode:
        datasets.append(
            (ds["aerosol_coarse_mode_optical_depth"].values, BAND_TO_COARSE_QUANTITY)
        )

    # Compute IMERG land/sea mask on the full spatial grid once, reused per band.
    # I am using 100 as a strict ocean mask. The documentation mentions:
    # > Land sea is also called land ocean or land water. Typical percentage thresholds
    # > used to define sea are 100% (strictly open water) or 75% (including sea-ward
    # > coast areas). Typical percentages used to define strictly land are 25% or 15%;
    # > too low a percentage masks out humid regions that have many lakes and reservoirs.
    # > Users should inspect the resulting masks to check that they correspond to the expected map.
    # Source: https://web.archive.org/web/20250209035349/https://gpm.nasa.gov/data/directory/imerg-land-sea-mask-netcdf/
    is_over_land_flat: np.ndarray | None = None
    if imerg_path is not None:
        imerg_ds = xr.open_dataset(imerg_path)
        imerg_ds = imerg_ds.assign_coords(
            lon=(((imerg_ds.lon + 180) % 360) - 180)
        ).sortby("lon")
        lat_da = xr.DataArray(latitude.flatten(), dims="points")
        lon_da = xr.DataArray(longitude.flatten(), dims="points")
        interp = imerg_ds["landseamask"].interp(lat=lat_da, lon=lon_da, method="linear")
        is_over_land_flat = (interp.values != 100).astype(int)

    all_dfs: list[pd.DataFrame] = []

    y_indices, x_indices = np.meshgrid(
        np.arange(y_size), np.arange(x_size), indexing="ij"
    )
    y_flat = y_indices.flatten()
    x_flat = x_indices.flatten()

    for aod_data, band_to_qty in datasets:
        for band_idx, band_str in enumerate(bands):
            if band_str in disabled_bands:
                continue
            if band_str not in band_to_qty:
                raise ValueError(
                    f"Unmapped HARP2 band '{band_str}'. Known bands: {list(band_to_qty.keys())}"
                )
            quantity = band_to_qty[band_str]

            aod = aod_data[:, :, band_idx]  # (y, x)
            time_band = time[:, :, band_idx]  # (y, x)
            valid_mask = (
                ~np.isnan(aod) & ~np.isnan(latitude) & ~np.isnan(longitude) & (aod >= 0)
            )

            if not np.any(valid_mask):
                continue

            aod_flat = aod.flatten()
            t_flat = time_band.flatten()
            lat_flat = latitude.flatten()
            lon_flat = longitude.flatten()
            valid_flat = valid_mask.flatten()

            n = len(aod_flat)
            orig_coords = [
                {
                    "indices": (int(y_flat[i]), int(x_flat[i])),
                    "shape": orig_shape,
                    "names": orig_names,
                }
                for i in range(n)
            ]

            if is_over_land_flat is not None:
                metadata_list = [
                    {"is_over_land": int(is_over_land_flat[i])} for i in range(n)
                ]
            else:
                metadata_list = [pd.NA] * n

            # Uncertainty based on 1-week O-B analysis
            uncertainty = np.maximum(aod_flat * 0.2853 + 0.0267, 0.0267)
            uncertainty[uncertainty > 0.3] = 0.3

            df = pd.DataFrame(
                {
                    "instrument": "GRASP_HARP2",
                    "quantity": quantity,
                    "time": t_flat,
                    "latitude": lat_flat,
                    "longitude": lon_flat,
                    "z": 0.0,
                    "z_type": "columnar",
                    "value": aod_flat,
                    "value_uncertainty": uncertainty,
                    "qc_flag": (~valid_flat).astype(int),
                    "orig_filename": nc_path.name,
                    "metadata": metadata_list,
                }
            )
            df["orig_coords"] = orig_coords
            df["time"] = df["time"].dt.tz_localize("UTC")

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
@click.option(
    "--imerg-land-sea",
    "imerg_land_sea",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to IMERG Land/Sea mask netCDF to tag observations with is_over_land (0=land, 1=sea).",
)
@click.option(
    "--disable-fine-mode",
    "disable_fine_mode",
    is_flag=True,
    default=False,
    help="Skip aerosol_fine_mode_optical_depth conversion.",
)
@click.option(
    "--disable-coarse-mode",
    "disable_coarse_mode",
    is_flag=True,
    default=False,
    help="Skip aerosol_coarse_mode_optical_depth conversion.",
)
def grasp_harp2(
    input_path: Path,
    output_path: Path,
    disabled_bands: tuple[str, ...],
    imerg_land_sea: Path | None,
    disable_fine_mode: bool,
    disable_coarse_mode: bool,
):
    """Convert a GRASP HARP2 netCDF file to WRF-Ensembly observation format.

    INPUT_PATH: Path to the GRASP HARP2 netCDF file.
    OUTPUT_PATH: Path to save the converted parquet file.

    Converts aerosol_optical_depth_total, aerosol_fine_mode_optical_depth, and
    aerosol_coarse_mode_optical_depth from all four HARP2 bands (440, 550, 665, 870 nm)
    into multi-quantity observations. Use --disable-band to exclude specific bands, or
    --disable-fine-mode / --disable-coarse-mode to skip those datasets entirely.
    """

    print(f"Converting GRASP HARP2 file: {input_path}")
    if disabled_bands:
        print(f"Disabled bands: {', '.join(disabled_bands)}")
    if disable_fine_mode:
        print("Fine mode AOD disabled")
    if disable_coarse_mode:
        print("Coarse mode AOD disabled")
    if imerg_land_sea:
        print(f"IMERG Land/Sea mask: {imerg_land_sea}")
    print(f"Output path: {output_path}")

    converted_df = convert_grasp_harp2(
        input_path,
        disabled_bands=disabled_bands,
        imerg_path=imerg_land_sea,
        disable_fine_mode=disable_fine_mode,
        disable_coarse_mode=disable_coarse_mode,
    )

    if converted_df is None or converted_df.empty:
        print("No valid observations found in the input file, aborting")
        return

    obs_io.write_obs(converted_df, output_path)

    counts = converted_df["quantity"].value_counts()
    for qty, n in counts.items():
        print(f"  {qty}: {n} observations")
    print(f"Total: {len(converted_df)} observations saved to {output_path}")
