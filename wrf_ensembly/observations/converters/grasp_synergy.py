"""Converter for GRASP synergy (OLCI + TROPOMI) AOD observations."""

from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr

from wrf_ensembly.observations import io as obs_io

# The synergy product retrieves over the union of the OLCI (412.5, 442.5, 490.0, 510.0,
# 560.0, 665.0, 753.0, 865.0, 1020.0) and TROPOMI (340, 367, 380, 416, 440, 494, 670,
# 747, 772, 2313) band sets. Only bands landing within ~10nm of a quantity we already
# know about are converted, the rest are skipped.
# Where both instruments contribute a nearby band (442.5/440, 490.0/494, 670/665.0) only
# one of the pair is mapped on purpose: GRASP fits a single aerosol state across all 19
# wavelengths, so two neighbouring bands are the same retrieval sampled twice and
# assimilating both would double-count it.
BAND_TO_QUANTITY: dict[float, str] = {
    440.0: "AOD_440nm",
    494.0: "AOD_500nm",
    560.0: "AOD_550nm",
    665.0: "AOD_665nm",
    865.0: "AOD_870nm",
}

BAND_TO_FINE_QUANTITY: dict[float, str] = {
    440.0: "AOD_Fine_440nm",
    494.0: "AOD_Fine_500nm",
    560.0: "AOD_Fine_550nm",
    665.0: "AOD_Fine_665nm",
    865.0: "AOD_Fine_870nm",
}

BAND_TO_COARSE_QUANTITY: dict[float, str] = {
    440.0: "AOD_Coarse_440nm",
    494.0: "AOD_Coarse_500nm",
    560.0: "AOD_Coarse_550nm",
    665.0: "AOD_Coarse_665nm",
    865.0: "AOD_Coarse_870nm",
}

MAPPED_BAND_LABELS = ["440", "494", "560.0", "665.0", "865.0"]
"""The mapped bands as they are labelled inside the files, for --disable-band."""

UNCERTAINTY_INTERCEPT = 0.05
UNCERTAINTY_SLOPE = 0.15
"""
Generic AOD error envelope, `0.05 + 0.15 * AOD`. Unlike the grasp_harp2 coefficients this
is a literature default rather than a fit against this product's first departures, so
refit it once synergy O-B statistics are available.
"""


def _wavelength(band_label: str) -> float | None:
    """Band labels are inconsistently formatted ('440' vs '560.0'), so compare as floats.

    Returns None for labels that aren't a plain wavelength, which are then skipped.
    """

    try:
        return float(band_label)
    except ValueError:
        return None


def _nullable(value: float) -> float | None:
    """NaN isn't representable in JSON, so a missing QA flag becomes a null instead."""

    return None if np.isnan(value) else float(value)


def convert_grasp_synergy(
    nc_path: Path,
    disabled_bands: tuple[str, ...] = (),
    disable_fine_mode: bool = False,
    disable_coarse_mode: bool = False,
    keep_invalid_pixels: bool = False,
) -> pd.DataFrame | None:
    """Convert a GRASP synergy netCDF file to WRF-Ensembly Observation format.

    Args:
        nc_path: Path to the GRASP synergy netCDF file.
        disabled_bands: Band labels to exclude (e.g. ("440", "865.0")).
        disable_fine_mode: Skip aerosol_fine_mode_optical_depth conversion.
        disable_coarse_mode: Skip aerosol_coarse_mode_optical_depth conversion.
        keep_invalid_pixels: Emit every pixel, with the ones failing the validity check
            carrying a NaN value and qc_flag=1, instead of dropping them. Granules are
            mostly empty so this inflates the output considerably; it is meant for
            inspecting a file, not for assimilation.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format, or None if no valid
        observations remain after filtering.
    """

    ds = xr.open_dataset(nc_path, decode_times=True)

    if ds.sizes["t"] != 1:
        raise ValueError(
            f"Expected one timestep in {nc_path.name}, found {ds.sizes['t']}. "
            "Multi-timestep synergy files are not supported."
        )
    ds = ds.isel(t=0)

    # The whole granule shares one timestamp. There is also a per-pixel `real_time` but
    # it carries no units attribute, so we stick to the CF-decoded coordinate.
    time = pd.Timestamp(ds["t"].values).round("s").tz_localize("UTC")

    latitude = ds["latitude"].values  # (y, x)
    longitude = ds["longitude"].values  # (y, x)
    bands = ds["band"].values.tolist()  # list of str

    aod_total = ds["aerosol_optical_depth_total"].values  # (y, x, band)
    y_size, x_size, _ = aod_total.shape
    orig_shape = (y_size, x_size)
    orig_names = ("y", "x")

    datasets: list[tuple[np.ndarray, dict[float, str]]] = [
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

    disabled_wavelengths = {_wavelength(band) for band in disabled_bands}

    # Unlike HARP2, the synergy product carries its own land/sea information, so no
    # external IMERG mask is needed. land_percentage is how much of the pixel is land,
    # so anything above zero contains some; this matches the strict-ocean threshold
    # grasp_harp2.py applies to IMERG.
    is_over_land_flat = (ds["land_percentage"].values > 0).astype(int).flatten()
    qa_normal_flat = ds["qa_flag_normal"].values.flatten()
    qa_extended_flat = ds["qa_flag_extended"].values.flatten()

    all_dfs: list[pd.DataFrame] = []

    y_indices, x_indices = np.meshgrid(
        np.arange(y_size), np.arange(x_size), indexing="ij"
    )
    y_flat = y_indices.flatten()
    x_flat = x_indices.flatten()
    lat_flat = latitude.flatten()
    lon_flat = longitude.flatten()

    for aod_data, band_to_qty in datasets:
        for band_idx, band_label in enumerate(bands):
            wavelength = _wavelength(band_label)
            if (
                wavelength is None
                or wavelength in disabled_wavelengths
                or wavelength not in band_to_qty
            ):
                continue
            quantity = band_to_qty[wavelength]

            aod_flat = aod_data[:, :, band_idx].flatten()
            valid_flat = (
                ~np.isnan(aod_flat)
                & ~np.isnan(lat_flat)
                & ~np.isnan(lon_flat)
                & (aod_flat >= 0)
            )

            if not np.any(valid_flat):
                continue

            # Granules are mostly empty, so by default only the surviving pixels are
            # ever materialised - building an orig_coords dict per grid point costs
            # millions of throwaway objects otherwise. reconstruct_array() puts the
            # observations back onto the full grid from orig_coords, so nothing
            # downstream needs the dropped rows to be present.
            if keep_invalid_pixels:
                sel = np.arange(valid_flat.size)
            else:
                sel = np.flatnonzero(valid_flat)

            orig_coords = [
                {
                    "indices": (int(y_flat[i]), int(x_flat[i])),
                    "shape": orig_shape,
                    "names": orig_names,
                }
                for i in sel
            ]
            metadata_list = [
                {
                    "is_over_land": int(is_over_land_flat[i]),
                    "qa_flag_normal": _nullable(qa_normal_flat[i]),
                    "qa_flag_extended": _nullable(qa_extended_flat[i]),
                }
                for i in sel
            ]

            values = aod_flat[sel]
            df = pd.DataFrame(
                {
                    "instrument": "GRASP_SYNERGY",
                    "quantity": quantity,
                    "time": time,
                    "latitude": lat_flat[sel],
                    "longitude": lon_flat[sel],
                    "z": 0.0,
                    "z_type": "columnar",
                    "value": values,
                    "value_uncertainty": UNCERTAINTY_INTERCEPT
                    + UNCERTAINTY_SLOPE * values,
                    "qc_flag": (~valid_flat[sel]).astype(int),
                    "orig_filename": nc_path.name,
                    "metadata": metadata_list,
                }
            )
            df["orig_coords"] = orig_coords

            all_dfs.append(df)

    if not all_dfs:
        return None

    result = pd.concat(all_dfs, ignore_index=True)
    result = result[obs_io.REQUIRED_COLUMNS]
    return result


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--disable-band",
    "disabled_bands",
    multiple=True,
    type=click.Choice(MAPPED_BAND_LABELS),
    help="Band to exclude (repeatable). E.g. --disable-band 440",
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
@click.option(
    "--keep-invalid-pixels",
    "keep_invalid_pixels",
    is_flag=True,
    default=False,
    help="Emit pixels with no retrieval too (NaN value, qc_flag=1) instead of dropping "
    "them. Produces much larger files; useful for inspecting a granule.",
)
def grasp_synergy(
    input_path: Path,
    output_path: Path,
    disabled_bands: tuple[str, ...],
    disable_fine_mode: bool,
    disable_coarse_mode: bool,
    keep_invalid_pixels: bool,
):
    """Convert a GRASP synergy netCDF file to WRF-Ensembly observation format.

    INPUT_PATH: Path to the GRASP synergy netCDF file.
    OUTPUT_PATH: Path to save the converted parquet file.

    Converts aerosol_optical_depth_total, aerosol_fine_mode_optical_depth, and
    aerosol_coarse_mode_optical_depth. Of the product's 19 bands only the five that sit
    close to a known quantity are used (440, 494, 560, 665, 865 nm, mapped onto 440, 500,
    550, 665 and 870 nm); the remaining bands are skipped. Use --disable-band to exclude
    specific bands, or --disable-fine-mode / --disable-coarse-mode to skip those datasets
    entirely.
    """

    print(f"Converting GRASP synergy file: {input_path}")
    if disabled_bands:
        print(f"Disabled bands: {', '.join(disabled_bands)}")
    if disable_fine_mode:
        print("Fine mode AOD disabled")
    if disable_coarse_mode:
        print("Coarse mode AOD disabled")
    if keep_invalid_pixels:
        print("Keeping pixels without a retrieval (qc_flag=1)")
    print(f"Output path: {output_path}")

    converted_df = convert_grasp_synergy(
        input_path,
        disabled_bands=disabled_bands,
        disable_fine_mode=disable_fine_mode,
        disable_coarse_mode=disable_coarse_mode,
        keep_invalid_pixels=keep_invalid_pixels,
    )

    if converted_df is None or converted_df.empty:
        print("No valid observations found in the input file, aborting")
        return

    obs_io.write_obs(converted_df, output_path)

    counts = converted_df["quantity"].value_counts()
    for qty, n in counts.items():
        print(f"  {qty}: {n} observations")
    print(f"Total: {len(converted_df)} observations saved to {output_path}")
