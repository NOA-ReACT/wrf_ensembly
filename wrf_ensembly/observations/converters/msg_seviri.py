"""Converter for MSG SEVIRI brightness temperature observations."""

from pathlib import Path

import click
import numpy as np
import pandas as pd
import xarray as xr

from wrf_ensembly.observations import io as obs_io

HAS_SATPY = False
try:
    import satpy
    from pyresample import create_area_def

    HAS_SATPY = True
except ImportError:
    pass

SEVIRI_CHANNELS = {
    "WV_062": "BT_WV62",
    "WV_073": "BT_WV73",
    "IR_087": "BT_IR87",
    "IR_108": "BT_IR108",
    "IR_120": "BT_IR120",
}


def get_index_tuples(arr):
    """Get array of index tuples for flattened N-dimensional array."""
    flat_indices = np.arange(arr.size)
    return [np.unravel_index(i, arr.shape) for i in flat_indices]


def _build_wrf_target_area(wrf_path: Path):
    """Build a pyresample area definition from a WRF file's projection attributes.

    Args:
        wrf_path: Path to a WRF file (wrfinput, wrfout, or forecast_mean/sd).

    Returns:
        Tuple of (target_area, lons_2d, lats_2d).
    """

    wrf = xr.open_dataset(wrf_path).isel(t=0).set_coords(["latitude", "longitude"])

    target_area = create_area_def(
        "wrf",
        {
            "x_0": 0,
            "y_0": 0,
            "a": 6370000,
            "b": 6370000,
            "proj": "lcc",
            "lat_1": wrf.attrs["TRUELAT1"],
            "lat_2": wrf.attrs["TRUELAT2"],
            "lat_0": wrf.attrs["CEN_LAT"],
            "lon_0": wrf.attrs["STAND_LON"],
        },
        resolution=wrf.attrs["DX"],
        area_extent=[
            wrf.x.min().item(),
            wrf.y.min().item(),
            wrf.x.max().item(),
            wrf.y.max().item(),
        ],
    )

    lons, lats = target_area.get_lonlats()
    return target_area, lons, lats


def convert_msg_seviri(
    seviri_path: Path,
    wrf_path: Path,
    uncertainty: float = 2.0,
) -> pd.DataFrame | None:
    """Convert a MSG SEVIRI native file to WRF-Ensembly Observation format.

    Reads SEVIRI brightness temperatures using satpy, resamples them onto the
    WRF grid using pyresample's bucket averaging, and returns a standardized
    observation DataFrame.

    Args:
        seviri_path: Path to the MSG SEVIRI native file.
        wrf_path: Path to a WRF file for grid definition.
        uncertainty: Brightness temperature uncertainty in K.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format, or None if no
        valid observations are found.
    """

    if not HAS_SATPY:
        raise ImportError(
            "satpy and pyresample are required to read MSG SEVIRI files. Install them with: pip install satpy pyresample"
        )

    # Load SEVIRI scene
    scn = satpy.Scene(filenames=[str(seviri_path)], reader="seviri_l1b_native")
    scn.load(list(SEVIRI_CHANNELS.keys()))

    # Build WRF target area and resample
    target_area, lons, lats = _build_wrf_target_area(wrf_path)
    resampled = scn.resample(target_area, resampler="bucket_avg")

    # Extract observation time from the scene
    first_channel = next(iter(SEVIRI_CHANNELS.keys()))
    obs_time = pd.Timestamp(scn[first_channel].attrs["start_time"], tz="UTC")

    # Process each channel
    all_dfs = []
    for seviri_name, quantity_name in SEVIRI_CHANNELS.items():
        data = np.array(resampled[seviri_name])
        valid_mask = ~np.isnan(data)

        if not np.any(valid_mask):
            continue

        shape_2d = data.shape
        indices = get_index_tuples(data)

        # Flatten
        data_flat = data.flatten()
        lats_flat = lats.flatten()
        lons_flat = lons.flatten()
        valid_flat = valid_mask.flatten()

        df = pd.DataFrame(
            {
                "instrument": "MSG_SEVIRI",
                "quantity": quantity_name,
                "time": obs_time,
                "latitude": lats_flat,
                "longitude": lons_flat,
                "z": 0.0,
                "z_type": "columnar",
                "value": data_flat,
                "value_uncertainty": uncertainty,
                "qc_flag": 0,
                "orig_filename": Path(seviri_path).name,
                "metadata": pd.NA,
            }
        )

        df["orig_coords"] = [
            {"indices": indices[i], "names": ("y", "x"), "shape": shape_2d}
            for i in range(df.shape[0])
        ]

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
    "--wrf-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to a WRF file (wrfinput, wrfout, or forecast_mean) for grid definition.",
)
@click.option(
    "--uncertainty",
    default=2.0,
    type=float,
    help="Brightness temperature uncertainty in K (default: 2.0).",
)
def msg_seviri(
    input_path: Path,
    output_path: Path,
    wrf_file: Path,
    uncertainty: float,
):
    """Convert MSG SEVIRI native file to WRF-Ensembly observation format.

    INPUT_PATH: Path to the input MSG SEVIRI native file.
    OUTPUT_PATH: Path to save the converted parquet file.

    This converter reads SEVIRI brightness temperatures for channels WV_062,
    WV_073, IR_087, IR_108, and IR_120, resamples them onto the WRF grid,
    and saves the result in the standardized observation format.
    """

    print(f"Converting MSG SEVIRI file: {input_path}")
    print(f"Output path: {output_path}")
    print(f"WRF file: {wrf_file}")
    print(f"Uncertainty: {uncertainty} K")

    converted_df = convert_msg_seviri(input_path, wrf_file, uncertainty)

    if converted_df is None or converted_df.empty:
        print("No valid observations found in the input file, aborting")
        return

    obs_io.write_obs(converted_df, output_path)

    print(f"Successfully converted {len(converted_df)} observations")
    print(f"Saved to: {output_path}")
