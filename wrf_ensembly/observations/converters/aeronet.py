"""Converter for AERONET CSV-ish files to WRF-Ensembly Observation format."""

from pathlib import Path
from typing import List

import click
import numpy as np
import pandas as pd

from wrf_ensembly.observations import io as obs_io


def convert_aeronet(
    path: Path, quantities: list[str] = ["AOD_340", "AOD_500"]
) -> None | pd.DataFrame:
    """Convert an AERONET file to WRF-Ensembly Observation format.

    Args:
        path: Path to the AERONET tabular file.
        quantities: List of quantities to extract from the file. Default is ["AOD_550nm"].

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format (correct columns etc).
    """

    aeronet_df = pd.read_csv(path, skiprows=6, delimiter=",", encoding="latin-1")
    for col in quantities:
        if col not in aeronet_df.columns:
            raise ValueError(
                f"Requested quantity '{col}' not found in AERONET file columns"
            )

    # Ensure TZ is UTC
    aeronet_df["timestamp"] = pd.to_datetime(
        aeronet_df["Date(dd:mm:yyyy)"] + " " + aeronet_df["Time(hh:mm:ss)"],
        format="%d:%m:%Y %H:%M:%S",
    )
    aeronet_df["timestamp"] = aeronet_df["timestamp"].dt.tz_localize("UTC")
    aeronet_df = aeronet_df.drop(columns=["Date(dd:mm:yyyy)", "Time(hh:mm:ss)"])

    # -999 is used as a missing data marker
    aeronet_df = aeronet_df.replace(-999.0, pd.NA)

    # The AERONET files have many columns (wide format), we need to convert them to long format
    # where each row only has one quantity.
    # AERONET files don't have any uncertainty, so we will set that to NaN for now.
    aeronet_df = aeronet_df.reset_index().melt(
        id_vars=[
            "index",
            "timestamp",
            "Site_Latitude(Degrees)",
            "Site_Longitude(Degrees)",
        ],
        value_vars=quantities,
        var_name="quantity",
        value_name="value",
    )

    # Drop rows without values
    aeronet_df = aeronet_df.dropna(subset=["value"]).reset_index(drop=True)
    if aeronet_df.empty:
        return None  # Nothing to do

    # Ensure the value column is float
    aeronet_df["value"] = aeronet_df["value"].astype(float)

    # Rename existing columns to the correct names, assign the rest
    aeronet_df = aeronet_df.rename(
        columns={
            "Site_Latitude(Degrees)": "latitude",
            "Site_Longitude(Degrees)": "longitude",
            "timestamp": "time",
        }
    )
    aeronet_df["instrument"] = "AERONET"
    aeronet_df["z"] = 0.0
    aeronet_df["z_type"] = "surface"
    aeronet_df["value_uncertainty"] = pd.NA
    aeronet_df["qc_flag"] = 0  # No QC available, so set to "good"
    aeronet_df["orig_coords"] = aeronet_df.apply(
        lambda row: {
            "indices": np.array((row["index"],), dtype=int),
            "shape": np.array((aeronet_df.shape[0],), dtype=int),
            "names": np.array(("row",), dtype=object),
        },
        axis=1,
    )
    aeronet_df["orig_filename"] = path.name
    aeronet_df["metadata"] = pd.NA  # No additional metadata available

    aeronet_df = aeronet_df.drop(columns=["index"])

    # Sort columns as defined in the schema, do the sanity check
    aeronet_df = aeronet_df[obs_io.REQUIRED_COLUMNS]
    obs_io.validate_schema(aeronet_df)

    return aeronet_df


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--quantities",
    multiple=True,
    default=["AOD_380nm", "AOD_500nm"],
    help="Quantities to extract from the AERONET file. Can be specified multiple times.",
)
def aeronet(input_path: Path, output_path: Path, quantities: List[str]):
    """Convert AERONET CSV file to WRF-Ensembly observation format.

    INPUT_PATH: Path to the AERONET CSV file
    OUTPUT_PATH: Path where to save the converted observations (will be saved as parquet)
    """

    print(f"Converting AERONET file: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Quantities: {', '.join(quantities)}")

    # Convert the data
    converted_df = convert_aeronet(input_path, list(quantities))
    if converted_df is None or converted_df.empty:
        print("No observations found in the input file, aborting")
        return

    # Save to output path as parquet
    obs_io.write_obs(converted_df, output_path)

    print(f"Successfully converted {len(converted_df)} observations")
    print(f"Saved to: {output_path}")
