"""Converter for AERONET CSV-ish files to WRF-Ensembly Observation format."""

import re
from pathlib import Path
from typing import List

import click
import numpy as np
import pandas as pd

from wrf_ensembly.observations import io as obs_io

MISSING_VALUE = -999.0
"""AERONET writes -999 (in various formattings) wherever a field is missing."""

AOD_UNCERTAINTY_FLOOR = 0.01
"""
Nominal AERONET AOD uncertainty. The per-observation error is this added in quadrature
with the triplet variability of the matching wavelength.
"""

DATE_COLUMN = "Date(dd:mm:yyyy)"
TIME_COLUMN = "Time(hh:mm:ss)"
LATITUDE_COLUMN = "Site_Latitude(Degrees)"
LONGITUDE_COLUMN = "Site_Longitude(Degrees)"

METADATA_COLUMNS = {
    # metadata key -> AERONET column, for fields that are the same for all wavelengths
    "site_name": "AERONET_Site_Name",
    "instrument_number": "AERONET_Instrument_Number",
    "data_quality_level": "Data_Quality_Level",
    "last_date_processed": "Last_Date_Processed",
    "solar_zenith_angle": "Solar_Zenith_Angle(Degrees)",
    "optical_air_mass": "Optical_Air_Mass",
    "angstrom_440_870": "440-870_Angstrom_Exponent",
    "angstrom_500_870": "500-870_Angstrom_Exponent",
    "n_wavelengths": "Number_of_Wavelengths",
}

INT_METADATA_KEYS = ("instrument_number", "n_wavelengths")

AOD_QUANTITY_RE = re.compile(r"^AOD_(?P<wavelength>[0-9.]+)nm$")


def _wavelength_columns(quantity: str) -> tuple[str | None, str | None]:
    """
    Names of the triplet variability and exact wavelength columns matching an AOD
    quantity, or (None, None) if the quantity isn't a per-wavelength AOD.

    Note the inconsistent naming in the AERONET files: the triplet columns drop the
    `nm` suffix (`Triplet_Variability_500`) while the exact wavelength ones keep it
    (`Exact_Wavelengths_of_AOD(um)_500nm`).
    """

    match = AOD_QUANTITY_RE.match(quantity)
    if match is None:
        return None, None

    wavelength = match.group("wavelength")
    return (
        f"Triplet_Variability_{wavelength}",
        f"Exact_Wavelengths_of_AOD(um)_{wavelength}nm",
    )


def _nullable(value: np.floating) -> float | None:
    """NaN isn't representable in JSON, so a missing field becomes a null instead."""

    return None if np.isnan(value) else float(value)


def convert_aeronet(
    path: Path, quantities: list[str] = ["AOD_380nm", "AOD_500nm"]
) -> None | pd.DataFrame:
    """Convert an AERONET file to WRF-Ensembly Observation format.

    Args:
        path: Path to the AERONET tabular file.
        quantities: List of quantities to extract from the file.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format (correct columns etc).
    """

    # AERONET files carry 113 columns and the big stations reach half a million rows,
    # so only parse what we actually need. -999 is the missing data marker; handling it
    # at parse time keeps the columns as floats instead of objects.
    wavelength_columns = {q: _wavelength_columns(q) for q in quantities}
    needed_columns = {
        DATE_COLUMN,
        TIME_COLUMN,
        LATITUDE_COLUMN,
        LONGITUDE_COLUMN,
        *quantities,
        *METADATA_COLUMNS.values(),
        *(col for cols in wavelength_columns.values() for col in cols if col),
    }
    aeronet_df = pd.read_csv(
        path,
        skiprows=6,
        delimiter=",",
        encoding="latin-1",
        usecols=lambda col: col in needed_columns,
        na_values=[MISSING_VALUE],
    )
    for col in quantities:
        if col not in aeronet_df.columns:
            raise ValueError(
                f"Requested quantity '{col}' not found in AERONET file columns"
            )

    # Ensure TZ is UTC
    time = pd.to_datetime(
        aeronet_df[DATE_COLUMN] + " " + aeronet_df[TIME_COLUMN],
        format="%d:%m:%Y %H:%M:%S",
    ).dt.tz_localize("UTC")

    # The date the file was processed is the only other date field, same weird format
    last_date_processed = pd.to_datetime(
        aeronet_df[METADATA_COLUMNS["last_date_processed"]], format="%d:%m:%Y"
    ).dt.strftime("%Y-%m-%d")

    # Metadata fields shared by all wavelengths, as lists of plain Python values (NaN
    # isn't representable in JSON, so missing fields become nulls) indexed by file row.
    shared_metadata: dict[str, list] = {}
    for key, col in METADATA_COLUMNS.items():
        series = (
            last_date_processed if key == "last_date_processed" else aeronet_df[col]
        )
        values = series.to_numpy().tolist()
        if key in INT_METADATA_KEYS:
            shared_metadata[key] = [None if pd.isna(v) else int(v) for v in values]
        else:
            shared_metadata[key] = [None if pd.isna(v) else v for v in values]

    orig_shape = (len(aeronet_df.index),)
    orig_names = ("row",)

    # The AERONET files are in wide format (one column per wavelength), we need one row
    # per observation. Each quantity brings its own triplet variability and exact
    # wavelength, so the frames are built separately and concatenated.
    all_dfs = []
    for quantity in quantities:
        triplet_column, exact_wavelength_column = wavelength_columns[quantity]

        values = aeronet_df[quantity].to_numpy(dtype=float)
        sel = np.flatnonzero(~np.isnan(values))
        if sel.size == 0:
            continue
        values = values[sel]

        if triplet_column is not None and triplet_column in aeronet_df.columns:
            triplet = aeronet_df[triplet_column].to_numpy(dtype=float)[sel]
        else:
            triplet = np.full(sel.size, np.nan)
        if (
            exact_wavelength_column is not None
            and exact_wavelength_column in aeronet_df.columns
        ):
            exact_wavelength = aeronet_df[exact_wavelength_column].to_numpy(
                dtype=float
            )[sel]
        else:
            exact_wavelength = np.full(sel.size, np.nan)

        # A missing triplet leaves just the nominal uncertainty, never a NaN error
        uncertainty = np.sqrt(
            AOD_UNCERTAINTY_FLOOR**2 + np.nan_to_num(triplet, nan=0.0) ** 2
        )

        metadata = [
            {
                **{key: field[row] for key, field in shared_metadata.items()},
                "triplet_variability": _nullable(triplet[i]),
                "exact_wavelength_um": _nullable(exact_wavelength[i]),
            }
            for i, row in enumerate(sel.tolist())
        ]
        orig_coords = [
            {
                "indices": (row,),
                "shape": orig_shape,
                "names": orig_names,
            }
            for row in sel.tolist()
        ]

        df = pd.DataFrame(
            {
                "instrument": "AERONET",
                "quantity": quantity,
                "time": time.iloc[sel].reset_index(drop=True),
                "latitude": aeronet_df[LATITUDE_COLUMN].to_numpy()[sel],
                "longitude": aeronet_df[LONGITUDE_COLUMN].to_numpy()[sel],
                "z": 0.0,
                "z_type": "surface",
                "value": values,
                "value_uncertainty": uncertainty,
                "qc_flag": 0,  # No QC available, so set to "good"
                "orig_filename": path.name,
                "metadata": metadata,
            }
        )
        df["orig_coords"] = orig_coords

        all_dfs.append(df)

    if not all_dfs:
        return None  # Nothing to do

    aeronet_df = pd.concat(all_dfs, ignore_index=True)

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
