"""Converter for AERONET SDA CSV-ish files to WRF-Ensembly Observation format.

The SDA (Spectral Deconvolution Algorithm) product splits the 500nm total AOD into its
fine and coarse mode contributions. Files are shaped like the direct-sun AOD downloads
(six preamble lines, then a header row, then CSV) but share none of their column names,
hence the separate converter.
"""

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
with the error the SDA retrieval reports for the matching quantity, which on its own
goes down to 1e-6 and would weight these observations absurdly heavily.
"""

DATE_COLUMN = "Date_(dd:mm:yyyy)"
TIME_COLUMN = "Time_(hh:mm:ss)"
LATITUDE_COLUMN = "Site_Latitude(Degrees)"
LONGITUDE_COLUMN = "Site_Longitude(Degrees)"

SDA_QUANTITIES = {
    # quantity -> (value column, reported error column)
    "AOD_500nm": (
        "Total_AOD_500nm[tau_a]",
        "2nd_Order_Reg_Fit_Error-Total_AOD_500nm[regression_dtau_a]",
    ),
    "AOD_Fine_500nm": (
        "Fine_Mode_AOD_500nm[tau_f]",
        "RMSE_Fine_Mode_AOD_500nm[Dtau_f]",
    ),
    "AOD_Coarse_500nm": (
        "Coarse_Mode_AOD_500nm[tau_c]",
        "RMSE_Coarse_Mode_AOD_500nm[Dtau_c]",
    ),
}

DEFAULT_QUANTITIES = list(SDA_QUANTITIES)

METADATA_COLUMNS = {
    # metadata key -> SDA column, for fields shared by all three quantities
    "site_name": "AERONET_Site_Name",
    "instrument_number": "AERONET_Instrument_Number",
    "data_quality_level": "Data_Quality_Level",
    "last_date_processed": "Last_Processing_Date",
    "solar_zenith_angle": "Solar_Zenith_Angle(Degrees)",
    "air_mass": "Air_Mass",
    "site_elevation_m": "Site_Elevation(m)",
    "measurement_type": "Measurement_Type(solar or lunar)",
    "angstrom_total_500": "Angstrom_Exponent(AE)-Total_500nm[alpha]",
    "angstrom_fine_500": "AE-Fine_Mode_500nm[alpha_f]",
    "fine_mode_fraction": "FineModeFraction_500nm[eta]",
    "fine_mode_fraction_rmse": "RMSE_FineModeFraction_500nm[Deta]",
    "n_wavelengths": "Number_of_Wavelengths",
    "exact_wavelength_um": "Exact_Wavelengths_of_AOD(um)_500nm",
}

INT_METADATA_KEYS = ("instrument_number", "n_wavelengths")


def _nullable(value: np.floating) -> float | None:
    """NaN isn't representable in JSON, so a missing field becomes a null instead."""

    return None if np.isnan(value) else float(value)


def convert_aeronet_sda(
    path: Path, quantities: list[str] = DEFAULT_QUANTITIES
) -> None | pd.DataFrame:
    """Convert an AERONET SDA file to WRF-Ensembly Observation format.

    Args:
        path: Path to the AERONET SDA tabular file.
        quantities: List of quantities to extract from the file, see `SDA_QUANTITIES`.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format (correct columns etc).
    """

    unknown = [q for q in quantities if q not in SDA_QUANTITIES]
    if unknown:
        raise ValueError(
            f"Unknown SDA quantities: {', '.join(unknown)}. "
            f"Available: {', '.join(SDA_QUANTITIES)}"
        )

    # SDA files carry 79 columns and the big stations reach half a million rows, so only
    # parse what we actually need. -999 is the missing data marker; handling it at parse
    # time keeps the columns as floats instead of objects.
    needed_columns = {
        DATE_COLUMN,
        TIME_COLUMN,
        LATITUDE_COLUMN,
        LONGITUDE_COLUMN,
        *METADATA_COLUMNS.values(),
        *(col for q in quantities for col in SDA_QUANTITIES[q]),
    }
    sda_df = pd.read_csv(
        path,
        skiprows=6,
        delimiter=",",
        encoding="latin-1",
        usecols=lambda col: col in needed_columns,
        na_values=[MISSING_VALUE],
    )
    for quantity in quantities:
        value_column = SDA_QUANTITIES[quantity][0]
        if value_column not in sda_df.columns:
            raise ValueError(
                f"Column '{value_column}' for quantity '{quantity}' not found in "
                "AERONET SDA file"
            )

    # Ensure TZ is UTC
    time = pd.to_datetime(
        sda_df[DATE_COLUMN] + " " + sda_df[TIME_COLUMN],
        format="%d:%m:%Y %H:%M:%S",
    ).dt.tz_localize("UTC")

    # The date the file was processed is the only other date field, same weird format
    last_date_processed = pd.to_datetime(
        sda_df[METADATA_COLUMNS["last_date_processed"]], format="%d:%m:%Y"
    ).dt.strftime("%Y-%m-%d")

    # Metadata fields shared by all quantities, as lists of plain Python values (NaN
    # isn't representable in JSON, so missing fields become nulls) indexed by file row.
    shared_metadata: dict[str, list] = {}
    for key, col in METADATA_COLUMNS.items():
        series = last_date_processed if key == "last_date_processed" else sda_df[col]
        values = series.to_numpy().tolist()
        if key in INT_METADATA_KEYS:
            shared_metadata[key] = [None if pd.isna(v) else int(v) for v in values]
        else:
            shared_metadata[key] = [None if pd.isna(v) else v for v in values]

    orig_shape = (len(sda_df.index),)
    orig_names = ("row",)

    # The SDA files are in wide format (one column per mode), we need one row per
    # observation. Each quantity brings its own reported error, so the frames are built
    # separately and concatenated.
    all_dfs = []
    for quantity in quantities:
        value_column, error_column = SDA_QUANTITIES[quantity]

        values = sda_df[value_column].to_numpy(dtype=float)
        sel = np.flatnonzero(~np.isnan(values))
        if sel.size == 0:
            continue
        values = values[sel]

        if error_column in sda_df.columns:
            reported_error = sda_df[error_column].to_numpy(dtype=float)[sel]
        else:
            reported_error = np.full(sel.size, np.nan)

        # A missing error leaves just the nominal uncertainty, never a NaN error
        uncertainty = np.sqrt(
            AOD_UNCERTAINTY_FLOOR**2 + np.nan_to_num(reported_error, nan=0.0) ** 2
        )

        metadata = [
            {
                **{key: field[row] for key, field in shared_metadata.items()},
                "reported_error": _nullable(reported_error[i]),
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
                "instrument": "AERONET_SDA",
                "quantity": quantity,
                "time": time.iloc[sel].reset_index(drop=True),
                "latitude": sda_df[LATITUDE_COLUMN].to_numpy()[sel],
                "longitude": sda_df[LONGITUDE_COLUMN].to_numpy()[sel],
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

    sda_df = pd.concat(all_dfs, ignore_index=True)

    # Sort columns as defined in the schema, do the sanity check
    sda_df = sda_df[obs_io.REQUIRED_COLUMNS]
    obs_io.validate_schema(sda_df)

    return sda_df


@click.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option(
    "--quantities",
    multiple=True,
    default=DEFAULT_QUANTITIES,
    help="Quantities to extract from the AERONET SDA file. Can be specified multiple times.",
)
def aeronet_sda(input_path: Path, output_path: Path, quantities: List[str]):
    """Convert AERONET SDA CSV file to WRF-Ensembly observation format.

    INPUT_PATH: Path to the AERONET SDA CSV file (`.ONEILL_lev20`)
    OUTPUT_PATH: Path where to save the converted observations (will be saved as parquet)
    """

    print(f"Converting AERONET SDA file: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Quantities: {', '.join(quantities)}")

    # Convert the data
    converted_df = convert_aeronet_sda(input_path, list(quantities))
    if converted_df is None or converted_df.empty:
        print("No observations found in the input file, aborting")
        return

    # Save to output path as parquet
    obs_io.write_obs(converted_df, output_path)

    print(f"Successfully converted {len(converted_df)} observations")
    print(f"Saved to: {output_path}")
