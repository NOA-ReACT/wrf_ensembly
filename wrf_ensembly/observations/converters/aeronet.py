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

DERIVED_QUANTITY = "AOD_500nm"
DERIVED_SOURCES = ("AOD_490nm", "AOD_510nm")
"""
Plenty of sites carry 490nm and 510nm channels instead of a 500nm one. Since 500nm is the
wavelength wired into DART, those rows would otherwise be lost, so we interpolate onto it
with an Angstrom exponent fitted to the two neighbours. The 20nm lever means this is
interpolation throughout, never extrapolation.
"""


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


def _nominal_wavelength_um(quantity: str) -> float | None:
    """
    The wavelength an AOD quantity is named after, in um, or None if the quantity isn't a
    per-wavelength AOD. Used as the fallback when a file doesn't report the exact one.
    """

    match = AOD_QUANTITY_RE.match(quantity)
    if match is None:
        return None

    return float(match.group("wavelength")) / 1000.0


def _nullable(value: np.floating) -> float | None:
    """NaN isn't representable in JSON, so a missing field becomes a null instead."""

    return None if np.isnan(value) else float(value)


def _angstrom_interpolate(
    value_low: np.ndarray,
    value_high: np.ndarray,
    lambda_low: np.ndarray,
    lambda_high: np.ndarray,
    lambda_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate AOD onto `lambda_target` using an Angstrom exponent fitted to the two
    bracketing channels. Elementwise over all five arrays; only wavelength ratios enter,
    so any consistent unit works.

    Returns `(value, angstrom, lever_arm, weight)`, where the lever arm is the
    log-wavelength distance `ln(target / low)` the fit is carried over and the weight is
    that distance as a fraction of the channel separation, i.e. the target's position
    between the two channels in log space.
    """

    span = np.log(lambda_high / lambda_low)
    lever_arm = np.log(lambda_target / lambda_low)
    angstrom = -np.log(value_high / value_low) / span

    return (
        value_low * np.exp(-angstrom * lever_arm),
        angstrom,
        lever_arm,
        lever_arm / span,
    )


def convert_aeronet(
    path: Path,
    quantities: list[str] = ["AOD_380nm", "AOD_500nm"],
    derive_500nm: bool = True,
) -> None | pd.DataFrame:
    """Convert an AERONET file to WRF-Ensembly Observation format.

    Args:
        path: Path to the AERONET tabular file.
        quantities: List of quantities to extract from the file.
        derive_500nm: Whether to synthesise `AOD_500nm` from the 490nm and 510nm channels
            for rows that lack a measured 500nm one. Ignored unless `AOD_500nm` was
            requested.

    Returns:
        A pandas DataFrame in WRF-Ensembly Observation format (correct columns etc).
    """

    derive_500nm = derive_500nm and DERIVED_QUANTITY in quantities

    # AERONET files carry 113 columns and the big stations reach half a million rows,
    # so only parse what we actually need. -999 is the missing data marker; handling it
    # at parse time keeps the columns as floats instead of objects.
    columns_of_interest = list(quantities)
    if derive_500nm:
        columns_of_interest += [
            q for q in DERIVED_SOURCES if q not in columns_of_interest
        ]
    wavelength_columns = {q: _wavelength_columns(q) for q in columns_of_interest}
    needed_columns = {
        DATE_COLUMN,
        TIME_COLUMN,
        LATITUDE_COLUMN,
        LONGITUDE_COLUMN,
        *columns_of_interest,
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

    def optional_column(name: str | None) -> np.ndarray:
        """A column as a float array, all-NaN if this file doesn't carry it."""

        if name is None or name not in aeronet_df.columns:
            return np.full(len(aeronet_df.index), np.nan)
        return aeronet_df[name].to_numpy(dtype=float)

    def channel_uncertainty(quantity: str, sel: np.ndarray) -> np.ndarray:
        """
        A channel's per-observation error: the nominal floor and the triplet variability
        added in quadrature. A missing triplet leaves just the floor, never a NaN error.
        """

        triplet = optional_column(wavelength_columns[quantity][0])[sel]
        return np.sqrt(AOD_UNCERTAINTY_FLOOR**2 + np.nan_to_num(triplet, nan=0.0) ** 2)

    def frame(
        quantity: str,
        sel: np.ndarray,
        values: np.ndarray,
        uncertainty: np.ndarray,
        extra_metadata: list[dict],
    ) -> pd.DataFrame:
        """
        One quantity's observations, `sel` indexing the file rows they came from.
        `extra_metadata` holds the per-observation fields to merge over the shared ones.
        """

        metadata = [
            {**{key: field[row] for key, field in shared_metadata.items()}, **extra}
            for row, extra in zip(sel.tolist(), extra_metadata)
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

        return df

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

        triplet = optional_column(triplet_column)[sel]
        exact_wavelength = optional_column(exact_wavelength_column)[sel]

        all_dfs.append(
            frame(
                quantity,
                sel,
                values,
                channel_uncertainty(quantity, sel),
                [
                    {
                        "triplet_variability": _nullable(triplet[i]),
                        "exact_wavelength_um": _nullable(exact_wavelength[i]),
                        "derived": False,
                    }
                    for i in range(sel.size)
                ],
            )
        )

    if derive_500nm:
        low_quantity, high_quantity = DERIVED_SOURCES

        def wavelength(quantity: str) -> np.ndarray:
            """
            Per-row wavelength of a channel in um: the exact one the file reports, or the
            nominal one where it doesn't. Sites without a 500nm channel also leave its
            exact wavelength empty, so the target is normally the nominal 0.5um.
            """

            nominal = _nominal_wavelength_um(quantity)
            assert nominal is not None  # All of DERIVED_SOURCES parse as AOD quantities
            return np.nan_to_num(
                optional_column(wavelength_columns[quantity][1]), nan=nominal
            )

        low = optional_column(low_quantity)
        high = optional_column(high_quantity)
        lambda_low = wavelength(low_quantity)
        lambda_high = wavelength(high_quantity)
        lambda_target = wavelength(DERIVED_QUANTITY)

        # Only rows that are actually missing the measured channel; a NaN AOD compares
        # false, so this also rejects rows where either source is absent. Everything the
        # logs divide by or take has to be positive, and the two channels have to be
        # apart, otherwise the fit blows up into infinities and silent NaN values.
        sel = np.flatnonzero(
            np.isnan(optional_column(DERIVED_QUANTITY))
            & (low > 0.0)
            & (high > 0.0)
            & (lambda_low > 0.0)
            & (lambda_target > 0.0)
            & (lambda_high > lambda_low)
        )
        if sel.size > 0:
            low_value, high_value = low[sel], high[sel]
            target_wavelength = lambda_target[sel]
            values, angstrom, lever_arm, weight = _angstrom_interpolate(
                low_value,
                high_value,
                lambda_low[sel],
                lambda_high[sel],
                target_wavelength,
            )

            # AERONET's channel errors share a calibration, so the two sources add
            # linearly rather than in quadrature. The weights sum to one, which keeps the
            # result between the two channel errors and never below the nominal floor.
            low_uncertainty = channel_uncertainty(low_quantity, sel)
            high_uncertainty = channel_uncertainty(high_quantity, sel)
            uncertainty = (1.0 - weight) * low_uncertainty + weight * high_uncertainty

            all_dfs.append(
                frame(
                    DERIVED_QUANTITY,
                    sel,
                    values,
                    uncertainty,
                    [
                        {
                            "triplet_variability": None,  # Nothing was measured here
                            "exact_wavelength_um": float(target_wavelength[i]),
                            "derived": True,
                            "derivation": {
                                "method": "angstrom",
                                "sources": [
                                    {
                                        "quantity": low_quantity,
                                        "value": float(low_value[i]),
                                        "uncertainty": float(low_uncertainty[i]),
                                    },
                                    {
                                        "quantity": high_quantity,
                                        "value": float(high_value[i]),
                                        "uncertainty": float(high_uncertainty[i]),
                                    },
                                ],
                                "angstrom": {
                                    "value": float(angstrom[i]),
                                    "origin": "fitted_from_sources",
                                },
                                "lever_arm": float(lever_arm[i]),
                            },
                        }
                        for i in range(sel.size)
                    ],
                )
            )

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
@click.option(
    "--derive-500nm/--no-derive-500nm",
    default=True,
    help="Synthesise AOD_500nm from the 490nm and 510nm channels where the site doesn't measure it.",
)
def aeronet(
    input_path: Path, output_path: Path, quantities: List[str], derive_500nm: bool
):
    """Convert AERONET CSV file to WRF-Ensembly observation format.

    INPUT_PATH: Path to the AERONET CSV file
    OUTPUT_PATH: Path where to save the converted observations (will be saved as parquet)
    """

    print(f"Converting AERONET file: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Quantities: {', '.join(quantities)}")

    # Convert the data
    converted_df = convert_aeronet(input_path, list(quantities), derive_500nm)
    if converted_df is None or converted_df.empty:
        print("No observations found in the input file, aborting")
        return

    # Save to output path as parquet
    obs_io.write_obs(converted_df, output_path)

    print(f"Successfully converted {len(converted_df)} observations")
    print(f"Saved to: {output_path}")
