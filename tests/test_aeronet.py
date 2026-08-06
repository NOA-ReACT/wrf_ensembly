"""Tests for the AERONET (AOD, all points) observation converter."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wrf_ensembly.observations import io as obs_io
from wrf_ensembly.observations.converters.aeronet import (
    AOD_UNCERTAINTY_FLOOR,
    convert_aeronet,
)

MISSING = "-999.000000"

# A subset of the real 113-column header: everything the converter touches, plus a few
# decoys (other wavelengths, the repeated `AOD_Empty` columns) so the column matching is
# actually exercised. Note the real files' inconsistent suffixes, which we keep here:
# the triplet columns drop the `nm`, the exact wavelength ones keep it.
COLUMNS = [
    "Date(dd:mm:yyyy)",
    "Time(hh:mm:ss)",
    "Day_of_Year",
    "AOD_1020nm",
    "AOD_500nm",
    "AOD_440nm",
    "AOD_380nm",
    "AOD_Empty",
    "AOD_Empty",
    "Triplet_Variability_1020",
    "Triplet_Variability_500",
    "Triplet_Variability_440",
    "Triplet_Variability_380",
    "Triplet_Variability_AOD_Empty",
    "440-870_Angstrom_Exponent",
    "500-870_Angstrom_Exponent",
    "Data_Quality_Level",
    "AERONET_Instrument_Number",
    "AERONET_Site_Name",
    "Site_Latitude(Degrees)",
    "Site_Longitude(Degrees)",
    "Site_Elevation(m)",
    "Solar_Zenith_Angle(Degrees)",
    "Optical_Air_Mass",
    "Last_Date_Processed",
    "Number_of_Wavelengths",
    "Exact_Wavelengths_of_AOD(um)_1020nm",
    "Exact_Wavelengths_of_AOD(um)_500nm",
    "Exact_Wavelengths_of_AOD(um)_440nm",
    "Exact_Wavelengths_of_AOD(um)_380nm",
]

SITE_FIELDS = {
    "Data_Quality_Level": "lev20",
    "AERONET_Instrument_Number": "1073",
    "AERONET_Site_Name": "Cabo_Verde",
    "Site_Latitude(Degrees)": "16.733000",
    "Site_Longitude(Degrees)": "-22.935000",
    "Site_Elevation(m)": "60.000000",
    "Last_Date_Processed": "14:03:2025",
}

# Row 0: both requested quantities present.
# Row 1: only 500nm, 380nm is missing.
# Row 2: nothing at all, must be dropped.
# Row 3: 500nm present but its triplet and the 500-870 Angstrom exponent are missing.
ROWS = [
    {
        **SITE_FIELDS,
        "Date(dd:mm:yyyy)": "07:02:2024",
        "Time(hh:mm:ss)": "09:03:08",
        "Day_of_Year": "38",
        "AOD_500nm": "0.412300",
        "AOD_380nm": "0.501200",
        "Triplet_Variability_500": "0.004000",
        "Triplet_Variability_380": "0.006000",
        "440-870_Angstrom_Exponent": "0.280000",
        "500-870_Angstrom_Exponent": "0.310000",
        "Solar_Zenith_Angle(Degrees)": "34.200000",
        "Optical_Air_Mass": "1.210000",
        "Number_of_Wavelengths": "9",
        "Exact_Wavelengths_of_AOD(um)_500nm": "0.500800",
        "Exact_Wavelengths_of_AOD(um)_380nm": "0.379800",
    },
    {
        **SITE_FIELDS,
        "Date(dd:mm:yyyy)": "07:02:2024",
        "Time(hh:mm:ss)": "09:18:11",
        "Day_of_Year": "38",
        "AOD_500nm": "0.398700",
        "Triplet_Variability_500": "0.002000",
        "440-870_Angstrom_Exponent": "0.290000",
        "500-870_Angstrom_Exponent": "0.320000",
        "Solar_Zenith_Angle(Degrees)": "31.100000",
        "Optical_Air_Mass": "1.170000",
        "Number_of_Wavelengths": "8",
        "Exact_Wavelengths_of_AOD(um)_500nm": "0.500800",
    },
    {
        **SITE_FIELDS,
        "Date(dd:mm:yyyy)": "07:02:2024",
        "Time(hh:mm:ss)": "09:33:47",
        "Day_of_Year": "38",
        "Solar_Zenith_Angle(Degrees)": "28.900000",
        "Optical_Air_Mass": "1.140000",
        "Number_of_Wavelengths": "3",
    },
    {
        **SITE_FIELDS,
        "Date(dd:mm:yyyy)": "08:02:2024",
        "Time(hh:mm:ss)": "10:01:00",
        "Day_of_Year": "39",
        "AOD_500nm": "0.350000",
        "440-870_Angstrom_Exponent": "0.300000",
        "Solar_Zenith_Angle(Degrees)": "22.500000",
        "Optical_Air_Mass": "1.080000",
        "Number_of_Wavelengths": "7",
        "Exact_Wavelengths_of_AOD(um)_500nm": "0.500800",
    },
]


def write_aeronet_file(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write a file shaped like a real AERONET all-points download."""

    lines = [
        "AERONET Version 3",
        "Cabo_Verde",
        "Version 3: AOD Level 2.0",
        "The following data are automatically cloud cleared and quality assured.",
        "Contact: PI=Somebody; PI Email=somebody@example.com",
        "All Points,UNITS can be found at,,, https://aeronet.gsfc.nasa.gov/",
        ",".join(COLUMNS),
    ]
    lines += [",".join(row.get(col, MISSING) for col in COLUMNS) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


@pytest.fixture
def aeronet_file(tmp_path: Path) -> Path:
    return write_aeronet_file(tmp_path / "19930101_20250906_Cabo_Verde.lev20", ROWS)


def test_schema_and_dropped_rows(aeronet_file: Path):
    df = convert_aeronet(aeronet_file)
    assert df is not None

    # Three observations: 500nm from rows 0, 1, 3 and 380nm from row 0 only
    assert len(df) == 4
    assert df["quantity"].value_counts().to_dict() == {"AOD_500nm": 3, "AOD_380nm": 1}
    assert list(df.columns) == obs_io.REQUIRED_COLUMNS
    obs_io.validate_schema(df)

    assert (df["instrument"] == "AERONET").all()
    assert (df["qc_flag"] == 0).all()
    assert (df["z"] == 0.0).all()
    assert (df["z_type"] == "surface").all()
    assert (df["orig_filename"] == aeronet_file.name).all()
    assert df["time"].dt.tz is not None
    assert df["value"].dtype == np.float64
    assert df["value_uncertainty"].dtype == np.float64


def test_values_and_uncertainty(aeronet_file: Path):
    df = convert_aeronet(aeronet_file)
    assert df is not None
    df = df.set_index(["quantity", df["time"]])

    first = df.loc[("AOD_500nm", pd.Timestamp("2024-02-07 09:03:08", tz="UTC"))]
    assert first["value"] == pytest.approx(0.4123)
    assert first["value_uncertainty"] == pytest.approx(
        math.sqrt(AOD_UNCERTAINTY_FLOOR**2 + 0.004**2)
    )

    uv = df.loc[("AOD_380nm", pd.Timestamp("2024-02-07 09:03:08", tz="UTC"))]
    assert uv["value"] == pytest.approx(0.5012)
    assert uv["value_uncertainty"] == pytest.approx(
        math.sqrt(AOD_UNCERTAINTY_FLOOR**2 + 0.006**2)
    )

    # No triplet variability means only the nominal uncertainty is left
    no_triplet = df.loc[("AOD_500nm", pd.Timestamp("2024-02-08 10:01:00", tz="UTC"))]
    assert no_triplet["value_uncertainty"] == pytest.approx(AOD_UNCERTAINTY_FLOOR)


def test_metadata(aeronet_file: Path):
    df = convert_aeronet(aeronet_file)
    assert df is not None

    row = df[
        (df["quantity"] == "AOD_500nm")
        & (df["time"] == pd.Timestamp("2024-02-07 09:03:08", tz="UTC"))
    ].iloc[0]
    assert row["metadata"] == {
        "site_name": "Cabo_Verde",
        "instrument_number": 1073,
        "data_quality_level": "lev20",
        "last_date_processed": "2025-03-14",
        "solar_zenith_angle": 34.2,
        "optical_air_mass": 1.21,
        "triplet_variability": 0.004,
        "angstrom_440_870": 0.28,
        "angstrom_500_870": 0.31,
        "exact_wavelength_um": 0.5008,
        "n_wavelengths": 9,
    }

    # The 380nm observation of the same file row gets its own wavelength fields
    uv = df[df["quantity"] == "AOD_380nm"].iloc[0]
    assert uv["metadata"]["triplet_variability"] == pytest.approx(0.006)
    assert uv["metadata"]["exact_wavelength_um"] == pytest.approx(0.3798)
    assert uv["metadata"]["site_name"] == "Cabo_Verde"


def test_missing_metadata_becomes_null(aeronet_file: Path):
    """-999 markers must end up as None (JSON null), not NaN."""

    df = convert_aeronet(aeronet_file)
    assert df is not None

    row = df[
        (df["quantity"] == "AOD_500nm")
        & (df["time"] == pd.Timestamp("2024-02-08 10:01:00", tz="UTC"))
    ].iloc[0]
    assert row["metadata"]["angstrom_500_870"] is None
    assert row["metadata"]["triplet_variability"] is None
    assert row["metadata"]["angstrom_440_870"] == pytest.approx(0.30)


def test_orig_coords_point_at_source_rows(aeronet_file: Path):
    df = convert_aeronet(aeronet_file)
    assert df is not None

    for coords in df["orig_coords"]:
        assert tuple(coords["names"]) == ("row",)
        assert tuple(coords["shape"]) == (len(ROWS),)

    aod_500 = df[df["quantity"] == "AOD_500nm"]
    assert [c["indices"][0] for c in aod_500["orig_coords"]] == [0, 1, 3]
    aod_380 = df[df["quantity"] == "AOD_380nm"]
    assert [c["indices"][0] for c in aod_380["orig_coords"]] == [0]


def test_file_without_requested_quantities(tmp_path: Path):
    """A station that never measured the requested wavelengths yields nothing."""

    path = write_aeronet_file(tmp_path / "empty.lev20", [ROWS[2]])
    assert convert_aeronet(path) is None


def test_unknown_quantity_raises(aeronet_file: Path):
    with pytest.raises(ValueError, match="AOD_1234nm"):
        convert_aeronet(aeronet_file, ["AOD_1234nm"])


def test_roundtrip_through_parquet(aeronet_file: Path, tmp_path: Path):
    df = convert_aeronet(aeronet_file)
    assert df is not None

    out = tmp_path / "obs.parquet"
    obs_io.write_obs(df, out)
    read_back = obs_io.read_obs(out)

    assert len(read_back) == len(df)
    metadata = read_back[read_back["quantity"] == "AOD_500nm"].iloc[0]["metadata"]
    assert metadata["site_name"] == "Cabo_Verde"
    assert metadata["instrument_number"] == 1073
    assert metadata["exact_wavelength_um"] == pytest.approx(0.5008)
    assert metadata["angstrom_500_870"] == pytest.approx(0.31)
