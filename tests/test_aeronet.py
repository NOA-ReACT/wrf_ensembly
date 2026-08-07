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
    "AOD_510nm",
    "AOD_500nm",
    "AOD_490nm",
    "AOD_440nm",
    "AOD_380nm",
    "AOD_Empty",
    "AOD_Empty",
    "Triplet_Variability_1020",
    "Triplet_Variability_510",
    "Triplet_Variability_500",
    "Triplet_Variability_490",
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
    "Exact_Wavelengths_of_AOD(um)_510nm",
    "Exact_Wavelengths_of_AOD(um)_500nm",
    "Exact_Wavelengths_of_AOD(um)_490nm",
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

# Row 0: both requested quantities present, and 490/510 too - the measured 500nm wins.
# Row 1: only 500nm, 380nm is missing.
# Row 2: nothing at all, must be dropped.
# Row 3: 500nm present but its triplet and the 500-870 Angstrom exponent are missing.
# Row 4: no 500nm but both neighbours, so an AOD_500nm gets derived from them.
# Row 5: only one neighbour, not enough to fit an Angstrom exponent.
# Row 6: both neighbours but one of them is zero, which the log-space fit can't take.
ROWS = [
    {
        **SITE_FIELDS,
        "Date(dd:mm:yyyy)": "07:02:2024",
        "Time(hh:mm:ss)": "09:03:08",
        "Day_of_Year": "38",
        "AOD_500nm": "0.412300",
        "AOD_490nm": "0.415000",
        "AOD_510nm": "0.409000",
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
    {
        **SITE_FIELDS,
        "Date(dd:mm:yyyy)": "08:02:2024",
        "Time(hh:mm:ss)": "10:16:30",
        "Day_of_Year": "39",
        "AOD_490nm": "0.412000",
        "AOD_510nm": "0.401000",
        "Triplet_Variability_490": "0.003000",
        "Triplet_Variability_510": "0.005000",
        "440-870_Angstrom_Exponent": "0.305000",
        "Solar_Zenith_Angle(Degrees)": "24.100000",
        "Optical_Air_Mass": "1.090000",
        "Number_of_Wavelengths": "8",
        "Exact_Wavelengths_of_AOD(um)_490nm": "0.490400",
        "Exact_Wavelengths_of_AOD(um)_510nm": "0.510300",
    },
    {
        **SITE_FIELDS,
        "Date(dd:mm:yyyy)": "08:02:2024",
        "Time(hh:mm:ss)": "10:31:12",
        "Day_of_Year": "39",
        "AOD_490nm": "0.398000",
        "Solar_Zenith_Angle(Degrees)": "23.400000",
        "Optical_Air_Mass": "1.070000",
        "Number_of_Wavelengths": "6",
    },
    {
        **SITE_FIELDS,
        "Date(dd:mm:yyyy)": "08:02:2024",
        "Time(hh:mm:ss)": "10:46:55",
        "Day_of_Year": "39",
        "AOD_490nm": "0.301000",
        "AOD_510nm": "0.000000",
        "Solar_Zenith_Angle(Degrees)": "22.800000",
        "Optical_Air_Mass": "1.060000",
        "Number_of_Wavelengths": "6",
    },
]

# The wavelengths and AODs of row 4, which the derivation tests work off
DERIVED_TIME = pd.Timestamp("2024-02-08 10:16:30", tz="UTC")
L_490, L_510, L_500 = 0.4904, 0.5103, 0.500
T_490, T_510 = 0.412, 0.401
S_490 = math.sqrt(AOD_UNCERTAINTY_FLOOR**2 + 0.003**2)
S_510 = math.sqrt(AOD_UNCERTAINTY_FLOOR**2 + 0.005**2)


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

    # 500nm measured on rows 0, 1, 3 and derived on row 4; 380nm from row 0 only
    assert len(df) == 5
    assert df["quantity"].value_counts().to_dict() == {"AOD_500nm": 4, "AOD_380nm": 1}
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
        "derived": False,
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

    # The derived observation from row 4 is traceable to its source row like any other
    aod_500 = df[df["quantity"] == "AOD_500nm"]
    assert [c["indices"][0] for c in aod_500["orig_coords"]] == [0, 1, 3, 4]
    aod_380 = df[df["quantity"] == "AOD_380nm"]
    assert [c["indices"][0] for c in aod_380["orig_coords"]] == [0]


def derived_row(df: pd.DataFrame) -> pd.Series | None:
    """The AOD_500nm observation derived from row 4, if the converter emitted one."""

    match = df[(df["quantity"] == "AOD_500nm") & (df["time"] == DERIVED_TIME)]
    return None if match.empty else match.iloc[0]


def test_derived_500nm_value_and_uncertainty(aeronet_file: Path):
    """The 490/510 pair is interpolated onto 500nm with a fitted Angstrom exponent."""

    df = convert_aeronet(aeronet_file)
    assert df is not None

    row = derived_row(df)
    assert row is not None

    # Same math as the converter, written out longhand
    span = math.log(L_510 / L_490)
    lever_arm = math.log(L_500 / L_490)
    angstrom = -math.log(T_510 / T_490) / span
    weight = lever_arm / span

    assert row["value"] == pytest.approx(T_490 * math.exp(-angstrom * lever_arm))
    assert row["value_uncertainty"] == pytest.approx(
        (1 - weight) * S_490 + weight * S_510
    )

    # 500nm sits between the two channels, so both the value and the error must too
    assert T_510 < row["value"] < T_490
    assert min(S_490, S_510) <= row["value_uncertainty"] <= max(S_490, S_510)


def test_derived_500nm_metadata(aeronet_file: Path):
    df = convert_aeronet(aeronet_file)
    assert df is not None

    row = derived_row(df)
    assert row is not None
    metadata = row["metadata"]

    # The shared site fields are there just like on a measured observation
    assert metadata["site_name"] == "Cabo_Verde"
    assert metadata["angstrom_440_870"] == pytest.approx(0.305)
    # Nothing was measured at 500nm, and the target wavelength is the nominal one since
    # an instrument without the channel doesn't report an exact wavelength for it
    assert metadata["triplet_variability"] is None
    assert metadata["exact_wavelength_um"] == pytest.approx(L_500)

    assert metadata["derived"] is True
    assert metadata["derivation"] == {
        "method": "angstrom",
        "sources": [
            {
                "quantity": "AOD_490nm",
                "value": pytest.approx(T_490),
                "uncertainty": pytest.approx(S_490),
            },
            {
                "quantity": "AOD_510nm",
                "value": pytest.approx(T_510),
                "uncertainty": pytest.approx(S_510),
            },
        ],
        "angstrom": {
            "value": pytest.approx(-math.log(T_510 / T_490) / math.log(L_510 / L_490)),
            "origin": "fitted_from_sources",
        },
        "lever_arm": pytest.approx(math.log(L_500 / L_490)),
    }


def test_measured_500nm_is_not_replaced(aeronet_file: Path):
    """Row 0 has 490 and 510 too, but a measured channel always wins."""

    df = convert_aeronet(aeronet_file)
    assert df is not None

    row = df[
        (df["quantity"] == "AOD_500nm")
        & (df["time"] == pd.Timestamp("2024-02-07 09:03:08", tz="UTC"))
    ]
    assert len(row) == 1
    assert row.iloc[0]["value"] == pytest.approx(0.4123)
    assert row.iloc[0]["metadata"]["derived"] is False
    assert "derivation" not in row.iloc[0]["metadata"]


def test_no_derivation_without_both_channels(aeronet_file: Path):
    """Rows 5 (one neighbour) and 6 (a zero neighbour) yield nothing."""

    df = convert_aeronet(aeronet_file)
    assert df is not None

    for time in ("2024-02-08 10:31:12", "2024-02-08 10:46:55"):
        assert df[df["time"] == pd.Timestamp(time, tz="UTC")].empty


def test_derivation_can_be_disabled(aeronet_file: Path):
    df = convert_aeronet(aeronet_file, derive_500nm=False)
    assert df is not None

    assert derived_row(df) is None
    assert len(df) == 4
    assert not any(meta["derived"] for meta in df["metadata"])


def test_derivation_skipped_when_500nm_not_requested(aeronet_file: Path):
    """Nothing to interpolate onto if the caller never asked for 500nm."""

    df = convert_aeronet(aeronet_file, ["AOD_380nm"])
    assert df is not None

    assert (df["quantity"] == "AOD_380nm").all()


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

    # Only some rows carry the nested derivation block; parquet unions the two shapes,
    # leaving the measured observations with a null there
    assert metadata["derived"] is False
    assert metadata["derivation"] is None

    derivation = derived_row(read_back)["metadata"]["derivation"]  # type: ignore[index]
    assert derivation["method"] == "angstrom"
    assert derivation["angstrom"]["origin"] == "fitted_from_sources"
    assert [s["quantity"] for s in derivation["sources"]] == ["AOD_490nm", "AOD_510nm"]
    assert derivation["lever_arm"] == pytest.approx(math.log(L_500 / L_490))
