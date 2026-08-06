"""Tests for the AERONET SDA (all points) observation converter."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wrf_ensembly.observations import io as obs_io
from wrf_ensembly.observations.converters.aeronet_sda import (
    AOD_UNCERTAINTY_FLOOR,
    convert_aeronet_sda,
)

MISSING = "-999.000000"

# A subset of the real 79-column header: everything the converter touches, plus a few
# decoys (the input AOD columns, the repeated `AOD_Empty` fields) so the column matching
# is actually exercised. Note the underscores in the date/time names, which the direct
# sun AOD product doesn't have.
COLUMNS = [
    "Date_(dd:mm:yyyy)",
    "Time_(hh:mm:ss)",
    "Day_of_Year",
    "Total_AOD_500nm[tau_a]",
    "Fine_Mode_AOD_500nm[tau_f]",
    "Coarse_Mode_AOD_500nm[tau_c]",
    "FineModeFraction_500nm[eta]",
    "2nd_Order_Reg_Fit_Error-Total_AOD_500nm[regression_dtau_a]",
    "RMSE_Fine_Mode_AOD_500nm[Dtau_f]",
    "RMSE_Coarse_Mode_AOD_500nm[Dtau_c]",
    "RMSE_FineModeFraction_500nm[Deta]",
    "Angstrom_Exponent(AE)-Total_500nm[alpha]",
    "AE-Fine_Mode_500nm[alpha_f]",
    "Solar_Zenith_Angle(Degrees)",
    "Air_Mass",
    "870nm_Input_AOD",
    "500nm_Input_AOD",
    "380nm_Input_AOD",
    "AOD_Empty",
    "AOD_Empty",
    "Last_Processing_Date",
    "Data_Quality_Level",
    "AERONET_Instrument_Number",
    "AERONET_Site_Name",
    "Site_Latitude(Degrees)",
    "Site_Longitude(Degrees)",
    "Site_Elevation(m)",
    "Measurement_Type(solar or lunar)",
    "Number_of_Wavelengths",
    "Exact_Wavelengths_of_AOD(um)_870nm",
    "Exact_Wavelengths_of_AOD(um)_500nm",
    "Exact_Wavelengths_of_AOD(um)_Empty",
]

SITE_FIELDS = {
    "Last_Processing_Date": "22:08:2020",
    "Data_Quality_Level": "lev20",
    "AERONET_Instrument_Number": "440",
    "AERONET_Site_Name": "ATHENS-NOA",
    "Site_Latitude(Degrees)": "37.972100",
    "Site_Longitude(Degrees)": "23.718000",
    "Site_Elevation(m)": "130.000000",
    "Measurement_Type(solar or lunar)": "solar",
    "Exact_Wavelengths_of_AOD(um)_500nm": "0.500100",
}

# Row 0: all three modes present.
# Row 1: only the total AOD, the deconvolution didn't produce fine/coarse.
# Row 2: nothing at all, must be dropped.
# Row 3: all three present but the total's fit error and the Angstrom exponents missing.
ROWS = [
    {
        **SITE_FIELDS,
        "Date_(dd:mm:yyyy)": "12:05:2008",
        "Time_(hh:mm:ss)": "04:52:40",
        "Day_of_Year": "133",
        "Total_AOD_500nm[tau_a]": "0.224844",
        "Fine_Mode_AOD_500nm[tau_f]": "0.096831",
        "Coarse_Mode_AOD_500nm[tau_c]": "0.128013",
        "FineModeFraction_500nm[eta]": "0.430658",
        "2nd_Order_Reg_Fit_Error-Total_AOD_500nm[regression_dtau_a]": "0.003389",
        "RMSE_Fine_Mode_AOD_500nm[Dtau_f]": "0.019239",
        "RMSE_Coarse_Mode_AOD_500nm[Dtau_c]": "0.018805",
        "RMSE_FineModeFraction_500nm[Deta]": "0.083702",
        "Angstrom_Exponent(AE)-Total_500nm[alpha]": "0.809244",
        "AE-Fine_Mode_500nm[alpha_f]": "2.077390",
        "Solar_Zenith_Angle(Degrees)": "72.849277",
        "Air_Mass": "3.357629",
        "Number_of_Wavelengths": "5",
    },
    {
        **SITE_FIELDS,
        "Date_(dd:mm:yyyy)": "12:05:2008",
        "Time_(hh:mm:ss)": "04:54:45",
        "Day_of_Year": "133",
        "Total_AOD_500nm[tau_a]": "0.219807",
        "2nd_Order_Reg_Fit_Error-Total_AOD_500nm[regression_dtau_a]": "0.003301",
        "Angstrom_Exponent(AE)-Total_500nm[alpha]": "0.871263",
        "Solar_Zenith_Angle(Degrees)": "72.445817",
        "Air_Mass": "3.284410",
        "Number_of_Wavelengths": "5",
    },
    {
        **SITE_FIELDS,
        "Date_(dd:mm:yyyy)": "12:05:2008",
        "Time_(hh:mm:ss)": "05:59:21",
        "Day_of_Year": "133",
        "Solar_Zenith_Angle(Degrees)": "59.795587",
        "Air_Mass": "1.982138",
        "Number_of_Wavelengths": "3",
    },
    {
        **SITE_FIELDS,
        "Date_(dd:mm:yyyy)": "13:05:2008",
        "Time_(hh:mm:ss)": "10:01:00",
        "Day_of_Year": "134",
        "Total_AOD_500nm[tau_a]": "0.181400",
        "Fine_Mode_AOD_500nm[tau_f]": "0.110492",
        "Coarse_Mode_AOD_500nm[tau_c]": "0.070908",
        "FineModeFraction_500nm[eta]": "0.609109",
        "RMSE_Fine_Mode_AOD_500nm[Dtau_f]": "0.020802",
        "RMSE_Coarse_Mode_AOD_500nm[Dtau_c]": "0.019295",
        "RMSE_FineModeFraction_500nm[Deta]": "0.108495",
        "Solar_Zenith_Angle(Degrees)": "22.500000",
        "Air_Mass": "1.080000",
        "Number_of_Wavelengths": "4",
    },
]


def write_sda_file(path: Path, rows: list[dict[str, str]]) -> Path:
    """Write a file shaped like a real AERONET SDA all-points download."""

    lines = [
        "AERONET Version 3; SDA Version 4.1 ",
        "ATHENS-NOA",
        "Version 3: SDA Retrieval Level 2.0",
        "The following data are automatically cloud cleared and quality assured.",
        "Contact: PI=Somebody; PI Email=somebody@example.com",
        "All Points,UNITS can be found at,,, https://aeronet.gsfc.nasa.gov/",
        ",".join(COLUMNS),
    ]
    lines += [",".join(row.get(col, MISSING) for col in COLUMNS) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return path


@pytest.fixture
def sda_file(tmp_path: Path) -> Path:
    return write_sda_file(tmp_path / "19930101_20260801_ATHENS-NOA.ONEILL_lev20", ROWS)


def test_schema_and_dropped_rows(sda_file: Path):
    df = convert_aeronet_sda(sda_file)
    assert df is not None

    # Total AOD from rows 0, 1, 3; fine and coarse from rows 0 and 3 only
    assert len(df) == 7
    assert df["quantity"].value_counts().to_dict() == {
        "AOD_500nm": 3,
        "AOD_Fine_500nm": 2,
        "AOD_Coarse_500nm": 2,
    }
    assert list(df.columns) == obs_io.REQUIRED_COLUMNS
    obs_io.validate_schema(df)

    assert (df["instrument"] == "AERONET_SDA").all()
    assert (df["qc_flag"] == 0).all()
    assert (df["z"] == 0.0).all()
    assert (df["z_type"] == "surface").all()
    assert (df["orig_filename"] == sda_file.name).all()
    assert df["time"].dt.tz is not None
    assert df["value"].dtype == np.float64
    assert df["value_uncertainty"].dtype == np.float64


def test_values_and_uncertainty(sda_file: Path):
    df = convert_aeronet_sda(sda_file)
    assert df is not None
    df = df.set_index(["quantity", df["time"]])

    first = pd.Timestamp("2008-05-12 04:52:40", tz="UTC")
    total = df.loc[("AOD_500nm", first)]
    assert total["value"] == pytest.approx(0.224844)
    assert total["value_uncertainty"] == pytest.approx(
        math.sqrt(AOD_UNCERTAINTY_FLOOR**2 + 0.003389**2)
    )

    fine = df.loc[("AOD_Fine_500nm", first)]
    assert fine["value"] == pytest.approx(0.096831)
    assert fine["value_uncertainty"] == pytest.approx(
        math.sqrt(AOD_UNCERTAINTY_FLOOR**2 + 0.019239**2)
    )

    coarse = df.loc[("AOD_Coarse_500nm", first)]
    assert coarse["value"] == pytest.approx(0.128013)
    assert coarse["value_uncertainty"] == pytest.approx(
        math.sqrt(AOD_UNCERTAINTY_FLOOR**2 + 0.018805**2)
    )

    # SDA splits the total into the two modes
    assert fine["value"] + coarse["value"] == pytest.approx(total["value"])

    # No reported error means only the nominal uncertainty is left
    no_error = df.loc[("AOD_500nm", pd.Timestamp("2008-05-13 10:01:00", tz="UTC"))]
    assert no_error["value_uncertainty"] == pytest.approx(AOD_UNCERTAINTY_FLOOR)


def test_metadata(sda_file: Path):
    df = convert_aeronet_sda(sda_file)
    assert df is not None

    first = pd.Timestamp("2008-05-12 04:52:40", tz="UTC")
    row = df[(df["quantity"] == "AOD_500nm") & (df["time"] == first)].iloc[0]
    assert row["metadata"] == {
        "site_name": "ATHENS-NOA",
        "instrument_number": 440,
        "data_quality_level": "lev20",
        "last_date_processed": "2020-08-22",
        "solar_zenith_angle": 72.849277,
        "air_mass": 3.357629,
        "site_elevation_m": 130.0,
        "measurement_type": "solar",
        "angstrom_total_500": 0.809244,
        "angstrom_fine_500": 2.07739,
        "fine_mode_fraction": 0.430658,
        "fine_mode_fraction_rmse": 0.083702,
        "n_wavelengths": 5,
        "exact_wavelength_um": 0.5001,
        "reported_error": 0.003389,
    }

    # The fine mode observation of the same file row carries its own reported error
    fine = df[(df["quantity"] == "AOD_Fine_500nm") & (df["time"] == first)].iloc[0]
    assert fine["metadata"]["reported_error"] == pytest.approx(0.019239)
    assert fine["metadata"]["site_name"] == "ATHENS-NOA"


def test_missing_metadata_becomes_null(sda_file: Path):
    """-999 markers must end up as None (JSON null), not NaN."""

    df = convert_aeronet_sda(sda_file)
    assert df is not None

    row = df[
        (df["quantity"] == "AOD_500nm")
        & (df["time"] == pd.Timestamp("2008-05-13 10:01:00", tz="UTC"))
    ].iloc[0]
    assert row["metadata"]["reported_error"] is None
    assert row["metadata"]["angstrom_total_500"] is None
    assert row["metadata"]["angstrom_fine_500"] is None
    assert row["metadata"]["fine_mode_fraction"] == pytest.approx(0.609109)


def test_orig_coords_point_at_source_rows(sda_file: Path):
    df = convert_aeronet_sda(sda_file)
    assert df is not None

    for coords in df["orig_coords"]:
        assert tuple(coords["names"]) == ("row",)
        assert tuple(coords["shape"]) == (len(ROWS),)

    total = df[df["quantity"] == "AOD_500nm"]
    assert [c["indices"][0] for c in total["orig_coords"]] == [0, 1, 3]
    fine = df[df["quantity"] == "AOD_Fine_500nm"]
    assert [c["indices"][0] for c in fine["orig_coords"]] == [0, 3]


def test_quantity_selection(sda_file: Path):
    """Only the requested quantities are extracted."""

    df = convert_aeronet_sda(sda_file, ["AOD_Fine_500nm"])
    assert df is not None
    assert set(df["quantity"]) == {"AOD_Fine_500nm"}
    assert len(df) == 2


def test_file_without_retrievals(tmp_path: Path):
    """A file where every row is missing yields nothing."""

    path = write_sda_file(tmp_path / "empty.ONEILL_lev20", [ROWS[2]])
    assert convert_aeronet_sda(path) is None


def test_unknown_quantity_raises(sda_file: Path):
    with pytest.raises(ValueError, match="AOD_550nm"):
        convert_aeronet_sda(sda_file, ["AOD_550nm"])


def test_roundtrip_through_parquet(sda_file: Path, tmp_path: Path):
    df = convert_aeronet_sda(sda_file)
    assert df is not None

    out = tmp_path / "obs.parquet"
    obs_io.write_obs(df, out)
    read_back = obs_io.read_obs(out)

    assert len(read_back) == len(df)
    metadata = read_back[read_back["quantity"] == "AOD_Fine_500nm"].iloc[0]["metadata"]
    assert metadata["site_name"] == "ATHENS-NOA"
    assert metadata["instrument_number"] == 440
