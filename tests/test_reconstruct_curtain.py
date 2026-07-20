import numpy as np
import pandas as pd
import pytest

from wrf_ensembly.observations.utils import reconstruct_curtain


def _curtain_df(names, shape, altitudes_along_axis):
    """
    Build a flat observation frame for a synthetic curtain.

    `altitudes_along_axis` says which axis altitude varies along, so the tests can
    present the same curtain in either dimension order.
    """
    rows = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            idx = (i, j)
            z = float(idx[altitudes_along_axis] * 500)
            rows.append(
                {
                    "value": float(i + j),
                    "model_forecast": float(i + j) * 0.5,
                    "latitude": 10.0 + i * 0.1,
                    "longitude": -20.0 + i * 0.1,
                    "z": z,
                    "qc_flag": 0,
                    "time": pd.Timestamp("2024-08-12 14:45:00"),
                    "orig_coords": {
                        "indices": list(idx),
                        "shape": list(shape),
                        "names": list(names),
                    },
                }
            )
    return pd.DataFrame(rows)


def test_detects_vertical_dimension_second():
    """Altitude varying along axis 1 makes axis 1 the vertical one."""
    df = _curtain_df(
        ["along_track", "height"], (5, 4), altitudes_along_axis=1
    )

    fields, altitude, along_dim, vertical_dim = reconstruct_curtain(
        df, value_columns=["value", "model_forecast"]
    )

    assert along_dim == "along_track"
    assert vertical_dim == "height"
    assert fields["value"].shape == (5, 4)
    assert list(altitude) == [0.0, 500.0, 1000.0, 1500.0]


def test_detects_vertical_dimension_first():
    """The dimension order is instrument-defined, so the reverse must work too."""
    df = _curtain_df(
        ["height", "along_track"], (4, 5), altitudes_along_axis=0
    )

    fields, altitude, along_dim, vertical_dim = reconstruct_curtain(
        df, value_columns=["value"]
    )

    assert along_dim == "along_track"
    assert vertical_dim == "height"
    # Output is always oriented (along_track, vertical)
    assert fields["value"].shape == (5, 4)
    assert list(altitude) == [0.0, 500.0, 1000.0, 1500.0]


def test_altitude_is_sorted_increasing():
    """Rows are ordered by altitude so pcolormesh draws the curtain upright."""
    df = _curtain_df(["along_track", "height"], (3, 4), altitudes_along_axis=1)
    # Invert the altitudes so the native order is descending
    df["z"] = 1500.0 - df["z"]

    _, altitude, _, _ = reconstruct_curtain(df, value_columns=["value"])

    assert list(altitude) == sorted(altitude)


def test_three_dimensional_input_is_rejected():
    """Swath instruments are not curtain-shaped and must fail clearly."""
    df = pd.DataFrame(
        [
            {
                "value": 1.0,
                "latitude": 10.0,
                "longitude": -20.0,
                "z": 100.0,
                "qc_flag": 0,
                "time": pd.Timestamp("2024-08-12 14:45:00"),
                "orig_coords": {
                    "indices": [0, 0, 0],
                    "shape": [2, 2, 2],
                    "names": ["x", "y", "height"],
                },
            }
        ]
    )

    with pytest.raises(ValueError, match="exactly two native dimensions"):
        reconstruct_curtain(df, value_columns=["value"])


def test_one_dimensional_input_is_rejected():
    """Point instruments such as sun photometers are not curtain-shaped."""
    df = pd.DataFrame(
        [
            {
                "value": 1.0,
                "latitude": 10.0,
                "longitude": -20.0,
                "z": 0.0,
                "qc_flag": 0,
                "time": pd.Timestamp("2024-08-12 14:45:00"),
                "orig_coords": {
                    "indices": [0],
                    "shape": [4],
                    "names": ["observation"],
                },
            }
        ]
    )

    with pytest.raises(ValueError, match="exactly two native dimensions"):
        reconstruct_curtain(df, value_columns=["value"])


def test_non_finite_altitude_bins_are_dropped():
    """A vertical bin with no observation has no altitude and must not survive."""
    df = _curtain_df(["along_track", "height"], (3, 4), altitudes_along_axis=1)
    # Remove every observation in height bin 2, leaving it entirely empty
    keep = df["orig_coords"].apply(lambda c: c["indices"][1] != 2)
    df = df[keep]

    fields, altitude, _, _ = reconstruct_curtain(df, value_columns=["value"])

    assert len(altitude) == 3
    assert fields["value"].shape == (3, 3)
    assert np.isfinite(altitude).all()
