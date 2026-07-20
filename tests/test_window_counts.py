import numpy as np
import pandas as pd
import pytest
import click

from wrf_ensembly.observations.window_counts import _parse_bin_spec, count_by_window
from wrf_ensembly.superobs import grid_bin


def _raw_obs(orig_filename: str, n_along: int = 4, n_height: int = 4):
    """Pre-superob observations on a native (along_track, height) grid."""
    rows = []
    for i in range(n_along):
        for j in range(n_height):
            rows.append(
                {
                    "instrument": "TEST_LIDAR",
                    "quantity": "EXTINCTION",
                    "time": pd.Timestamp("2024-08-12 14:00:00", tz="UTC"),
                    "longitude": -20.0 + i * 0.1,
                    "latitude": 10.0 + i * 0.1,
                    "x": float(i),
                    "y": float(i),
                    "z": float(j * 500),
                    "z_type": "height",
                    "value": 1.0,
                    "value_uncertainty": 0.1,
                    "qc_flag": 0,
                    "orig_coords": {
                        "indices": np.array([i, j], dtype=np.int32),
                        "shape": np.array([n_along, n_height], dtype=np.int32),
                        "names": np.array(
                            ["along_track", "height"], dtype=object
                        ),
                    },
                    "orig_filename": orig_filename,
                    "metadata": None,
                }
            )
    return pd.DataFrame(rows)


def test_grid_bin_does_not_merge_across_source_files():
    """
    Native grid indices restart at zero in every file, so two overpasses cover the
    same indices. Binning them in one call must still keep them apart.
    """
    a = _raw_obs("overpass_a.h5")
    b = _raw_obs("overpass_b.h5")
    both = pd.concat([a, b], ignore_index=True)

    bins_h, bins_v = {"along_track": 2}, {"height": 2}

    together = grid_bin(both, bins_h, bins_v)
    per_file = pd.concat(
        [grid_bin(a, bins_h, bins_v), grid_bin(b, bins_h, bins_v)], ignore_index=True
    )

    # 2x2 superobs per file, so 8 in total either way round
    assert len(per_file) == 8
    assert len(together) == 8


def test_grid_bin_preserves_source_provenance():
    """Every superob is attributed to the file its observations came from."""
    both = pd.concat(
        [_raw_obs("overpass_a.h5"), _raw_obs("overpass_b.h5")], ignore_index=True
    )

    together = grid_bin(both, {"along_track": 2}, {"height": 2})

    assert set(together["orig_filename"]) == {"overpass_a.h5", "overpass_b.h5"}
    # Each file contributed the same number of superobs
    assert together["orig_filename"].value_counts().to_dict() == {
        "overpass_a.h5": 4,
        "overpass_b.h5": 4,
    }


def test_grid_bin_shape_is_per_source_file():
    """A file's recorded grid extent describes that file, not the whole frame."""
    small = _raw_obs("small.h5", n_along=4, n_height=4)
    large = _raw_obs("large.h5", n_along=8, n_height=4)

    together = grid_bin(
        pd.concat([small, large], ignore_index=True),
        {"along_track": 2},
        {"height": 2},
    )

    shapes = {
        row["orig_filename"]: tuple(row["orig_coords"]["shape"])
        for _, row in together.iterrows()
    }
    assert shapes["small.h5"] == (2, 2)
    assert shapes["large.h5"] == (4, 2)


def test_grid_bin_does_not_mutate_input_metadata():
    """The superob record must not be written back into the caller's frame."""
    df = _raw_obs("overpass_a.h5")
    df["metadata"] = [{"existing": "value"} for _ in range(len(df))]

    grid_bin(df, {"along_track": 2}, {"height": 2})

    assert all(m == {"existing": "value"} for m in df["metadata"])


def test_grid_bin_accepts_raw_metadata():
    """
    Converters that populate metadata emit dicts, and dicts survive the parquet
    round-trip, so a raw file with metadata must bin normally. Only the serialised
    form produced by the experiment database is rejected.
    """
    df = _raw_obs("overpass_a.h5")
    df["metadata"] = [{"is_over_land": 1} for _ in range(len(df))]

    result = grid_bin(df, {"along_track": 2}, {"height": 2})

    assert len(result) == 4
    # The original metadata is carried through alongside the superob record
    assert result["metadata"].iloc[0]["is_over_land"] == 1
    assert "superob" in result["metadata"].iloc[0]


def test_grid_bin_accepts_null_metadata():
    """Most converters set metadata to NA; that must bin normally too."""
    df = _raw_obs("overpass_a.h5")  # _raw_obs already sets metadata to None

    result = grid_bin(df, {"along_track": 2}, {"height": 2})

    assert len(result) == 4
    assert "superob" in result["metadata"].iloc[0]


def test_grid_bin_rejects_serialised_metadata():
    """Observations read back from an experiment database cannot be re-binned."""
    df = _raw_obs("overpass_a.h5")
    df["metadata"] = ['{"superob": {"n_contributing": 4}}' for _ in range(len(df))]

    with pytest.raises(ValueError, match="serialised metadata"):
        grid_bin(df, {"along_track": 2}, {"height": 2})


def test_count_by_window_counts_within_half_open_window():
    times = pd.Series(
        pd.to_datetime(
            [
                "2024-08-12 13:00:00",  # 1h before centre
                "2024-08-12 13:45:00",  # 15 min before
                "2024-08-12 14:15:00",  # 15 min after
                "2024-08-12 16:00:00",  # 2h after
            ],
            utc=True,
        )
    )
    centres = pd.DatetimeIndex([pd.Timestamp("2024-08-12 14:00:00", tz="UTC")])

    counts = count_by_window(times, centres, [1.0, 4.0, 8.0])

    assert counts.loc[1.0, "total"] == 2  # the two within +-30 min
    assert counts.loc[4.0, "total"] == 3  # adds the 13:00 observation
    assert counts.loc[8.0, "total"] == 4  # adds the 16:00 observation


def test_count_by_window_handles_naive_timestamps():
    """Observation frames may carry naive timestamps; they are treated as UTC."""
    naive = pd.Series(pd.to_datetime(["2024-08-12 14:00:00"]))
    centres = pd.DatetimeIndex([pd.Timestamp("2024-08-12 14:00:00", tz="UTC")])

    counts = count_by_window(naive, centres, [1.0])

    assert counts.loc[1.0, "total"] == 1


def test_count_by_window_mean_per_cycle():
    times = pd.Series(
        pd.to_datetime(
            ["2024-08-12 14:00:00", "2024-08-13 14:00:00", "2024-08-13 14:10:00"],
            utc=True,
        )
    )
    centres = pd.DatetimeIndex(
        [
            pd.Timestamp("2024-08-12 14:00:00", tz="UTC"),
            pd.Timestamp("2024-08-13 14:00:00", tz="UTC"),
        ]
    )

    counts = count_by_window(times, centres, [2.0])

    assert counts.loc[2.0, "total"] == 3
    assert counts.loc[2.0, "mean_per_cycle"] == 1.5


def test_parse_bin_spec():
    assert _parse_bin_spec(("along_track=27", "JSG_height=3")) == {
        "along_track": 27,
        "JSG_height": 3,
    }
    assert _parse_bin_spec(()) == {}


@pytest.mark.parametrize("bad", ["along_track", "along_track=x", "along_track=0"])
def test_parse_bin_spec_rejects_malformed(bad):
    with pytest.raises(click.BadParameter):
        _parse_bin_spec((bad,))
