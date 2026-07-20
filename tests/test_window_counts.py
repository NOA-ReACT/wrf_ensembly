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


def test_grid_bin_merges_across_source_files():
    """
    grid_bin groups only on binned indices, with no source file in the key.

    Two overpasses covering the same native grid indices therefore collapse into
    each other if they are binned in one call. This is why `superob_files` calls
    grid_bin once per file, and is a hazard for any other caller that concatenates
    first.
    """
    a = _raw_obs("overpass_a.h5")
    b = _raw_obs("overpass_b.h5")
    both = pd.concat([a, b], ignore_index=True)

    bins_h, bins_v = {"along_track": 2}, {"height": 2}

    merged = grid_bin(both, bins_h, bins_v)
    per_file = pd.concat(
        [grid_bin(a, bins_h, bins_v), grid_bin(b, bins_h, bins_v)], ignore_index=True
    )

    # Binning each file gives 2x2 superobs per file, so 8 in total
    assert len(per_file) == 8
    # Binning them together collapses the two overpasses onto one grid
    assert len(merged) == 4
    assert len(merged) < len(per_file)


def test_grid_bin_merge_loses_source_provenance():
    """A merged superob silently inherits whichever file happened to sort first."""
    both = pd.concat(
        [_raw_obs("overpass_a.h5"), _raw_obs("overpass_b.h5")], ignore_index=True
    )

    merged = grid_bin(both, {"along_track": 2}, {"height": 2})

    # Both overpasses contributed, but only one filename survives
    assert set(merged["orig_filename"]) == {"overpass_a.h5"}


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
