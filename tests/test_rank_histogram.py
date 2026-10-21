from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from wrf_ensembly.config import ObservationsConfig
from wrf_ensembly.diagnostics import (
    RankHistogramAccumulator,
    compute_rank_histogram,
    rank_histogram,
    rank_in_ensemble,
)
from wrf_ensembly.experiment.observations import ExperimentObservations
from wrf_ensembly.observations import io
from wrf_ensembly.validation.rank_histogram import RankHistogramAnalysis


def _reliable(n=100_000, n_members=20, sigma_o=0.5, seed=1):
    """Truth and members drawn from the same distribution, obs = truth + obs error."""
    rng = np.random.default_rng(seed)
    centre = rng.normal(size=n)
    truth = centre + rng.normal(size=n)
    members = centre[:, None] + rng.normal(size=(n, n_members))
    obs = truth + rng.normal(scale=sigma_o, size=n)
    return obs, members, sigma_o


def test_reliable_ensemble_with_obs_error_is_flat():
    obs, members, sigma_o = _reliable()

    hist, _, diag = rank_histogram(obs, members, sigma_o)

    expected = len(obs) / (members.shape[1] + 1)
    assert np.all(np.abs(hist - expected) < 5 * np.sqrt(expected))
    assert diag["frac_below"] == pytest.approx(1 / 21, abs=0.003)
    assert diag["frac_above"] == pytest.approx(1 / 21, abs=0.003)
    assert diag["spread_skill"] == pytest.approx(1.0, abs=0.02)


def test_ignoring_the_obs_error_makes_a_reliable_ensemble_look_underdispersive():
    """The reason for perturbing the members: without it the end bins fill up."""
    obs, members, _ = _reliable(sigma_o=1.0)  # obs error as large as the spread

    hist, _, diag = rank_histogram(obs, members, 0.0)

    assert diag["frac_below"] + diag["frac_above"] > 1.5 * diag["frac_outside_expected"]
    assert hist[0] > 1.5 * hist[10]


def test_ties_are_split_evenly():
    """Bit-rounded members equal to the obs must not all land in rank 0."""
    n = 30_000
    members = np.full((n, 4), 0.2)
    obs = np.full(n, 0.2)

    ranks = rank_in_ensemble(obs, members, 0.0)

    assert np.bincount(ranks, minlength=5) == pytest.approx(np.full(5, n / 5), rel=0.05)


def test_rank_is_reproducible_by_default():
    obs, members, sigma_o = _reliable(n=1000)

    a = rank_in_ensemble(obs, members, sigma_o)
    b = rank_in_ensemble(obs, members, sigma_o)

    assert np.array_equal(a, b)


def test_compute_rank_histogram_reads_dart_columns():
    obs, members, sigma_o = _reliable(n=20_000, n_members=5)
    df = pd.DataFrame(
        {"obs": obs, "obs_variance": sigma_o**2, "prior_mean": members.mean(1), "prior_spread": members.std(1, ddof=1)}
        | {f"member_{i}_prior": members[:, i] for i in range(5)}
    )

    hist, ranks, diag = compute_rank_histogram(df)

    assert len(hist) == 6
    assert diag["n_ensemble"] == 5
    assert diag["frac_below"] == pytest.approx(1 / 6, abs=0.01)


def test_accumulating_in_chunks_matches_one_go():
    """Counts and diagnostics don't depend on how the observations are chunked."""
    obs, members, _ = _reliable(n=10_000, n_members=8)  # obs_std = 0 and no ties: ranks are exact

    one = RankHistogramAccumulator(8)
    one.add(obs, members)
    chunked = RankHistogramAccumulator(8)
    for part in np.array_split(np.arange(len(obs)), 7):
        chunked.add(obs[part], members[part])

    assert np.array_equal(one.histogram, chunked.histogram)
    a, b = one.diagnostics(), chunked.diagnostics()
    for key in a:
        assert a[key] == pytest.approx(b[key], rel=1e-9), key
    # and the sums give the same as the direct formulas
    mean = members.mean(1)
    assert a["correlation"] == pytest.approx(np.corrcoef(obs, mean)[0, 1], rel=1e-9)
    assert a["innovation_std"] == pytest.approx(np.std(obs - mean), rel=1e-9)


def _obs_db(tmp_path, n_members=3):
    """Four observations in three different hours, and a per-member parquet for them."""
    n = 4
    df = pd.DataFrame(
        {
            "instrument": "MTG_REACT",
            "quantity": "DOD_355nm",
            "time": pd.to_datetime(
                ["2026-03-26T12:00", "2026-03-26T12:10", "2026-03-26T13:00", "2026-03-26T15:00"], utc=True
            ),
            "longitude": np.arange(n, dtype=float), "latitude": 30.0,
            "x": 0.0, "y": 0.0, "z": 0.0, "z_type": "columnar",
            "value": [0.1, 0.2, 0.3, 0.4], "value_uncertainty": 0.1,
            "qc_flag": [0, -1, 0, 1],
            "orig_coords": [{"indices": [i], "shape": [n], "names": ["pixel"]} for i in range(n)],
            "orig_filename": "slot.parquet",
            "metadata": [{"model_id": "test"} for _ in range(n)],
        }
    )
    path = tmp_path / "slot.parquet"
    io.write_obs(df, path)
    validation = tmp_path / "validation"
    validation.mkdir()
    cfg = SimpleNamespace(
        observations=ObservationsConfig(),
        assimilation=SimpleNamespace(half_window_length_minutes=60, n_members=n_members),
    )
    obs = ExperimentObservations(
        cfg, [], SimpleNamespace(obs_db=tmp_path / "observations.duckdb", data_validation=validation)
    )
    obs.add_observation_file(path)

    # rowids 0..3; row 2 lacks member 2, row 1 has a NaN member; members written out of order
    rows = [(r, m, 10 * r + m) for r in range(n) for m in (2, 0, 1) if not (r == 2 and m == 2)]
    pm = pd.DataFrame(rows, columns=["rowid", "member", "value"]).astype({"rowid": "int64", "member": "int32"})
    pm.loc[(pm.rowid == 1) & (pm.member == 0), "value"] = np.nan
    pm["instrument"], pm["quantity"] = "MTG_REACT", "DOD_355nm"
    pm["time"] = df["time"].to_numpy()[pm["rowid"]]
    pm.to_parquet(validation / "model_member_forecast.parquet")
    return obs


@pytest.mark.parametrize("max_obs", [1, 2, 1000])
def test_iter_member_ensemble_pivots_and_drops_incomplete_rows(tmp_path, max_obs):
    obs = _obs_db(tmp_path)

    chunks = list(obs.iter_member_ensemble("MTG_REACT", "DOD_355nm", qc_flags=[0, 1], max_obs_per_chunk=max_obs))

    got_obs = pd.concat([c[0] for c in chunks], ignore_index=True)
    members = np.concatenate([c[1] for c in chunks])
    assert list(got_obs["value"]) == [0.1, 0.4]
    assert members.tolist() == [[0.0, 1.0, 2.0], [30.0, 31.0, 32.0]]
    assert list(obs.iter_member_ensemble("MTG_REACT", "DOD_355nm", qc_flags=[-1])) == []


def test_iter_member_ensemble_packs_hours_into_chunks(tmp_path):
    obs = _obs_db(tmp_path)

    sizes = [len(c[0]) for c in obs.iter_member_ensemble("MTG_REACT", "DOD_355nm", max_obs_per_chunk=2)]

    # hours 12 (2 obs, one of them incomplete), 13 (1, incomplete) and 15 (1): 12 alone, 13+15 together
    assert sizes == [1, 1]


def test_analysis_scales_sigma_by_the_inflation_factor_and_uses_abs(tmp_path):
    """The filter assumed factor x |value_uncertainty|; the histogram must too."""
    obs, members, sigma_o = _reliable(n=20_000, n_members=10, sigma_o=0.5)
    frame = pd.DataFrame({"value": obs, "value_uncertainty": sigma_o / 2, "qc_flag": 0})
    frame.loc[::10, "value_uncertainty"] *= -1  # a few negative ones, as VIIRS has
    exp = SimpleNamespace(
        cfg=SimpleNamespace(observations=SimpleNamespace(error_inflation_factor={"INSTR.Q": 2.0})),
        paths=SimpleNamespace(data_validation=tmp_path),
    )

    res = RankHistogramAnalysis(exp, "INSTR", "Q", "forecast").run(
        [(frame.iloc[:7000], members[:7000]), (frame.iloc[7000:], members[7000:])]
    )
    row = res["summary"].iloc[0]

    _, _, direct = rank_histogram(obs, members, sigma_o)
    assert row.sigma_o_factor == 2.0
    assert row.n_negative_uncertainty == 2000
    assert row.spread_skill == pytest.approx(direct["spread_skill"], rel=1e-9)
    assert (tmp_path / "rank_histograms" / "INSTR" / "Q" / "summary_forecast.csv").exists()
