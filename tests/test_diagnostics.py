import numpy as np
import pandas as pd

from wrf_ensembly.diagnostics import (
    compute_cycle_consistency,
    compute_forward_operator_metrics,
    consistency_metrics_to_dataframe,
    match_dart_to_db,
)


def _frame(obs, prior, posterior, prior_spread=1.0, obs_variance=1.0, dart_qc=0):
    """Build a minimal obs_seq-shaped frame for the consistency metrics."""
    n = len(obs)
    return pd.DataFrame(
        {
            "obs": np.asarray(obs, dtype=float),
            "prior_mean": np.asarray(prior, dtype=float),
            "posterior_mean": np.asarray(posterior, dtype=float),
            "prior_spread": np.full(n, prior_spread, dtype=float),
            "obs_variance": np.full(n, obs_variance, dtype=float),
            "dart_qc": np.full(n, dart_qc, dtype=float),
        }
    )


def test_consistency_known_values():
    """O-B of +2 everywhere, O-A of +1: RMS halves, so a 50% reduction."""
    df = _frame(obs=[3.0, 5.0, 7.0], prior=[1.0, 3.0, 5.0], posterior=[2.0, 4.0, 6.0])

    m = compute_cycle_consistency(df, cycle=0)

    assert m.n_total == 3
    assert m.n_assim == 3
    assert m.omb_mean == 2.0
    assert m.omb_rms == 2.0
    assert m.oma_mean == 1.0
    assert m.oma_rms == 1.0
    assert m.rms_reduction_pct == 50.0
    # obs and prior differ by a constant, so both correlations are exactly 1
    assert np.isclose(m.r_prior, 1.0)
    assert np.isclose(m.r_posterior, 1.0)
    assert m.mean_sigma_o == 1.0
    assert m.mean_prior_spread == 1.0
    # mean((O-B)^2) = 4, sigma_o^2 + spread^2 = 2
    assert np.isclose(m.spread_skill_ratio, 2.0)
    # O-B is a constant +2, all bias: nothing left once it is removed
    assert np.isclose(m.spread_skill_ratio_debiased, 0.0)
    # E[d_a . d_b] = 1 * 2 = 2
    assert np.isclose(m.sigma_o_desroziers, np.sqrt(2.0))
    # E[(A-B) . d_b] = 1 * 2 = 2
    assert np.isclose(m.sigma_b_desroziers, np.sqrt(2.0))


def test_consistency_qc_counts_use_all_rows():
    """QC counts span every row; the statistics only use the assimilated ones."""
    df = pd.concat(
        [
            _frame(obs=[3.0, 5.0], prior=[1.0, 3.0], posterior=[2.0, 4.0], dart_qc=0),
            _frame(obs=[99.0], prior=[0.0], posterior=[0.0], dart_qc=4),
            _frame(obs=[99.0, 99.0], prior=[0.0, 0.0], posterior=[0.0, 0.0], dart_qc=7),
        ],
        ignore_index=True,
    )

    m = compute_cycle_consistency(df, cycle=1)

    assert m.n_total == 5
    assert m.n_assim == 2
    assert m.n_qc4_forward_op_fail == 1
    assert m.n_qc7_outlier == 2
    # the rejected rows must not leak into the statistics
    assert m.omb_mean == 2.0
    assert m.omb_rms == 2.0


def test_consistency_empty_frame_is_all_nan():
    df = _frame(obs=[], prior=[], posterior=[])

    m = compute_cycle_consistency(df, cycle=0)

    assert m.n_total == 0
    assert m.n_assim == 0
    assert np.isnan(m.omb_rms)
    assert np.isnan(m.rms_reduction_pct)
    assert np.isnan(m.spread_skill_ratio)
    assert np.isnan(m.spread_skill_ratio_debiased)
    assert np.isnan(m.sigma_b_desroziers)
    assert np.isnan(m.r_prior)


def test_consistency_single_observation_is_all_nan():
    """One observation cannot support a correlation, so the row is nan."""
    df = _frame(obs=[3.0], prior=[1.0], posterior=[2.0])

    m = compute_cycle_consistency(df, cycle=0)

    assert m.n_assim == 1
    assert np.isnan(m.omb_rms)
    assert np.isnan(m.r_prior)


def test_consistency_no_assimilated_observations():
    """Every observation rejected: counts survive, statistics do not."""
    df = _frame(obs=[3.0, 5.0], prior=[1.0, 3.0], posterior=[2.0, 4.0], dart_qc=7)

    m = compute_cycle_consistency(df, cycle=2)

    assert m.n_total == 2
    assert m.n_assim == 0
    assert m.n_qc7_outlier == 2
    assert np.isnan(m.omb_rms)


def test_consistency_constant_series_gives_nan_correlation():
    """A constant prior makes the correlation undefined, not a warning."""
    df = _frame(obs=[1.0, 2.0, 3.0], prior=[5.0, 5.0, 5.0], posterior=[1.0, 2.0, 3.0])

    m = compute_cycle_consistency(df, cycle=0)

    assert np.isnan(m.r_prior)
    assert np.isclose(m.r_posterior, 1.0)


def test_consistency_zero_omb_gives_nan_reduction():
    """A perfect prior would divide by zero; report nan rather than inf."""
    df = _frame(obs=[1.0, 2.0, 3.0], prior=[1.0, 2.0, 3.0], posterior=[1.0, 2.0, 3.0])

    m = compute_cycle_consistency(df, cycle=0)

    assert m.omb_rms == 0.0
    assert np.isnan(m.rms_reduction_pct)


def test_consistency_nan_posterior_survives():
    """Cycles without a posterior still report the prior-side statistics."""
    df = _frame(
        obs=[3.0, 5.0, 7.0],
        prior=[1.0, 3.0, 5.0],
        posterior=[np.nan, np.nan, np.nan],
    )

    m = compute_cycle_consistency(df, cycle=0)

    assert m.omb_rms == 2.0
    assert np.isnan(m.oma_rms)
    assert np.isnan(m.rms_reduction_pct)


def test_metrics_to_dataframe_roundtrip():
    rows = [
        compute_cycle_consistency(
            _frame(obs=[3.0, 5.0], prior=[1.0, 3.0], posterior=[2.0, 4.0]),
            cycle=c,
            obs_type="TEST_TYPE",
        )
        for c in (0, 1)
    ]

    df = consistency_metrics_to_dataframe(rows)

    assert list(df["cycle"]) == [0, 1]
    assert set(df["obs_type"]) == {"TEST_TYPE"}
    assert "spread_skill_ratio" in df.columns
    # field order is preserved
    assert list(df.columns)[:2] == ["cycle", "obs_type"]


def test_consistency_desroziers_recovers_true_errors():
    """With a correctly specified optimal update, both Desroziers estimates come
    back to the true errors and the debiased ratio to 1."""
    rng = np.random.default_rng(0)
    n, sigma_b, sigma_o, bias = 200_000, 0.3, 0.2, 0.5
    truth = rng.normal(size=n)
    prior = truth + rng.normal(scale=sigma_b, size=n)
    obs = truth + rng.normal(scale=sigma_o, size=n) + bias
    gain = sigma_b**2 / (sigma_b**2 + sigma_o**2)
    posterior = prior + gain * (obs - bias - prior)  # the bias is known here, so d_b is debiased
    df = _frame(obs - bias, prior, posterior, prior_spread=sigma_b, obs_variance=sigma_o**2)
    df_biased = _frame(obs, prior, posterior + gain * bias, prior_spread=sigma_b, obs_variance=sigma_o**2)

    m = compute_cycle_consistency(df, cycle=0)
    mb = compute_cycle_consistency(df_biased, cycle=0)

    assert np.isclose(m.sigma_o_desroziers, sigma_o, rtol=0.02)
    assert np.isclose(m.sigma_b_desroziers, sigma_b, rtol=0.02)
    assert np.isclose(m.spread_skill_ratio, 1.0, rtol=0.02)
    # A bias inflates the plain ratio but not the debiased one
    assert mb.spread_skill_ratio > 1.0 + bias**2 / (sigma_b**2 + sigma_o**2) * 0.9
    assert np.isclose(mb.spread_skill_ratio_debiased, 1.0, rtol=0.02)


def _dart(obs, prior, lon, lat, obs_type="GOCART_AOD_500nm", t="2026-04-01T12:00:00"):
    n = len(obs)
    return pd.DataFrame(
        {
            "obs": np.asarray(obs, float),
            "prior_mean": np.asarray(prior, float),
            "longitude": np.asarray(lon, float),
            "latitude": np.asarray(lat, float),
            "obs_type": [obs_type] * n,
            "timestamp": pd.to_datetime([t] * n),
            "dart_qc": np.zeros(n),
        }
    )


def test_match_dart_to_db_uses_value_to_pick_the_instrument():
    """Two instruments share a quantity, site and time; the value decides."""
    dart = _dart(obs=[0.31, 0.52], prior=[0.2, 0.4], lon=[10.0, 350.0 - 360.0], lat=[35.0, 20.0])
    db = pd.DataFrame(
        {
            "instrument": ["AERONET", "AERONET_SDA", "AERONET", "OTHER"],
            "quantity": ["AOD_500nm"] * 3 + ["DOD_355nm"],
            # DB times are UTC-aware and stored with sub-second noise
            "time": pd.to_datetime(["2026-04-01T12:00:00.0004"] * 4, utc=True),
            "longitude": [10.0, 10.0, -10.0, 10.0],
            "latitude": [35.0, 35.0, 20.0, 35.0],
            "value": [0.30, 0.31, 0.52, 0.31],
            "model_forecast": [9.0, 0.25, 0.5, 9.0],
        }
    )

    m = match_dart_to_db(dart, db)

    assert len(m) == 2
    assert list(m.sort_values("obs")["instrument"]) == ["AERONET_SDA", "AERONET"]
    assert list(m.sort_values("obs")["model_forecast"]) == [0.25, 0.5]


def test_match_dart_to_db_leaves_unmatched_out():
    dart = _dart(obs=[0.3, 0.3], prior=[0.2, 0.2], lon=[10.0, 11.0], lat=[35.0, 35.0])
    db = pd.DataFrame(
        {
            "instrument": ["AERONET"],
            "quantity": ["AOD_500nm"],
            "time": pd.to_datetime(["2026-04-01T13:00:00"], utc=True),  # other time
            "longitude": [10.0],
            "latitude": [35.0],
            "value": [0.3],
            "model_forecast": [0.25],
        }
    )

    assert match_dart_to_db(dart, db).empty


def test_forward_operator_metrics_factor():
    """DART sees 0.7x the interpolated model: slope and median ratio say so."""
    dart = _dart(obs=[1, 2, 3, 4], prior=[0.7, 1.4, 2.1, 2.8], lon=[0, 1, 2, 3], lat=[0, 0, 0, 0])
    matched = dart.iloc[:3].assign(model_forecast=[1.0, 2.0, 3.0])

    m = compute_forward_operator_metrics(dart, matched, cycle=5)

    assert m.n_dart == 4
    assert m.n_matched == 3
    assert np.isclose(m.slope, 0.7)
    assert np.isclose(m.median_ratio, 0.7)
    assert np.isclose(m.r, 1.0)
