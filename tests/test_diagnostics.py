import numpy as np
import pandas as pd

from wrf_ensembly.diagnostics import (
    compute_cycle_consistency,
    consistency_metrics_to_dataframe,
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
    # var(O-B) = 4, sigma_o^2 + spread^2 = 2
    assert np.isclose(m.spread_skill_ratio, 2.0)
    # E[d_a . d_b] = 1 * 2 = 2
    assert np.isclose(m.sigma_o_desroziers, np.sqrt(2.0))


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
