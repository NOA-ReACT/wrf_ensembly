"""Utilities for reading and computing statistics from DART filter diagnostics."""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr


def _find_copy_indices(
    fields: Iterable[str], copy_metadata: xr.DataArray
) -> dict[str, int]:
    """Find the index of each field name in the CopyMetaData variable."""
    indices = {}
    for field in fields:
        indices[field] = (copy_metadata == field).argmax().item()
    return indices


def read_obs_seq_nc(path: Path) -> pd.DataFrame:
    """
    Read a DART obs_seq NetCDF file (produced by obs_seq_to_netcdf) into a DataFrame.

    Reads all observation types present in the file. Returns a DataFrame with columns:
    obs, obs_variance, prior_mean, prior_spread, posterior_mean, posterior_spread,
    longitude, latitude, z, dart_qc, obs_type, timestamp.

    Args:
        path: Path to the NetCDF file (e.g. data/diagnostics/cycle_0.nc)

    Returns:
        DataFrame with one row per observation
    """
    ds = xr.open_dataset(path).load()

    # Fix fixed-length string fields
    ds["ObsTypesMetaData"] = ds["ObsTypesMetaData"].astype(str).str.strip()
    ds["QCMetaData"] = ds["QCMetaData"].astype(str).str.strip()
    ds["CopyMetaData"] = ds["CopyMetaData"].astype(str).str.strip()

    # CopyMetaData has some padding created with spaces that might not be constant every time
    # So we use some regex to replace multiple spaces with a single space
    ds["CopyMetaData"] = ds["CopyMetaData"].str.replace(r"\s+", " ", regex=True)

    # Build obs_type_id -> name mapping for all types present
    obs_type_ids = {}
    for i in range(ds.sizes.get("ObsTypes", 0)):
        name = ds["ObsTypesMetaData"].isel(ObsTypes=i).item()
        type_id = ds["ObsTypes"].isel(ObsTypes=i).item()
        obs_type_ids[type_id] = name

    # From CopyMetaData, find how many members we have and generate all the per_member value columns
    # The columns are called like `prior ensemble member     XX` and `posterior ensemble member     XX`
    # with a varying amount of spaces. XX is the member ID, starting from 1, no leading zeros.
    # We have fixed the spacing to use a single space instead of varying amounts above.
    max_member = (
        ds["CopyMetaData"].str.contains(r"^prior ensemble member \d+$").sum().item()
    )
    per_member_columns = {
        f"member_{i}_prior": f"prior ensemble member {i + 1}" for i in range(max_member)
    } | {
        f"member_{i}_posterior": f"posterior ensemble member {i + 1}"
        for i in range(max_member)
    }

    # Find copy indices for the fields we need
    column_mappings = {
        "obs": "observation",
        "obs_variance": "observation error variance",
        "prior_mean": "prior ensemble mean",
        "prior_spread": "prior ensemble spread",
        "posterior_mean": "posterior ensemble mean",
        "posterior_spread": "posterior ensemble spread",
    } | per_member_columns
    copy_indices = _find_copy_indices(column_mappings.values(), ds["CopyMetaData"])

    # Find DART QC index
    dart_qc_index = (ds["QCMetaData"] == "DART quality control").argmax().item()

    # "Wide" to long transform
    result = {}
    for col_name, copy_name in column_mappings.items():
        idx = copy_indices[copy_name]
        result[col_name] = ds["observations"].isel(copy=idx).values

    # obs_seq stores longitude in 0-360 degrees; normalise to -180..180 so it
    # matches the DB / model-domain convention used everywhere else (otherwise
    # obs west of the prime meridian wrap to ~347-360 and land-sea / map lookups
    # and plots place them in the wrong hemisphere).
    lon = ds["location"].isel(locdim=0).values
    result["longitude"] = ((lon + 180.0) % 360.0) - 180.0
    result["latitude"] = ds["location"].isel(locdim=1).values
    result["z"] = ds["location"].isel(locdim=2).values
    result["dart_qc"] = ds["qc"].isel(qc_copy=dart_qc_index).values
    result["timestamp"] = pd.to_datetime(ds["time"].values)

    # Observation error is given in variance, convert to stdev
    result["obs_std"] = np.sqrt(result["obs_variance"])

    # Map obs_type IDs to names
    obs_type_raw = ds["obs_type"].values
    result["obs_type"] = np.array(
        [obs_type_ids.get(int(t), f"unknown_{int(t)}") for t in obs_type_raw]
    )

    ds.close()
    return pd.DataFrame(result)


def rank_in_ensemble(
    obs: np.ndarray,
    ensemble: np.ndarray,
    obs_std: np.ndarray | float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Rank of each observation among its ensemble, accounting for observation error.

    Each member is perturbed with N(0, obs_std^2) before ranking (Hamill 2001). An
    ensemble that is a sample from the same distribution as the truth then ranks the
    observation (truth + observation error) uniformly. Perturbing the observation
    instead would give it the observation error twice, so even a perfect ensemble
    would look under-dispersive. Remaining ties (e.g. obs_std = 0 with
    bit-rounded members) are broken at random.

    Args:
        obs: Observations, shape (n,).
        ensemble: Member values at the observations, shape (n, n_members).
        obs_std: Observation error standard deviation, scalar or shape (n,).
        rng: Random generator. Defaults to a fixed seed, so the result is reproducible.

    Returns:
        Integer ranks 0 ... n_members, shape (n,).
    """
    rng = np.random.default_rng(0) if rng is None else rng
    obs = np.asarray(obs, dtype=float)
    ensemble = np.asarray(ensemble, dtype=float)
    obs_std = np.broadcast_to(np.asarray(obs_std, dtype=float), obs.shape)
    perturbed = ensemble + rng.normal(size=ensemble.shape) * obs_std[:, np.newaxis]
    below = np.sum(perturbed < obs[:, np.newaxis], axis=1)
    ties = np.sum(perturbed == obs[:, np.newaxis], axis=1)
    return below + rng.integers(0, ties + 1)


class RankHistogramAccumulator:
    """
    Rank histogram and its diagnostics, accumulated over chunks of observations.

    Only counts and sums are kept, so memory does not grow with the number of
    observations: feed it one chunk at a time with `add`, then read `histogram` and
    `diagnostics()`. Ranks come from `rank_in_ensemble`, so observation error is
    accounted for by perturbing the members.
    """

    _SUMS = (
        "n", "innov", "innov2", "n_norm", "norm", "norm2", "total_var", "spread2",
        "obs", "obs2", "mean", "mean2", "obs_mean",
    )

    def __init__(self, n_members: int, rng: np.random.Generator | None = None):
        self.n_members = n_members
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.histogram = np.zeros(n_members + 1, dtype=np.int64)
        self.sums = dict.fromkeys(self._SUMS, 0.0)

    def add(
        self,
        obs: np.ndarray,
        ensemble: np.ndarray,
        obs_std: np.ndarray | float = 0.0,
        ensemble_mean: np.ndarray | None = None,
        ensemble_spread: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Add a chunk of observations.

        Args:
            obs: Observations, shape (n,).
            ensemble: Member values, shape (n, n_members).
            obs_std: Observation error standard deviation, scalar or shape (n,).
            ensemble_mean, ensemble_spread: Mean and standard deviation to use for the
                innovation statistics, e.g. DART's own. Computed from `ensemble`
                (spread with ddof = 1) when not given.

        Returns:
            The ranks of this chunk's observations.
        """
        obs = np.asarray(obs, dtype=float)
        ensemble = np.asarray(ensemble, dtype=float)
        if ensemble.shape[1] != self.n_members:
            raise ValueError(f"Expected {self.n_members} members, got {ensemble.shape[1]}")
        obs_std = np.broadcast_to(np.asarray(obs_std, dtype=float), obs.shape)
        mean = ensemble.mean(axis=1) if ensemble_mean is None else np.asarray(ensemble_mean, dtype=float)
        spread = (
            ensemble.std(axis=1, ddof=1) if ensemble_spread is None else np.asarray(ensemble_spread, dtype=float)
        )

        ranks = rank_in_ensemble(obs, ensemble, obs_std, self.rng)
        self.histogram += np.bincount(ranks, minlength=self.n_members + 1)

        innov = obs - mean
        total_var = spread**2 + obs_std**2
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = innov / np.sqrt(total_var)
        finite = np.isfinite(norm)
        s = self.sums
        s["n"] += len(obs)
        s["innov"] += innov.sum()
        s["innov2"] += (innov**2).sum()
        s["n_norm"] += finite.sum()
        s["norm"] += norm[finite].sum()
        s["norm2"] += (norm[finite] ** 2).sum()
        s["total_var"] += total_var.sum()
        s["spread2"] += (spread**2).sum()
        s["obs"] += obs.sum()
        s["obs2"] += (obs**2).sum()
        s["mean"] += mean.sum()
        s["mean2"] += (mean**2).sum()
        s["obs_mean"] += (obs * mean).sum()
        return ranks

    def diagnostics(self) -> dict:
        """Summary statistics over everything added so far."""
        s = self.sums
        n = s["n"]
        nan = float("nan")
        if n == 0:
            return {"n_obs": 0, "n_ensemble": self.n_members, "frac_below": nan, "frac_above": nan,
                    "frac_outside_expected": 2 / (self.n_members + 1), "innovation_mean": nan,
                    "innovation_std": nan, "normalized_innov_mean": nan, "normalized_innov_std": nan,
                    "correlation": nan, "rmse": nan, "mean_spread": nan, "spread_skill": nan}

        innov_mean = s["innov"] / n
        rmse = np.sqrt(s["innov2"] / n)
        norm_mean = s["norm"] / s["n_norm"] if s["n_norm"] else nan
        norm_var = s["norm2"] / s["n_norm"] - norm_mean**2 if s["n_norm"] else nan
        var_obs = s["obs2"] / n - (s["obs"] / n) ** 2
        var_mean = s["mean2"] / n - (s["mean"] / n) ** 2
        cov = s["obs_mean"] / n - (s["obs"] / n) * (s["mean"] / n)
        corr = cov / np.sqrt(var_obs * var_mean) if n > 1 and var_obs > 0 and var_mean > 0 else nan
        return {
            "n_obs": int(n),
            "n_ensemble": self.n_members,
            "frac_below": self.histogram[0] / n,
            "frac_above": self.histogram[-1] / n,
            "frac_outside_expected": 2 / (self.n_members + 1),
            "innovation_mean": float(innov_mean),
            "innovation_std": float(np.sqrt(max(s["innov2"] / n - innov_mean**2, 0.0))),
            "normalized_innov_mean": float(norm_mean),
            "normalized_innov_std": float(np.sqrt(max(norm_var, 0.0))),
            "correlation": float(corr),
            "rmse": float(rmse),
            "mean_spread": float(np.sqrt(s["spread2"] / n)),
            # ~1 when the ensemble spread and the obs error account for the error of the mean
            "spread_skill": float(np.sqrt(s["total_var"] / n) / rmse) if rmse > 0 else nan,
        }


def rank_histogram(
    obs: np.ndarray,
    ensemble: np.ndarray,
    obs_std: np.ndarray | float = 0.0,
    ensemble_mean: np.ndarray | None = None,
    ensemble_spread: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Rank histogram of observations against an ensemble, with summary diagnostics.

    A flat histogram means a reliable ensemble; a U shape an under-dispersive one
    (or a biased one, if lopsided); a dome an over-dispersive one. For more
    observations than fit in memory at once, use `RankHistogramAccumulator`.

    Args:
        obs: Observations, shape (n,).
        ensemble: Member values, shape (n, n_members).
        obs_std: Observation error standard deviation, scalar or shape (n,).
        ensemble_mean, ensemble_spread: See `RankHistogramAccumulator.add`.
        rng: Random generator, see `rank_in_ensemble`.

    Returns:
        (histogram counts of length n_members + 1, per-observation ranks, diagnostics)
    """
    acc = RankHistogramAccumulator(np.asarray(ensemble).shape[1], rng)
    ranks = acc.add(obs, ensemble, obs_std, ensemble_mean, ensemble_spread)
    return acc.histogram, ranks, acc.diagnostics()


def compute_rank_histogram(
    df: pd.DataFrame,
    use_posterior: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Compute a rank histogram from a DART diagnostics DataFrame.

    Ranks each observation within the ensemble member values, with the members
    perturbed by the observation error (see `rank_in_ensemble`).

    Args:
        df: DataFrame with columns obs, obs_variance, prior_mean, prior_spread,
            and member_X_prior / member_X_posterior columns for each ensemble member X
            (zero-indexed, no leading zeros), and optionally posterior_mean,
            posterior_spread.
        use_posterior: Rank against the posterior members instead of the prior.
        rng: Random generator, see `rank_in_ensemble`.

    Returns:
        Tuple of (histogram, ranks, diagnostics_dict).
    """
    stage = "posterior" if use_posterior else "prior"

    member_cols = sorted(
        [c for c in df.columns if c.endswith(f"_{stage}") and c.startswith("member_")],
        key=lambda c: int(c.split("_")[1]),
    )
    if not member_cols:
        raise ValueError(f"No member columns found for stage '{stage}'.")

    return rank_histogram(
        df["obs"].to_numpy(dtype=float),
        df[member_cols].to_numpy(dtype=float),
        np.sqrt(df["obs_variance"].to_numpy(dtype=float)),
        ensemble_mean=df[f"{stage}_mean"].to_numpy(dtype=float),
        ensemble_spread=df[f"{stage}_spread"].to_numpy(dtype=float),
        rng=rng,
    )


def _rms(x) -> float:
    """Root-mean-square of the finite entries. NaN when nothing is finite."""

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x**2)))


def _correlation(a, b) -> float:
    """
    Pearson correlation over the pairwise-finite entries.

    Returns NaN rather than warning when fewer than two pairs survive or when
    either series is constant (`np.corrcoef` divides by a zero standard deviation
    in that case).
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _safe_mean(x) -> float:
    """Mean of the finite entries. NaN when nothing is finite."""

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


@dataclass
class CycleConsistencyMetrics:
    """
    Data assimilation consistency diagnostics for one cycle.

    O-B is the innovation before the analysis (obs minus prior mean) and O-A the
    residual after it (obs minus posterior mean). A working filter reduces the RMS
    from the former to the latter.
    """

    cycle: int
    obs_type: str
    """Observation type these metrics cover, or "ALL" for every type combined."""

    n_total: int
    """Observations in the cycle, at any quality control flag."""
    n_assim: int
    """Observations actually assimilated (DART QC 0). All statistics below use these."""
    n_qc4_forward_op_fail: int
    n_qc7_outlier: int

    omb_mean: float
    omb_rms: float
    oma_mean: float
    oma_rms: float
    rms_reduction_pct: float
    """Percentage drop from the O-B RMS to the O-A RMS. Positive means the analysis
    moved the model toward the observations."""

    r_prior: float
    r_posterior: float
    mean_sigma_o: float
    """Mean assigned observation error standard deviation."""
    mean_prior_spread: float

    spread_skill_ratio: float
    """mean((O-B)^2) / mean(sigma_o^2 + prior_spread^2), which is ~1 when the
    assigned observation error and the ensemble spread together explain the
    innovations. Below 1 means the observation error is over-specified
    (observations are under-weighted and increments are conservative); above 1
    means it is under-specified, risking over-fitting. The mean O-B is included,
    so a bias alone pushes this above 1."""
    spread_skill_ratio_debiased: float
    """The same with the cycle's mean O-B removed first, i.e. var(O-B) /
    mean(sigma_o^2 + prior_spread^2). When this is ~1 but `spread_skill_ratio` is
    not, the error budget is fine and the excess is a bias the filter cannot
    explain with random errors."""

    sigma_o_desroziers: float
    """Desroziers estimate of the observation error, sqrt(E[d_a . d_b]). Compare
    against `mean_sigma_o`: a large gap means the assigned error should move
    toward this value."""
    sigma_b_desroziers: float
    """Desroziers estimate of the background error in observation space,
    sqrt(E[(A-B) . d_b]). Compare against `mean_prior_spread`: an estimate well
    above the spread means the ensemble is under-dispersive (more inflation)."""


def compute_cycle_consistency(
    df: pd.DataFrame, cycle: int, obs_type: str = "ALL"
) -> CycleConsistencyMetrics:
    """
    Compute consistency diagnostics for one cycle's obs_seq.final table.

    Quality control counts are taken over every row; every other statistic is
    computed over the assimilated subset (DART QC 0) only. Values are in the
    observation's native units - no unit conversion is applied.

    Args:
        df: Frame from `read_obs_seq_nc`, containing all QC values. Filter it by
            observation type before calling if `obs_type` is not "ALL".
        cycle: Cycle index, recorded on the result.
        obs_type: Label for the observation type covered, or "ALL".

    Returns:
        Metrics for the cycle. Statistical fields are NaN when fewer than two
        observations were assimilated.
    """

    n_total = len(df)
    dart_qc = df["dart_qc"] if "dart_qc" in df.columns else pd.Series(dtype=float)
    n_qc4 = int((dart_qc == 4).sum())
    n_qc7 = int((dart_qc == 7).sum())

    assimilated = df[df["dart_qc"] == 0] if n_total else df
    n_assim = len(assimilated)

    nan = float("nan")
    if n_assim < 2:
        return CycleConsistencyMetrics(
            cycle=cycle,
            obs_type=obs_type,
            n_total=n_total,
            n_assim=n_assim,
            n_qc4_forward_op_fail=n_qc4,
            n_qc7_outlier=n_qc7,
            omb_mean=nan,
            omb_rms=nan,
            oma_mean=nan,
            oma_rms=nan,
            rms_reduction_pct=nan,
            r_prior=nan,
            r_posterior=nan,
            mean_sigma_o=nan,
            mean_prior_spread=nan,
            spread_skill_ratio=nan,
            spread_skill_ratio_debiased=nan,
            sigma_o_desroziers=nan,
            sigma_b_desroziers=nan,
        )

    obs = assimilated["obs"].to_numpy(dtype=float)
    prior_mean = assimilated["prior_mean"].to_numpy(dtype=float)
    posterior_mean = assimilated["posterior_mean"].to_numpy(dtype=float)
    prior_spread = assimilated["prior_spread"].to_numpy(dtype=float)
    sigma_o = np.sqrt(assimilated["obs_variance"].to_numpy(dtype=float))

    omb = obs - prior_mean
    oma = obs - posterior_mean
    amb = posterior_mean - prior_mean

    omb_rms = _rms(omb)
    oma_rms = _rms(oma)
    if not np.isfinite(omb_rms) or omb_rms == 0.0 or not np.isfinite(oma_rms):
        rms_reduction_pct = nan
    else:
        rms_reduction_pct = 100.0 * (1.0 - oma_rms / omb_rms)

    # mean((O-B)^2) should equal sigma_o^2 + prior_spread^2 if the error budget is right
    mean_omb_sq = _safe_mean(omb**2)
    var_omb = _safe_mean((omb - _safe_mean(omb)) ** 2)
    denominator = _safe_mean(sigma_o**2) + _safe_mean(prior_spread**2)
    if not np.isfinite(denominator) or denominator == 0.0:
        spread_skill_ratio = nan
        spread_skill_ratio_debiased = nan
    else:
        spread_skill_ratio = mean_omb_sq / denominator
        spread_skill_ratio_debiased = var_omb / denominator

    # Desroziers: E[d_a . d_b] estimates the observation error variance, and
    # E[(A-B) . d_b] the background error variance, both in observation space
    def _sqrt_or_nan(var: float) -> float:
        return float(np.sqrt(max(var, 0.0))) if np.isfinite(var) else nan

    sigma_o_desroziers = _sqrt_or_nan(_safe_mean(oma * omb))
    sigma_b_desroziers = _sqrt_or_nan(_safe_mean(amb * omb))

    return CycleConsistencyMetrics(
        cycle=cycle,
        obs_type=obs_type,
        n_total=n_total,
        n_assim=n_assim,
        n_qc4_forward_op_fail=n_qc4,
        n_qc7_outlier=n_qc7,
        omb_mean=_safe_mean(omb),
        omb_rms=omb_rms,
        oma_mean=_safe_mean(oma),
        oma_rms=oma_rms,
        rms_reduction_pct=rms_reduction_pct,
        r_prior=_correlation(obs, prior_mean),
        r_posterior=_correlation(obs, posterior_mean),
        mean_sigma_o=_safe_mean(sigma_o),
        mean_prior_spread=_safe_mean(prior_spread),
        spread_skill_ratio=spread_skill_ratio,
        spread_skill_ratio_debiased=spread_skill_ratio_debiased,
        sigma_o_desroziers=sigma_o_desroziers,
        sigma_b_desroziers=sigma_b_desroziers,
    )


def consistency_metrics_to_dataframe(
    rows: Iterable[CycleConsistencyMetrics],
) -> pd.DataFrame:
    """Collect consistency metrics into a DataFrame, one row each, in field order."""

    columns = [f.name for f in fields(CycleConsistencyMetrics)]
    records = [{c: getattr(row, c) for c in columns} for row in rows]
    return pd.DataFrame(records, columns=columns)


# ---------------------------------------------------------------------------
# Forward operator check: what the filter saw vs the postprocessed model
# ---------------------------------------------------------------------------


def match_dart_to_db(
    dart: pd.DataFrame, db: pd.DataFrame, max_distance_deg: float = 1e-4
) -> pd.DataFrame:
    """
    Find the database observation behind each DART observation.

    The obs_seq is written from the database (`convert_to_dart_obs_seq`) without any
    identifier, so observations are matched on their quantity (through the DART
    observation type), their time to the second, their position and their value.
    The value disambiguates instruments that share a quantity and a site (e.g.
    AERONET and AERONET_SDA AOD at 500 nm).

    Args:
        dart: Frame from `read_obs_seq_nc`.
        db: Database rows with instrument, quantity, time, longitude, latitude,
            value and model_forecast.
        max_distance_deg: Largest lon/lat distance still counted as the same place.

    Returns:
        The matched DART rows, with `instrument`, `db_value` and `model_forecast`
        added. Unmatched DART rows are left out.
    """
    from scipy.spatial import cKDTree

    from wrf_ensembly.observations.dart import OBS_TYPE_TABLE

    quantity_of = {v: k for k, v in OBS_TYPE_TABLE.items()}
    dart = dart.assign(
        quantity=dart["obs_type"].map(lambda t: quantity_of.get(t, t)),
        _t=pd.to_datetime(dart["timestamp"]).dt.round("s").dt.tz_localize(None),
    )
    db = db.assign(
        _t=pd.to_datetime(db["time"], utc=True).dt.round("s").dt.tz_localize(None),
        longitude=((db["longitude"] + 180.0) % 360.0) - 180.0,
    )

    parts = []
    db_groups = {key: g for key, g in db.groupby(["quantity", "_t"])}
    for key, d in dart.groupby(["quantity", "_t"]):
        candidates = db_groups.get(key)
        if candidates is None:
            continue
        k = min(4, len(candidates))
        dist, idx = cKDTree(candidates[["longitude", "latitude"]].to_numpy()).query(
            d[["longitude", "latitude"]].to_numpy(),
            k=k,
            distance_upper_bound=max_distance_deg,
        )
        dist, idx = dist.reshape(len(d), k), idx.reshape(len(d), k)
        values = np.append(candidates["value"].to_numpy(), np.nan)  # idx == len -> no hit
        obs = d["obs"].to_numpy()[:, None]
        same = (np.abs(values[idx] - obs) <= 1e-6 * np.maximum(1.0, np.abs(obs))) & np.isfinite(dist)
        hit = same.any(axis=1)
        first = idx[np.arange(len(d)), same.argmax(axis=1)][hit]
        picked = candidates.iloc[first]
        parts.append(
            d[hit].assign(
                instrument=picked["instrument"].to_numpy(),
                db_value=picked["value"].to_numpy(),
                model_forecast=picked["model_forecast"].to_numpy(),
            )
        )

    if not parts:
        return dart.iloc[0:0].assign(instrument=[], db_value=[], model_forecast=[]).drop(columns=["_t"])
    return pd.concat(parts).drop(columns=["_t"])


@dataclass
class ForwardOperatorMetrics:
    """
    DART's prior ensemble mean at the assimilated observations against the
    postprocessed forecast mean interpolated to the same observations
    (`validation interpolate-model`).

    These should agree. A factor between them is something the filter sees but
    the validation does not, e.g. different optics in DART's forward operator and
    in the postprocessing. DART evaluates every observation in the window against
    the state at the analysis time, while the interpolation follows the obs time,
    so some scatter is expected for wide windows; the slope should still be ~1.
    """

    cycle: int
    obs_type: str
    n_dart: int
    """Observations DART evaluated (finite prior mean)."""
    n_matched: int
    """Of those, found in the database with a model_forecast."""
    mean_dart_prior: float
    mean_model_forecast: float
    slope: float
    """Least-squares slope through zero of DART prior mean on model_forecast."""
    median_ratio: float
    """Median of DART prior mean / model_forecast where model_forecast > 0."""
    r: float


def compute_forward_operator_metrics(
    dart: pd.DataFrame, matched: pd.DataFrame, cycle: int, obs_type: str = "ALL"
) -> ForwardOperatorMetrics:
    """
    Compare DART's prior mean with the interpolated model at the matched observations.

    Args:
        dart: Frame from `read_obs_seq_nc`, filtered to the observations considered.
        matched: The rows of `dart` that `match_dart_to_db` matched.
        cycle: Cycle index, recorded on the result.
        obs_type: Label for the observation type covered, or "ALL".
    """
    n_dart = int(np.isfinite(dart["prior_mean"]).sum()) if len(dart) else 0
    p = matched["prior_mean"].to_numpy(dtype=float)
    m = matched["model_forecast"].to_numpy(dtype=float)
    ok = np.isfinite(p) & np.isfinite(m)
    p, m = p[ok], m[ok]

    nan = float("nan")
    denom = float(np.sum(m**2))
    positive = m > 0
    return ForwardOperatorMetrics(
        cycle=cycle,
        obs_type=obs_type,
        n_dart=n_dart,
        n_matched=int(ok.sum()),
        mean_dart_prior=_safe_mean(p),
        mean_model_forecast=_safe_mean(m),
        slope=float(np.sum(p * m) / denom) if denom > 0 else nan,
        median_ratio=float(np.median(p[positive] / m[positive])) if positive.any() else nan,
        r=_correlation(p, m),
    )


def forward_operator_metrics_to_dataframe(
    rows: Iterable[ForwardOperatorMetrics],
) -> pd.DataFrame:
    """Collect forward operator metrics into a DataFrame, one row each, in field order."""

    columns = [f.name for f in fields(ForwardOperatorMetrics)]
    records = [{c: getattr(row, c) for c in columns} for row in rows]
    return pd.DataFrame(records, columns=columns)
