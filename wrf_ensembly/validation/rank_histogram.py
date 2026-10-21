"""Rank histograms of observations against the ensemble members.

The DART diagnostics give rank histograms only for the observations the filter
assimilated. These come from the per-member interpolation instead
(`validation interpolate-model-per-member`), so they cover every observation in
the database: held-out ones (e.g. superobs thinned out, qc_flag = -1), and
instruments that were never assimilated, i.e. independent verification of the
ensemble's reliability.
"""

from collections.abc import Iterable
from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wrf_ensembly.console import logger
from wrf_ensembly.diagnostics import RankHistogramAccumulator
from wrf_ensembly.experiment.experiment import Experiment
from wrf_ensembly.plotting import plot_rank_histogram


@dataclass
class RankHistogramSummary:
    """Reliability summary for one (source, QC flag) group of observations."""

    source: str
    """'forecast' (the prior, when the files are the DA cycles') or 'analysis'."""
    qc_flag: str
    """The observations' QC flag, or 'all' when several were pooled."""
    n_obs: int
    n_members: int
    frac_below: float
    """Share of observations below every (perturbed) member."""
    frac_above: float
    """Share of observations above every (perturbed) member."""
    frac_outside_expected: float
    """2 / (n_members + 1), what frac_below + frac_above is for a reliable ensemble."""
    frac_below_no_obs_error: float
    """frac_below ranking against the raw members, without the obs error."""
    frac_above_no_obs_error: float
    innovation_mean: float
    """Mean O - ensemble mean."""
    rmse: float
    """RMS of O - ensemble mean."""
    mean_spread: float
    """RMS ensemble spread at the observations."""
    spread_skill: float
    """sqrt(mean(spread^2 + sigma_o^2)) / rmse, ~1 when the spread and the obs error
    account for the error of the ensemble mean."""


class RankHistogramAnalysis:
    """
    Rank histograms of one instrument/quantity against the per-member model values.

    Members are perturbed with the observation's own uncertainty before ranking
    (Hamill 2001, see `diagnostics.rank_in_ensemble`); the unperturbed ranking is
    kept alongside, since it shows how much of the reliability is down to the
    observation error.
    """

    def __init__(self, experiment: Experiment, instrument: str, quantity: str, source: str):
        self.exp = experiment
        self.instrument = instrument
        self.quantity = quantity
        self.source = source
        self.output_dir = (
            experiment.paths.data_validation / "rank_histograms" / instrument / quantity
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, chunks: Iterable[tuple[pd.DataFrame, np.ndarray]]) -> dict | None:
        """
        Args:
            chunks: (observations, members) pairs, e.g. from
                `ExperimentObservations.iter_member_ensemble`. Observations need value,
                value_uncertainty and qc_flag; members are aligned with them, shape
                (n_obs, n_members). Only counts and sums are kept between chunks.

        Returns:
            Dict with the output directory, the summary and the counts, or None if
            there were no observations.
        """
        perturbed: dict[str, RankHistogramAccumulator] = {}
        raw: dict[str, RankHistogramAccumulator] = {}
        flags: set[str] = set()
        rng = np.random.default_rng(0)
        for obs, members in chunks:
            n_members = members.shape[1]
            qc = obs["qc_flag"].to_numpy()
            values = obs["value"].to_numpy(dtype=float)
            sigma = obs["value_uncertainty"].fillna(0.0).to_numpy(dtype=float)
            for label, sel in [("all", slice(None))] + [(str(f), qc == f) for f in np.unique(qc)]:
                if label != "all":
                    flags.add(label)
                for accs, s in ((perturbed, sigma[sel]), (raw, 0.0)):
                    if label not in accs:
                        accs[label] = RankHistogramAccumulator(n_members, rng)
                    accs[label].add(values[sel], members[sel], s)

        if not perturbed:
            return None

        # "all" only adds information when several flags were pooled
        labels = sorted(flags, key=int) + (["all"] if len(flags) > 1 else [])
        summaries, counts = [], []
        for label in labels:
            diag, diag_raw = perturbed[label].diagnostics(), raw[label].diagnostics()
            hist, hist_raw = perturbed[label].histogram, raw[label].histogram
            summaries.append(
                RankHistogramSummary(
                    source=self.source,
                    qc_flag=label,
                    n_obs=diag["n_obs"],
                    n_members=diag["n_ensemble"],
                    frac_below=diag["frac_below"],
                    frac_above=diag["frac_above"],
                    frac_outside_expected=diag["frac_outside_expected"],
                    frac_below_no_obs_error=diag_raw["frac_below"],
                    frac_above_no_obs_error=diag_raw["frac_above"],
                    innovation_mean=diag["innovation_mean"],
                    rmse=diag["rmse"],
                    mean_spread=diag["mean_spread"],
                    spread_skill=diag["spread_skill"],
                )
            )
            counts.append(
                pd.DataFrame(
                    {
                        "source": self.source,
                        "qc_flag": label,
                        "rank": np.arange(len(hist)),
                        "count": hist,
                        "count_no_obs_error": hist_raw,
                    }
                )
            )
            self._plot(hist, hist_raw, diag, label)

        summary = pd.DataFrame(
            [{f.name: getattr(s, f.name) for f in fields(RankHistogramSummary)} for s in summaries]
        )
        counts_df = pd.concat(counts, ignore_index=True)
        summary.to_csv(self.output_dir / f"summary_{self.source}.csv", index=False)
        counts_df.to_csv(self.output_dir / f"counts_{self.source}.csv", index=False)
        return {"output_dir": self.output_dir, "summary": summary, "counts": counts_df}

    def _plot(self, hist: np.ndarray, hist_raw: np.ndarray, diag: dict, label: str) -> Path:
        qc = "all QC flags" if label == "all" else f"qc_flag {label}"
        fig = plot_rank_histogram(
            hist,
            diag,
            title_suffix=f"{self.instrument}.{self.quantity}, {self.source}, {qc}",
            hist_no_obs_error=hist_raw,
        )
        path = self.output_dir / f"rank_histogram_{self.source}_qc{label}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")
        return path
