"""Forecast skill as a function of lead time.

Pooling every observation together conflates a genuinely skillful model with one
that simply had a recent analysis. Stratifying the departures by how long the
forecast had been running separates the two.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wrf_ensembly.console import logger
from wrf_ensembly.experiment.experiment import Experiment
from wrf_ensembly.validation.lead_time import assign_forecast_lead, bin_by_lead


@dataclass
class LeadTimeSkillRow:
    """Departure statistics for one lead-time bin."""

    lead_bin_hours: float
    """Left edge of the bin, in hours since the owning cycle started."""
    n: int
    omb_bias: float
    omb_rms: float
    omb_correlation: float
    n_oma: int
    """Observations in the bin that also have an analysis."""
    oma_bias: float
    oma_rms: float
    oma_correlation: float


class LeadTimeSkillAnalysis:
    """
    Scores O-B (and O-A where available) against forecast lead time.

    Lead time is reconstructed from the cycle definitions rather than read back
    from the database - see `validation.lead_time.assign_forecast_lead`.

    Note that an analysis only exists for the cycles where the filter ran, so the
    O-A curve typically covers only the shortest lead bins. The per-bin `n_oma`
    count makes that visible.
    """

    def __init__(
        self,
        experiment: Experiment,
        instrument: str,
        quantity: str,
        bin_hours: float = 1.0,
    ):
        """
        Args:
            experiment: The experiment to analyse.
            instrument: The instrument name.
            quantity: The observed quantity.
            bin_hours: Lead-time bin width in hours.
        """
        self.exp = experiment
        self.instrument = instrument
        self.quantity = quantity
        self.bin_hours = bin_hours
        self.output_dir = (
            experiment.paths.data_validation
            / "lead_time_skill"
            / instrument
            / quantity
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach the lead time and the departures to each observation.

        Observations that fall outside every cycle's forecast window are dropped;
        a non-zero count usually means the database holds observations from
        outside the experiment period.

        Args:
            df: Observations with time, value and model_forecast columns.

        Returns:
            The frame with cycle_index, lead_hours, lead_bin, omb and (where
            available) oma columns added.
        """
        lead = assign_forecast_lead(
            df["time"],
            self.exp.cycles,
            self.exp.cfg.validation.prefer_extended_forecast,
        )
        df = df.join(lead)

        outside = df["cycle_index"].isna()
        if outside.any():
            logger.warning(
                f"{int(outside.sum())} observation(s) fall outside every forecast "
                "window and will be ignored"
            )
            df = df[~outside]

        df = df.copy()
        df["lead_bin"] = bin_by_lead(df["lead_hours"], self.bin_hours)
        df["omb"] = df["value"] - df["model_forecast"]
        if "model_analysis" in df.columns:
            df["oma"] = df["value"] - df["model_analysis"]
        else:
            df["oma"] = np.nan

        return df

    def compute_skill(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce the prepared departures to one row per lead-time bin.

        Args:
            df: Output of `prepare`.

        Returns:
            Frame of `LeadTimeSkillRow` records, ordered by lead.
        """
        rows: list[LeadTimeSkillRow] = []

        for lead_bin, group in df.groupby("lead_bin", sort=True):
            omb = group["omb"].to_numpy(dtype=float)
            oma = group["oma"].to_numpy(dtype=float)
            value = group["value"].to_numpy(dtype=float)
            forecast = group["model_forecast"].to_numpy(dtype=float)
            analysis = (
                group["model_analysis"].to_numpy(dtype=float)
                if "model_analysis" in group.columns
                else np.full(len(group), np.nan)
            )

            rows.append(
                LeadTimeSkillRow(
                    lead_bin_hours=float(lead_bin),
                    n=int(np.isfinite(omb).sum()),
                    omb_bias=_safe_mean(omb),
                    omb_rms=_rms(omb),
                    omb_correlation=_correlation(value, forecast),
                    n_oma=int(np.isfinite(oma).sum()),
                    oma_bias=_safe_mean(oma),
                    oma_rms=_rms(oma),
                    oma_correlation=_correlation(value, analysis),
                )
            )

        columns = [f.name for f in fields(LeadTimeSkillRow)]
        return pd.DataFrame(
            [{c: getattr(r, c) for c in columns} for r in rows], columns=columns
        )

    def plot_skill(self, skill: pd.DataFrame) -> Path:
        """Draw bias, RMS, correlation and count against lead time."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        lead = skill["lead_bin_hours"]

        ax = axes[0, 0]
        ax.plot(lead, skill["omb_bias"], "o-", label="O-B")
        ax.plot(lead, skill["oma_bias"], "s--", label="O-A")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Bias")
        ax.set_title("Departure bias")

        ax = axes[0, 1]
        ax.plot(lead, skill["omb_rms"], "o-", label="O-B")
        ax.plot(lead, skill["oma_rms"], "s--", label="O-A")
        ax.set_ylabel("RMS")
        ax.set_title("Departure RMS")

        ax = axes[1, 0]
        ax.plot(lead, skill["omb_correlation"], "o-", label="O-B")
        ax.plot(lead, skill["oma_correlation"], "s--", label="O-A")
        ax.set_ylabel("Correlation with observations")
        ax.set_title("Correlation")

        ax = axes[1, 1]
        width = self.bin_hours * 0.4
        ax.bar(lead - width / 2, skill["n"], width=width, label="O-B")
        ax.bar(lead + width / 2, skill["n_oma"], width=width, label="O-A")
        ax.set_ylabel("Observations")
        ax.set_title("Sample size")

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        for ax in axes[1, :]:
            ax.set_xlabel("Forecast lead time (hours)")

        fig.suptitle(
            f"{self.exp.cfg.metadata.name} - forecast skill vs lead time for "
            f"{self.instrument}.{self.quantity}",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()

        path = self.output_dir / "skill_by_lead.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def run(self, df: pd.DataFrame) -> dict:
        """
        Score the departures by lead time and save the table, plot and raw frame.

        Args:
            df: Observations for one instrument-quantity pair.

        Returns:
            Dict of output paths.
        """
        results: dict = {
            "instrument": self.instrument,
            "quantity": self.quantity,
            "output_dir": self.output_dir,
        }

        prepared = self.prepare(df)
        if prepared.empty:
            logger.warning("No observations left after lead-time assignment")
            return results

        skill = self.compute_skill(prepared)
        skill_path = self.output_dir / "skill_by_lead.csv"
        skill.to_csv(skill_path, index=False)
        results["skill_file"] = skill_path
        results["skill"] = skill

        departures_path = self.output_dir / "departures_with_lead.parquet"
        prepared.drop(columns=["lead"]).to_parquet(departures_path, index=False)
        results["departures_file"] = departures_path

        results["skill_plot"] = self.plot_skill(skill)

        return results


def _rms(x) -> float:
    """Root-mean-square of the finite entries. NaN when nothing is finite."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(x**2)))


def _safe_mean(x) -> float:
    """Mean of the finite entries. NaN when nothing is finite."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.mean(x))


def _correlation(a, b) -> float:
    """Pearson correlation over pairwise-finite entries, NaN when undefined."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
