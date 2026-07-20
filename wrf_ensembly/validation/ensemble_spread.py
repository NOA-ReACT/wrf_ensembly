"""Domain-mean ensemble spread across the whole experiment.

Shows whether the perturbation design sustains a healthy ensemble spread over the
run, or whether the ensemble collapses.
"""

from dataclasses import dataclass, fields
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from wrf_ensembly.console import logger
from wrf_ensembly.experiment.experiment import Experiment
from wrf_ensembly.validation.lead_time import assign_forecast_lead

TIME_DIM = "t"
"""Time dimension of the postprocessed forecast files."""

NON_SPATIAL_DIMS = {TIME_DIM, "time", "Time", "member"}
"""Dimensions never averaged over when reducing a field to a domain mean."""


@dataclass
class EnsembleSpreadStatistics:
    """Summary of one variable's ensemble spread over the whole experiment."""

    variable: str
    n_times: int
    mean_value: float
    """Time mean of the domain-mean ensemble mean."""
    mean_spread: float
    """Time mean of the domain-mean ensemble spread."""
    relative_spread: float
    """`mean_spread` as a percentage of `mean_value`."""
    min_spread: float
    max_spread: float


class EnsembleSpreadAnalysis:
    """
    Tracks the domain-mean ensemble mean and spread across every cycle.

    Reads the per-cycle `forecast_mean` / `forecast_sd` files written by
    `postprocess run` and concatenates them into one continuous series.

    Note that the spread reported is the domain mean of the pointwise ensemble
    standard deviation, which is not the same as the standard deviation of the
    domain mean. The former measures how much the members disagree locally, which
    is what the perturbation design controls.
    """

    def __init__(self, experiment: Experiment, variables: list[str]):
        """
        Args:
            experiment: The experiment to analyse.
            variables: Names of the variables to track.
        """
        self.exp = experiment
        self.variables = list(variables)
        self.output_dir = experiment.paths.data_validation / "ensemble_spread"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _domain_mean(self, da: xr.DataArray) -> np.ndarray:
        """
        Reduce a field to one value per timestep by averaging over the domain.

        Every dimension except time is averaged over, so a three-dimensional field
        is also collapsed vertically.
        """
        reduce_dims = [d for d in da.dims if d not in NON_SPATIAL_DIMS]
        if reduce_dims:
            da = da.mean(dim=reduce_dims)
        return da.values

    def load_series(self) -> pd.DataFrame:
        """
        Concatenate the per-cycle forecasts into one continuous domain-mean series.

        Returns:
            Frame indexed by time with `{variable}_mean` and `{variable}_sd`
            columns. Empty if no cycle had usable output.
        """
        frames: list[pd.DataFrame] = []
        missing_vars: set[str] = set()

        for cycle in self.exp.cycles:
            forecast_dir = self.exp.paths.forecast_path(cycle.index)
            mean_file = forecast_dir / f"forecast_mean_cycle_{cycle.index:03d}.nc"
            sd_file = forecast_dir / f"forecast_sd_cycle_{cycle.index:03d}.nc"

            if not mean_file.exists() or not sd_file.exists():
                logger.warning(
                    f"Cycle {cycle.index}: missing forecast mean/sd files, skipping"
                )
                continue

            with (
                xr.open_dataset(mean_file) as mean_ds,
                xr.open_dataset(sd_file) as sd_ds,
            ):
                times = pd.to_datetime(mean_ds[TIME_DIM].values)
                data: dict[str, np.ndarray] = {}

                for variable in self.variables:
                    if variable not in mean_ds or variable not in sd_ds:
                        missing_vars.add(variable)
                        continue
                    data[f"{variable}_mean"] = self._domain_mean(mean_ds[variable])
                    data[f"{variable}_sd"] = self._domain_mean(sd_ds[variable])

                if not data:
                    continue

                frame = pd.DataFrame(data, index=times)
                frame["cycle_index"] = cycle.index
                frames.append(frame)

        for variable in sorted(missing_vars):
            logger.warning(f"Variable '{variable}' not found in some forecast files")

        if not frames:
            return pd.DataFrame()

        series = pd.concat(frames)
        series.index.name = "time"
        return self._resolve_overlaps(series)

    def _resolve_overlaps(self, series: pd.DataFrame) -> pd.DataFrame:
        """
        Keep one frame per timestamp when cycles overlap.

        With a forecast extension, consecutive cycles both produce output for the
        same timestamp. Rather than blindly keeping the first, defer to the same
        rule the model interpolation uses, so this series and the O-B statistics
        describe the same forecasts.
        """
        if not series.index.duplicated().any():
            return series.sort_index().drop(columns="cycle_index")

        lead = assign_forecast_lead(
            pd.Series(series.index, index=series.index),
            self.exp.cycles,
            self.exp.cfg.validation.prefer_extended_forecast,
        )
        chosen = series["cycle_index"].to_numpy() == lead["cycle_index"].to_numpy()
        resolved = series[chosen]

        # A timestamp whose selected cycle produced no output would drop out
        # entirely, so fall back to the first available frame for those.
        dropped = series.index.difference(resolved.index)
        if len(dropped):
            logger.warning(
                f"{len(dropped)} timestamp(s) had no frame from the preferred cycle, "
                "falling back to the earliest available"
            )
            fallback = series.loc[dropped]
            fallback = fallback[~fallback.index.duplicated(keep="first")]
            resolved = pd.concat([resolved, fallback])

        return resolved.sort_index().drop(columns="cycle_index")

    def compute_statistics(
        self, series: pd.DataFrame
    ) -> list[EnsembleSpreadStatistics]:
        """Summarise each variable's spread over the whole series."""
        stats = []
        for variable in self.variables:
            mean_col, sd_col = f"{variable}_mean", f"{variable}_sd"
            if mean_col not in series or sd_col not in series:
                continue

            values = series[mean_col].dropna()
            spreads = series[sd_col].dropna()
            if values.empty or spreads.empty:
                continue

            mean_value = float(values.mean())
            mean_spread = float(spreads.mean())
            relative = (
                100.0 * mean_spread / mean_value if mean_value != 0 else float("nan")
            )

            stats.append(
                EnsembleSpreadStatistics(
                    variable=variable,
                    n_times=len(values),
                    mean_value=mean_value,
                    mean_spread=mean_spread,
                    relative_spread=relative,
                    min_spread=float(spreads.min()),
                    max_spread=float(spreads.max()),
                )
            )
        return stats

    def plot_timeseries(self, series: pd.DataFrame) -> Path:
        """Draw one panel per variable, with the mean, the spread, and a mean+-sd band."""
        plotted = [
            v
            for v in self.variables
            if f"{v}_mean" in series and f"{v}_sd" in series
        ]
        fig, axes = plt.subplots(
            len(plotted),
            1,
            figsize=(12, 3.5 * len(plotted)),
            sharex=True,
            squeeze=False,
        )

        for ax, variable in zip(axes[:, 0], plotted):
            mean = series[f"{variable}_mean"]
            sd = series[f"{variable}_sd"]

            ax.fill_between(
                series.index,
                mean - sd,
                mean + sd,
                alpha=0.2,
                lw=0,
                label="mean ± spread",
            )
            ax.plot(series.index, mean, lw=1.4, label="ensemble mean")
            ax.plot(series.index, sd, lw=1.2, ls="--", label="ensemble spread (σ)")

            for cycle in self.exp.cycles:
                ax.axvline(
                    pd.Timestamp(cycle.end).tz_localize(None),
                    color="0.7",
                    ls=":",
                    lw=0.8,
                )

            ax.set_ylabel(variable)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="upper left")

        axes[-1, 0].set_xlabel("Time (UTC)")
        axes[-1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

        fig.suptitle(
            f"{self.exp.cfg.metadata.name} - domain-mean ensemble spread "
            "(dotted lines mark cycle boundaries)",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()

        path = self.output_dir / "timeseries.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def run(self) -> dict:
        """
        Build the series, save it, and produce the summary statistics and plot.

        Returns:
            Dict of output paths and the computed statistics. Contains only
            `output_dir` when no forecast output could be read.
        """
        results: dict = {"output_dir": self.output_dir}

        series = self.load_series()
        if series.empty:
            logger.error("No forecast mean/sd files could be read")
            logger.error("Ensure `postprocess run` has been completed.")
            return results

        series_path = self.output_dir / "timeseries.csv"
        series.to_csv(series_path)
        results["timeseries_file"] = series_path
        logger.info(f"Saved {series_path}")

        stats = self.compute_statistics(series)
        if stats:
            columns = [f.name for f in fields(EnsembleSpreadStatistics)]
            stats_df = pd.DataFrame(
                [{c: getattr(s, c) for c in columns} for s in stats], columns=columns
            )
            stats_path = self.output_dir / "statistics.csv"
            stats_df.to_csv(stats_path, index=False)
            results["statistics_file"] = stats_path
            results["statistics"] = stats
            logger.info(f"Saved {stats_path}")

            for s in stats:
                logger.info(
                    f"{s.variable}: mean {s.mean_value:.4g}, "
                    f"spread {s.mean_spread:.4g} "
                    f"(relative spread {s.relative_spread:.1f}%)"
                )

        results["timeseries_plot"] = self.plot_timeseries(series)
        logger.info(f"Saved {results['timeseries_plot']}")

        return results
