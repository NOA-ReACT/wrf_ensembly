"""First departures (O-B) analysis for model validation."""

from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from wrf_ensembly.config import FirstDeparturesRegimeConfig
from wrf_ensembly.console import logger
from wrf_ensembly.experiment.experiment import Experiment


@dataclass
class FirstDeparturesStatistics:
    """Statistics for first departures analysis."""

    count: int
    bias: float
    std: float
    rmse: float
    min: float
    max: float
    q25: float
    median: float
    q75: float
    norm_bias: float
    norm_std: float
    spread_to_obserr_ratio: float
    r_est: float


class FirstDeparturesAnalysis:
    """Analyzes first departures (O-B) statistics.

    This class computes and visualizes the difference between observations
    and background model values (O - B), providing insights into model bias
    and uncertainty across different regimes, spatial regions, and time periods.
    """

    def __init__(
        self,
        experiment: Experiment,
        instrument: str,
        quantity: str,
        proj: ccrs.Projection | None = None,
    ):
        """Initialize first departures analysis for a specific instrument-quantity pair.

        Args:
            experiment: The experiment to analyze
            instrument: The instrument name (e.g., 'MODIS', 'VIIRS')
            quantity: The observation quantity to analyze (e.g., 'AOD_550nm')
            proj: Cartopy projection for map plots. If None, uses PlateCarree.
        """
        self.exp = experiment
        self.instrument = instrument
        self.quantity = quantity
        self.output_dir = (
            experiment.paths.data
            / "validation"
            / "first_departures"
            / instrument
            / quantity
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load regime config if available
        self.regime_config = self._get_regime_config()

        # Set map projection
        self.map_proj = proj or ccrs.PlateCarree()

    def _get_regime_config(self) -> FirstDeparturesRegimeConfig | None:
        """Get regime configuration for this instrument-quantity pair if it exists."""
        for regime in self.exp.cfg.validation.first_departures.regimes:
            if (
                regime.instrument == self.instrument
                and regime.quantity == self.quantity
            ):
                return regime
        return None

    def run(self, df: pd.DataFrame) -> dict:
        """Run the complete first departures analysis.

        Args:
            df: DataFrame with observations including 'value', 'model_forecast', and 'departure' columns

        Returns:
            Dictionary containing paths to generated outputs and computed statistics
        """
        logger.info(
            f"Running first departures analysis for {self.instrument}.{self.quantity}"
        )

        # Compute O-B if not already present
        if "departure" not in df.columns:
            df["departure"] = df["value"] - df["model_forecast"]

        if self.exp.cfg.validation.first_departures.analysis_bbox is not None:
            bbox = self.exp.cfg.validation.first_departures.analysis_bbox
            logger.debug(f"Only including region {bbox}")
            df = df[
                df["latitude"].between(bbox[0], bbox[2])
                & df["longitude"].between(bbox[1], bbox[3])
            ]

        # Remove any regions marked as excluded (cfg option `exclude_bboxes`)
        # After adding the new column so we avoid a `.copy()` and any 'assigning a view' problems
        for bbox in self.exp.cfg.validation.first_departures.excluded_bboxes:
            logger.debug(f"Excluding region {bbox}")
            df = df[
                ~df["latitude"].between(bbox[0], bbox[2])
                | ~df["longitude"].between(bbox[1], bbox[3])
            ]

        results = {
            "instrument": self.instrument,
            "quantity": self.quantity,
            "output_dir": self.output_dir,
        }

        # Compute overall statistics
        fp_stats = self.compute_statistics(df)
        results["statistics"] = fp_stats

        # Save statistics in a CSV file
        stats_file = self._save_statistics(fp_stats)
        results["statistics_file"] = stats_file

        # Generate plots if configured
        results["histogram"] = self.plot_histogram(df)
        results["timeseries"] = self.plot_timeseries(df)
        results["spatial_maps"] = self.plot_spatial_maps(df)

        # Regime analysis if configured
        if self.regime_config is not None:
            regime_stats, regime_plots = self.analyze_by_regime(df)
            results["regime_statistics"] = regime_stats
            results["regime_plots"] = regime_plots

        logger.info(
            f"First departures analysis complete. Results saved to {self.output_dir}"
        )
        return results

    @staticmethod
    def _compute_statistics(df: pd.DataFrame) -> FirstDeparturesStatistics:
        """Compute the full set of O-B statistics for a DataFrame.

        Args:
            df: DataFrame with 'departure', 'model_forecast_spread' and
                'value_uncertainty' columns

        Returns:
            Statistics dataclass
        """
        departure = df["departure"].dropna()
        d = df.dropna(
            subset=["departure", "model_forecast_spread", "value_uncertainty"]
        )
        total_var = d["model_forecast_spread"] ** 2 + d["value_uncertainty"] ** 2
        valid = total_var > 0
        normalised = d.loc[valid, "departure"] / np.sqrt(total_var[valid])

        return FirstDeparturesStatistics(
            count=len(departure),
            bias=float(departure.mean()),
            std=float(departure.std()),
            rmse=float(np.sqrt((departure**2).mean())),
            min=float(departure.min()),
            max=float(departure.max()),
            q25=float(departure.quantile(0.25)),
            median=float(departure.median()),
            q75=float(departure.quantile(0.75)),
            norm_bias=float(normalised.mean()),
            norm_std=float(normalised.std()),
            spread_to_obserr_ratio=float(
                (
                    d.loc[valid, "model_forecast_spread"]
                    / d.loc[valid, "value_uncertainty"]
                ).mean()
            ),
            r_est=d["departure"].var() - (d["model_forecast_spread"] ** 2).mean(),
        )

    def compute_statistics(self, df: pd.DataFrame) -> FirstDeparturesStatistics:
        """Compute overall O-B statistics.

        Args:
            df: DataFrame with 'departure' column

        Returns:
            Statistics dataclass
        """
        fp_stats = self._compute_statistics(df)

        logger.info(f"Statistics for {self.instrument}.{self.quantity}:")
        logger.info(f"  Count: {fp_stats.count}")
        logger.info(f"  Bias: {fp_stats.bias:.6f}")
        logger.info(f"  Std: {fp_stats.std:.6f}")
        logger.info(f"  RMSE: {fp_stats.rmse:.6f}")
        logger.info(f"  Normalised innovation mean: {fp_stats.norm_bias:+.3f}")
        logger.info(f"  Normalised innovation std: {fp_stats.norm_std:.3f}")
        logger.info(f"  Mean σ_B / σ_o ratio: {fp_stats.spread_to_obserr_ratio:.3f}")

        return fp_stats

    def _save_statistics(self, stats: FirstDeparturesStatistics) -> Path:
        """Save statistics to a CSV file.

        Args:
            stats: Statistics to save

        Returns:
            Path to the saved file
        """
        stats_file = self.output_dir / "statistics.csv"
        stats_dict = {
            "instrument": self.instrument,
            "quantity": self.quantity,
            **vars(stats),
        }
        df = pd.DataFrame([stats_dict])
        df.to_csv(stats_file, index=False)
        logger.info(f"Saved statistics to {stats_file}")
        return stats_file

    def plot_histogram(self, df: pd.DataFrame) -> Path:
        """Generate histogram of first departures (O-B) values.

        Args:
            df: DataFrame with 'departure' column

        Returns:
            Path to the saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        departure = df["departure"].dropna()
        n, bins, patches = ax.hist(
            departure,
            bins=50,
            density=True,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )

        # Add KDE
        kde = stats.gaussian_kde(departure)
        x_range = np.linspace(departure.min(), departure.max(), 200)
        ax.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")

        ax.set_xlabel("First Departure (O - B)")
        ax.set_ylabel("Density")
        ax.set_title(
            f"{self.exp.cfg.metadata.name} - Histogram of First Departures for {self.instrument}.{self.quantity}"
        )
        ax.axvline(0, color="red", linestyle="--", alpha=0.5, label="Zero bias")
        ax.legend()
        ax.grid(True, alpha=0.3)

        output_file = self.output_dir / "histogram.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved histogram to {output_file}")
        return output_file

    def plot_timeseries(self, df: pd.DataFrame) -> Path:
        """Generate time series of first departures mean and standard deviation.

        Args:
            df: DataFrame with 'time' and 'departure' columns

        Returns:
            Path to the saved plot
        """
        # Resample to hourly
        df_hourly = (
            df.set_index("time")
            .resample("1h")["departure"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Mean and std
        axes[0].plot(df_hourly["time"], df_hourly["mean"], label="Mean", color="blue")
        axes[0].fill_between(
            df_hourly["time"],
            df_hourly["mean"] - df_hourly["std"],
            df_hourly["mean"] + df_hourly["std"],
            alpha=0.3,
            color="blue",
            label="±1 Std Dev",
        )
        axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[0].set_ylabel("First Departure (O-B)")
        axes[0].set_title(
            f"{self.exp.cfg.metadata.name} - Time Series of First Departures for {self.instrument}.{self.quantity}"
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Count
        axes[1].bar(
            df_hourly["time"], df_hourly["count"], width=0.04, color="gray", alpha=0.6
        )
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Observation Count")
        axes[1].grid(True, alpha=0.3)

        output_file = self.output_dir / "timeseries.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved time series to {output_file}")
        return output_file

    def plot_spatial_maps(self, df: pd.DataFrame) -> Path:
        """Generate spatial maps of first departures statistics.

        Args:
            df: DataFrame with 'latitude', 'longitude', and 'departure' columns

        Returns:
            Path to the saved plot
        """
        # Get spatial resolution from config, default to 1.0
        resolution = (
            self.regime_config.spatial_resolution if self.regime_config else 1.0
        )

        # Bin spatially
        df = df.copy()
        df["lat_bin"] = (df["latitude"] // resolution) * resolution
        df["lon_bin"] = (df["longitude"] // resolution) * resolution

        spatial_stats = (
            df.groupby(["lat_bin", "lon_bin"])["departure"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .dropna()
        )

        # Create grids
        mean_grid = spatial_stats.pivot(
            index="lat_bin", columns="lon_bin", values="mean"
        )
        std_grid = spatial_stats.pivot(index="lat_bin", columns="lon_bin", values="std")
        count_grid = spatial_stats.pivot(
            index="lat_bin", columns="lon_bin", values="count"
        )

        # Plot
        fig, axes = plt.subplots(
            3, 1, figsize=(12, 10), subplot_kw={"projection": self.map_proj}
        )

        titles = [
            "Bias (Mean First Departure)",
            "Standard Deviation",
            "Observation Count",
        ]
        grids = [mean_grid, std_grid, count_grid]
        cmaps = ["RdBu_r", "viridis", "YlOrRd"]

        for ax, grid, title, cmap in zip(axes, grids, titles, cmaps):
            ax.coastlines()
            ax.gridlines(alpha=0.3)

            # Use diverging colormap centered at 0 for bias
            if "Bias" in title:
                cfg_override = self.exp.cfg.validation.first_departures.bias_map_colorbar_ranges.get(
                    f"{self.instrument}.{self.quantity}", None
                )
                if cfg_override is not None:
                    vmin, vmax = cfg_override
                else:
                    vmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)))
                    vmin = -vmax
            else:
                vmin, vmax = None, None

            pcm = ax.pcolormesh(
                grid.columns,
                grid.index,
                grid.values,
                cmap=cmap,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
                transform=ccrs.PlateCarree(),
            )
            ax.set_title(
                f"{self.exp.cfg.metadata.name} - {title} - {self.instrument}.{self.quantity}"
            )
            fig.colorbar(pcm, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)

        plt.tight_layout()

        output_file = self.output_dir / "spatial_maps.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved spatial maps to {output_file}")
        return output_file

    def analyze_by_regime(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Path] | tuple[None, None]:
        """Analyze first departures by regime.

        Args:
            df: DataFrame with 'model_forecast' and 'departure' columns

        Returns:
            Tuple of (statistics DataFrame, path to saved plot)
        """
        if self.regime_config is None:
            logger.warning(
                f"No regime configuration for {self.instrument}.{self.quantity}, skipping regime analysis"
            )
            return None, None

        logger.info(f"Analyzing by regime for {self.instrument}.{self.quantity}")

        # Create regime bins
        df = df.copy()
        df["regime"] = pd.cut(
            0.5 * (df["model_forecast"] + df["value"]),
            bins=self.regime_config.bins,
            labels=self.regime_config.labels,
        )

        # Compute the full set of statistics per regime
        regime_stats = pd.DataFrame(
            {
                regime: vars(self._compute_statistics(group))
                for regime, group in df.groupby("regime", observed=True)
            }
        ).T
        regime_stats.index.name = "regime"

        # Save statistics
        stats_file = self.output_dir / "regime_statistics.csv"
        regime_stats.to_csv(stats_file)
        logger.info(f"Saved regime statistics to {stats_file}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Box plot
        df.boxplot(column="departure", by="regime", ax=axes[0], showfliers=False)
        axes[0].set_title(
            f"{self.exp.cfg.metadata.name} - First Departure Distribution by Regime - {self.instrument}.{self.quantity}"
        )
        axes[0].set_xlabel("Regime")
        axes[0].set_ylabel("First Departure (O - B)")
        axes[0].axhline(0, color="red", linestyle="--", alpha=0.5)
        plt.sca(axes[0])
        plt.xticks(rotation=45, ha="right")

        # Std by regime
        regime_stats["std"].plot(kind="bar", ax=axes[1], color="steelblue")
        axes[1].set_title(
            f"{self.exp.cfg.metadata.name} - Standard Deviation by Regime - {self.instrument}.{self.quantity}"
        )
        axes[1].set_ylabel("Std(First Departure)")
        axes[1].set_xlabel("Regime")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        output_file = self.output_dir / "regime_analysis.png"
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Saved regime analysis to {output_file}")
        return regime_stats, output_file
