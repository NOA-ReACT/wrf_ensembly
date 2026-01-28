from pathlib import Path
from typing import Literal

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.figure import Figure

from wrf_ensembly.config import PlotVariableConfig
from wrf_ensembly.diagnostics import compute_rank_histogram


def plot_forecast_vs_analysis(
    forecast_ds: xr.Dataset,
    analysis_ds: xr.Dataset,
    variable_cfg: PlotVariableConfig,
    arrangement: str = "horizontal",
    proj: ccrs.Projection | None = None,
    experiment_name: str = "",
    cycle: int = 0,
    time_str: str = "",
    plot_type: Literal["mean", "spread"] = "mean",
) -> Figure:
    """
    Create a three-panel comparison plot: forecast, analysis, and analysis - forecast.

    Args:
        forecast_ds: xarray Dataset with the forecast data.
        analysis_ds: xarray Dataset with the analysis data.
        variable_cfg: Configuration for the variable to plot.
        arrangement: Panel layout, either "horizontal" (1x3) or "vertical" (3x1).
        proj: Cartopy projection for the map panels. If None, uses PlateCarree.
        experiment_name: Name of the experiment for the figure title.
        cycle: Cycle index for the figure title.
        time_str: Time string for the figure title.
        plot_type: Type of plot to create, either "mean" or "spread".

    Returns:
        A matplotlib Figure with three panels.
    """

    var_name = variable_cfg.name
    forecast_var = forecast_ds[var_name]
    analysis_var = analysis_ds[var_name]

    spatial_dims = {"south_north", "west_east", "x", "y"}
    time_dims = {"Time", "time", "t"}

    # Select vertical level if specified
    if variable_cfg.level is not None:
        level_dim = None
        for dim in forecast_var.dims:
            if dim not in spatial_dims and dim not in time_dims:
                level_dim = dim
                break
        if level_dim is not None:
            forecast_var = forecast_var.isel({level_dim: variable_cfg.level})
            analysis_var = analysis_var.isel({level_dim: variable_cfg.level})

    # Squeeze out time dimension if present (take first timestep)
    for dim in time_dims:
        if dim in forecast_var.dims:
            forecast_var = forecast_var.isel({dim: 0})
        if dim in analysis_var.dims:
            analysis_var = analysis_var.isel({dim: 0})

    # Use projected x/y coordinates for pcolormesh (postprocessed files store these
    # as 1D coordinates in the native map projection)
    has_xy = "x" in forecast_ds.coords and "y" in forecast_ds.coords

    diff_var = analysis_var - forecast_var

    map_proj = proj or ccrs.PlateCarree()

    if arrangement == "vertical":
        nrows, ncols = 3, 1
        figsize = (8, 18)
    else:
        nrows, ncols = 1, 3
        figsize = (22, 6)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        subplot_kw={"projection": map_proj},
        figsize=figsize,
    )

    # Configure panels based on plot type
    if plot_type == "spread":
        forecast_title = "Forecast Spread"
        analysis_title = "Analysis Spread"
        diff_title = "Analysis Spread - Forecast Spread"
        main_cmap = variable_cfg.spread_cmap
        main_vmin = variable_cfg.spread_vmin
        main_vmax = variable_cfg.spread_vmax
    else:  # mean
        forecast_title = "Forecast"
        analysis_title = "Analysis"
        diff_title = "Analysis - Forecast"
        main_cmap = variable_cfg.cmap
        main_vmin = variable_cfg.vmin
        main_vmax = variable_cfg.vmax

    # If vmin/vmax not specified, use combined range from both forecast and analysis
    # so they have the same color scale
    if main_vmin is None or main_vmax is None:
        combined_min = min(float(forecast_var.min()), float(analysis_var.min()))
        combined_max = max(float(forecast_var.max()), float(analysis_var.max()))
        if main_vmin is None:
            main_vmin = combined_min
        if main_vmax is None:
            main_vmax = combined_max

    panels = [
        (forecast_title, forecast_var, main_cmap, main_vmin, main_vmax),
        (analysis_title, analysis_var, main_cmap, main_vmin, main_vmax),
        (
            diff_title,
            diff_var,
            variable_cfg.diff_cmap,
            variable_cfg.diff_vmin,
            variable_cfg.diff_vmax,
        ),
    ]

    for ax, (title, data, cmap, vmin, vmax) in zip(axes, panels):
        if has_xy:
            mesh = ax.pcolormesh(
                forecast_ds["x"].values,
                forecast_ds["y"].values,
                data.values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
            )
        else:
            mesh = ax.pcolormesh(
                data.values,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

        ax.coastlines()

        level_str = (
            f" (level {variable_cfg.level})" if variable_cfg.level is not None else ""
        )
        ax.set_title(f"{title}: {var_name}{level_str}")

        fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)

    suptitle_parts = []
    if experiment_name:
        suptitle_parts.append(experiment_name)
    suptitle_parts.append(f"Cycle {cycle}")
    if time_str:
        suptitle_parts.append(time_str)
    fig.suptitle(" | ".join(suptitle_parts), fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig


def generate_filter_stats_plots(
    df: pd.DataFrame,
    output_dir: Path,
    label: str,
    dpi: int = 150,
    logger=None,
):
    """
    Generate all filter diagnostic plots for a given DataFrame subset.

    Creates scatter diagnostics and rank histograms, saving them to the output directory.

    Args:
        df: DataFrame with observation diagnostics.
        output_dir: Directory to save plots.
        label: Label for plot titles and log messages.
        dpi: Resolution for saved plots.
        logger: Logger instance for info/warning messages. If None, prints are silent.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def log_info(msg):
        if logger:
            logger.info(msg)

    def log_warning(msg):
        if logger:
            logger.warning(msg)

    # Clean data: replace inf with nan
    df = df.replace([float("inf"), float("-inf")], float("nan"))

    # For scatter plots, we want ALL observations (including rejected ones)
    # to diagnose what went wrong. Only drop rows where observation itself is invalid.
    df_scatter = df.dropna(subset=["obs", "obs_variance"])

    if len(df_scatter) == 0:
        log_warning(f"No valid observations for {label}, skipping")
        return

    # Scatter diagnostics (shows all obs, colored by QC flag)
    fig = plot_filter_scatter_diagnostics(df_scatter, title_suffix=label)
    fig.savefig(output_dir / "diagnostic_scatter.png", dpi=dpi, bbox_inches="tight")
    log_info(f"Saved {output_dir / 'diagnostic_scatter.png'}")
    plt.close(fig)

    # For rank histograms, only use observations with valid prior/posterior
    key_cols = [
        "obs",
        "obs_variance",
        "prior_mean",
        "prior_spread",
        "posterior_mean",
        "posterior_spread",
    ]
    available_key_cols = [c for c in key_cols if c in df.columns]
    df_valid = df.dropna(subset=available_key_cols)

    if len(df_valid) == 0:
        log_warning(
            f"No observations with valid prior/posterior for {label}, "
            "skipping rank histogram plots"
        )
        return

    # Rank histograms (use only QC-passed observations for meaningful statistics)
    qc_passed = df_valid.query("dart_qc == 0")
    if len(qc_passed) < 10:
        log_warning(
            f"Only {len(qc_passed)} QC-passed observations for {label}, "
            "skipping rank histograms"
        )
        return

    hist, _, diag = compute_rank_histogram(qc_passed, use_posterior=False)
    fig = plot_rank_histogram(hist, diag, title_suffix=f"(Prior) {label}")
    fig.savefig(output_dir / "rank_histogram_prior.png", dpi=dpi, bbox_inches="tight")
    log_info(f"Saved {output_dir / 'rank_histogram_prior.png'}")
    plt.close(fig)

    if "posterior_mean" in df.columns and "posterior_spread" in df.columns:
        hist, _, diag = compute_rank_histogram(qc_passed, use_posterior=True)
        fig = plot_rank_histogram(hist, diag, title_suffix=f"(Posterior) {label}")
        fig.savefig(
            output_dir / "rank_histogram_posterior.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        log_info(f"Saved {output_dir / 'rank_histogram_posterior.png'}")
        plt.close(fig)


def plot_filter_scatter_diagnostics(df: pd.DataFrame, title_suffix: str = "") -> Figure:
    """
    Create a 6-panel figure with scatter plots and diagnostics from DART filter output.

    Panels:
      (0,0) Prior mean vs observation (colored by dart_qc)
      (0,1) Prior spread vs observation error variance
      (1,0) Posterior mean vs observation (colored by dart_qc)
      (1,1) Posterior spread vs observation error variance
      (2,0) Correlation matrix heatmap
      (2,1) Innovation histogram

    Args:
        df: DataFrame with columns obs, obs_variance, prior_mean, prior_spread,
            posterior_mean, posterior_spread, dart_qc.
        title_suffix: Optional suffix for the figure title.

    Returns:
        A matplotlib Figure.
    """
    # QC flag labels
    QC_LABELS = {
        0: "OK",
        1: "Eval only",
        2: "Post fail",
        3: "Eval+post fail",
        4: "Fwd fail",
        5: "Ignored by config",
        6: "Bad obs",
        7: "Outlier",
        8: "Vert fail",
    }
    QC_LABELS = {key: f"{key} {value}" for key, value in QC_LABELS.items()}

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    qc_values = df["dart_qc"].values
    qc_unique = np.unique(qc_values)

    # Color scheme: green for QC=0, yellow/orange for partial use, red for rejected
    qc_colors = {
        0: "#2ecc71",  # green - OK
        1: "#f39c12",  # orange - evaluated only
        2: "#e67e22",  # darker orange
        3: "#e67e22",
        4: "#e74c3c",  # red - forward operator fail
        5: "#95a5a6",  # gray - ignored
        6: "#c0392b",  # dark red - bad obs
        7: "#e74c3c",  # red - outlier
        8: "#e74c3c",  # red - vertical fail
    }

    # (0,0) Prior mean vs obs
    ax = axes[0, 0]
    for qc_val in sorted(qc_unique, reverse=True):  # Plot OK (0) last so it's on top
        mask = qc_values == qc_val
        label = QC_LABELS.get(int(qc_val), f"QC={int(qc_val)}")
        color = qc_colors.get(int(qc_val), "#95a5a6")
        alpha = 0.7 if qc_val == 0 else 0.4
        zorder = 10 if qc_val == 0 else 5
        ax.scatter(
            df.loc[mask, "prior_mean"],
            df.loc[mask, "obs"],
            c=color,
            label=f"{label} (n={mask.sum()})",
            alpha=alpha,
            s=15 if qc_val == 0 else 10,
            zorder=zorder,
        )
    lims = _common_lims(df["prior_mean"], df["obs"])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1, zorder=1)
    ax.set_xlabel("Prior Mean")
    ax.set_ylabel("Observation")
    ax.set_title("Prior Mean vs Observation")
    ax.legend(fontsize=7, markerscale=1.5, loc="best")

    # (0,1) Prior spread vs obs variance
    ax = axes[0, 1]
    ax.scatter(df["obs_variance"], df["prior_spread"], alpha=0.5, s=10)
    lims = _common_lims(df["obs_variance"], df["prior_spread"])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlabel("Obs Variance")
    ax.set_ylabel("Prior Spread")
    ax.set_title("Prior Spread vs Obs Variance")

    # (1,0) Posterior mean vs obs
    ax = axes[1, 0]
    for qc_val in sorted(qc_unique, reverse=True):  # Plot OK (0) last so it's on top
        mask = qc_values == qc_val
        label = QC_LABELS.get(int(qc_val), f"QC={int(qc_val)}")
        color = qc_colors.get(int(qc_val), "#95a5a6")
        alpha = 0.7 if qc_val == 0 else 0.4
        zorder = 10 if qc_val == 0 else 5
        ax.scatter(
            df.loc[mask, "posterior_mean"],
            df.loc[mask, "obs"],
            c=color,
            label=f"{label} (n={mask.sum()})",
            alpha=alpha,
            s=15 if qc_val == 0 else 10,
            zorder=zorder,
        )
    lims = _common_lims(df["posterior_mean"], df["obs"])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1, zorder=1)
    ax.set_xlabel("Posterior Mean")
    ax.set_ylabel("Observation")
    ax.set_title("Posterior Mean vs Observation")
    ax.legend(fontsize=7, markerscale=1.5, loc="best")

    # (1,1) Posterior spread vs obs variance
    ax = axes[1, 1]
    ax.scatter(df["obs_variance"], df["posterior_spread"], alpha=0.5, s=10)
    lims = _common_lims(df["obs_variance"], df["posterior_spread"])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlabel("Obs Variance")
    ax.set_ylabel("Posterior Spread")
    ax.set_title("Posterior Spread vs Obs Variance")

    # (2,0) Correlation matrix
    ax = axes[2, 0]
    corr_cols = [
        "obs",
        "obs_variance",
        "prior_mean",
        "posterior_mean",
        "prior_spread",
        "posterior_spread",
    ]
    available_cols = [c for c in corr_cols if c in df.columns]
    # Only compute correlation if we have at least 2 columns with valid data
    df_corr = df[available_cols].dropna()
    if len(df_corr) > 1 and len(available_cols) >= 2:
        corr = df[available_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        masked_corr = np.where(mask, np.nan, corr.values)
        im = ax.imshow(masked_corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(len(available_cols)))
        ax.set_yticks(range(len(available_cols)))
        ax.set_xticklabels(available_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(available_cols, fontsize=8)
        for i in range(len(available_cols)):
            for j in range(len(available_cols)):
                if not mask[i, j]:
                    ax.text(
                        j,
                        i,
                        f"{corr.values[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Correlation Matrix")
    else:
        ax.text(
            0.5,
            0.5,
            "Insufficient data\nfor correlation",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
        )
        ax.set_title("Correlation Matrix")
        ax.axis("off")

    # (2,1) Innovation histogram
    ax = axes[2, 1]
    innovations = df["obs"] - df["prior_mean"]
    valid_innovations = innovations.dropna()
    if len(valid_innovations) > 0:
        ax.hist(valid_innovations, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Innovation (obs - prior_mean)")
        ax.set_ylabel("Frequency")
        ax.set_title("Innovation Distribution")
    else:
        ax.text(
            0.5,
            0.5,
            "No valid innovations\n(all obs rejected or\nno prior computed)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
        )
        ax.set_title("Innovation Distribution")
        ax.axis("off")

    fig.suptitle(
        f"Filter Diagnostics{' - ' + title_suffix if title_suffix else ''}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def _common_lims(series_a: pd.Series, series_b: pd.Series) -> tuple[float, float]:
    """Compute common axis limits from two series, with a small margin."""
    a = series_a.replace([np.inf, -np.inf], np.nan).dropna()
    b = series_b.replace([np.inf, -np.inf], np.nan).dropna()
    if len(a) == 0 or len(b) == 0:
        return (0.0, 1.0)
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    margin = (hi - lo) * 0.05 if hi != lo else 0.1
    return (float(lo - margin), float(hi + margin))


def plot_increment_by_height(df: pd.DataFrame, title_suffix: str = "") -> Figure:
    """
    Create a 2-panel figure showing mean increment and mean observation by height.

    Args:
        df: DataFrame with columns obs, prior_mean, posterior_mean, z.
        title_suffix: Optional suffix for the figure title.

    Returns:
        A matplotlib Figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    df = df.copy()
    df["increment"] = df["posterior_mean"] - df["prior_mean"]
    df["z_bin"] = pd.cut(df["z"], bins=70)
    grouped = df.groupby("z_bin", observed=True).mean(numeric_only=True).reset_index()

    axes[0].scatter(grouped["z"], grouped["increment"], s=15)
    axes[0].set_title("Mean Increment by Height")
    axes[0].set_xlabel("Height (m)")
    axes[0].set_ylabel("Increment")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(grouped["z"], grouped["obs"], s=15)
    axes[1].set_title("Mean Observation by Height")
    axes[1].set_xlabel("Height (m)")
    axes[1].set_ylabel("Observation")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Increment Analysis{' - ' + title_suffix if title_suffix else ''}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_rank_histogram(
    hist: np.ndarray,
    diagnostics: dict,
    title_suffix: str = "",
) -> Figure:
    """
    Plot a rank histogram with confidence intervals and diagnostics text.

    Args:
        hist: Histogram counts from compute_rank_histogram.
        diagnostics: Diagnostics dict from compute_rank_histogram.
        title_suffix: Optional suffix for the title.

    Returns:
        A matplotlib Figure.
    """
    n_obs = diagnostics["n_obs"]
    n_ensemble = diagnostics["n_ensemble"]
    expected_count = n_obs / (n_ensemble + 1)

    std_dev = np.sqrt(n_obs * (1 / (n_ensemble + 1)) * (1 - 1 / (n_ensemble + 1)))
    ci_95 = 1.96 * std_dev

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(n_ensemble + 1)
    ax1.bar(x, hist, alpha=0.7, edgecolor="black", color="steelblue")
    ax1.axhline(
        expected_count,
        color="red",
        linestyle="--",
        label=f"Expected (uniform): {expected_count:.1f}",
        linewidth=2,
    )
    ax1.axhline(
        expected_count + ci_95, color="red", linestyle=":", alpha=0.5, linewidth=1
    )
    ax1.axhline(
        expected_count - ci_95, color="red", linestyle=":", alpha=0.5, linewidth=1
    )
    ax1.fill_between(
        [-0.5, n_ensemble + 0.5],
        expected_count - ci_95,
        expected_count + ci_95,
        color="red",
        alpha=0.1,
        label="95% CI",
    )
    ax1.set_xlabel("Rank", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title(
        f"Rank Histogram{' ' + title_suffix if title_suffix else ''}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, n_ensemble + 0.5)

    diag_text = (
        f"Diagnostics:\n"
        f"{'=' * 40}\n"
        f"N observations: {n_obs}\n"
        f"Ensemble size: {n_ensemble}\n\n"
        f"Innovation mean: {diagnostics['innovation_mean']:.6f}\n"
        f"Innovation std: {diagnostics['innovation_std']:.6f}\n"
        f"RMSE: {diagnostics['rmse']:.6f}\n\n"
        f"Normalized innovation mean: {diagnostics['normalized_innov_mean']:.3f}\n"
        f"Normalized innovation std: {diagnostics['normalized_innov_std']:.3f}\n"
        f"  (should be ~0 and ~1 if well-calibrated)\n\n"
        f"Correlation (obs vs ensemble): {diagnostics['correlation']:.4f}"
    )
    ax2.text(
        0.05,
        0.95,
        diag_text,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax2.axis("off")

    fig.tight_layout()
    return fig
