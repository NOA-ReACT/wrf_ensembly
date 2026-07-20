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

DART_QC_LABELS = {
    0: "0 OK",
    1: "1 Eval only",
    2: "2 Post fail",
    3: "3 Eval+post fail",
    4: "4 Fwd fail",
    5: "5 Ignored by config",
    6: "6 Bad obs",
    7: "7 Outlier",
    8: "8 Vert fail",
}
"""DART quality control flag meanings, prefixed with the flag value."""

# Green for QC=0, yellow/orange for partial use, red for rejected
DART_QC_COLORS = {
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
"""Colour per DART quality control flag, shared across the filter diagnostics."""


def _symmetric_limit(values, percentile: float = 99.0, fallback: float = 1.0) -> float:
    """
    Symmetric colour/axis limit from a percentile of |values|.

    Returns `fallback` when the percentile is not finite or zero, which happens for
    empty or all-NaN inputs and for a field that is identically zero.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    limit = float(np.percentile(np.abs(values), percentile))
    if not np.isfinite(limit) or limit == 0.0:
        return fallback
    return limit


def _empty_axes_message(ax, message: str) -> None:
    """Render an explanatory message on an axes that has no data to show."""

    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="gray",
    )
    ax.axis("off")



def _interp_to_pressure_level(
    var: xr.DataArray,
    pressure: xr.DataArray,
    target_hpa: float,
    level_dim: str,
) -> xr.DataArray:
    """Interpolate a 3D variable to a target pressure level.

    Args:
        var: 3D DataArray with a vertical dimension.
        pressure: 3D DataArray of air pressure in Pa, same shape as var.
        target_hpa: Target pressure level in hPa.
        level_dim: Name of the vertical dimension in both arrays.

    Returns:
        2D DataArray with the vertical dimension removed.
    """
    target_pa = target_hpa * 100.0

    def _interp_profile(var_profile: np.ndarray, pres_profile: np.ndarray) -> float:
        # pressure decreases with altitude; np.interp requires ascending xp
        return np.interp(target_pa, pres_profile[::-1], var_profile[::-1])

    return xr.apply_ufunc(
        _interp_profile,
        var,
        pressure,
        input_core_dims=[[level_dim], [level_dim]],
        vectorize=True,
        dask="parallelized",
    )


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
    level_dim = None
    for dim in forecast_var.dims:
        if dim not in spatial_dims and dim not in time_dims:
            level_dim = dim
            break
    if level_dim is not None:
        if variable_cfg.pressure_level is not None:
            forecast_pres = forecast_ds["air_pressure"]
            analysis_pres = analysis_ds["air_pressure"]
            for tdim in time_dims:
                if tdim in forecast_pres.dims:
                    forecast_pres = forecast_pres.isel({tdim: 0})
                if tdim in analysis_pres.dims:
                    analysis_pres = analysis_pres.isel({tdim: 0})
            forecast_var = _interp_to_pressure_level(
                forecast_var, forecast_pres, variable_cfg.pressure_level, level_dim
            )
            analysis_var = _interp_to_pressure_level(
                analysis_var, analysis_pres, variable_cfg.pressure_level, level_dim
            )
        elif variable_cfg.level is not None:
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
        figsize = (2.5, 6)
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

    # If diff vmin and vmix are not defined, ensure that the center of the colorbar is 0
    if variable_cfg.diff_vmin is None and variable_cfg.diff_vmax is None:
        max_abs = np.abs(diff_var).max()
        variable_cfg.diff_vmin = -max_abs
        variable_cfg.diff_vmax = max_abs

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

        if variable_cfg.extent:
            ax.set_extent(variable_cfg.extent, crs=ccrs.PlateCarree())

        if variable_cfg.pressure_level is not None:
            level_str = f" ({variable_cfg.pressure_level} hPa)"
        elif variable_cfg.level is not None:
            level_str = f" (level {variable_cfg.level})"
        else:
            level_str = ""
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


def plot_forecast(
    ds: xr.Dataset,
    variable_cfg: PlotVariableConfig,
    proj: ccrs.Projection | None = None,
    experiment_name: str = "",
    cycle: int = 0,
    time_str: str = "",
    plot_type: Literal["mean", "spread"] = "mean",
) -> Figure:
    """
    Create a single-panel map plot of a forecast variable.

    Args:
        ds: xarray Dataset with the forecast data (already time-selected).
        variable_cfg: Configuration for the variable to plot.
        proj: Cartopy projection for the map panel. If None, uses PlateCarree.
        experiment_name: Name of the experiment for the figure title.
        cycle: Cycle index for the figure title.
        time_str: Time string for the figure title.
        plot_type: Type of plot to create, either "mean" or "spread".

    Returns:
        A matplotlib Figure with a single panel.
    """
    var_name = variable_cfg.name
    data_var = ds[var_name]

    spatial_dims = {"south_north", "west_east", "x", "y"}
    time_dims = {"Time", "time", "t"}

    # Select vertical level if specified
    level_dim = None
    print(data_var.dims)
    for dim in data_var.dims:
        if dim not in spatial_dims and dim not in time_dims:
            level_dim = dim
            break
    if level_dim is not None:
        if variable_cfg.pressure_level is not None:
            pres_var = ds["air_pressure"]
            for tdim in time_dims:
                if tdim in pres_var.dims:
                    pres_var = pres_var.isel({tdim: 0})
            data_var = _interp_to_pressure_level(
                data_var, pres_var, variable_cfg.pressure_level, level_dim
            )
        elif variable_cfg.level is not None:
            data_var = data_var.isel({level_dim: variable_cfg.level})

    # Squeeze out time dimension if present
    for dim in time_dims:
        if dim in data_var.dims:
            data_var = data_var.isel({dim: 0})

    has_xy = "x" in ds.coords and "y" in ds.coords

    if plot_type == "spread":
        panel_title = "Forecast Spread"
        cmap = variable_cfg.spread_cmap
        vmin = variable_cfg.spread_vmin
        vmax = variable_cfg.spread_vmax
    else:
        panel_title = "Forecast"
        cmap = variable_cfg.cmap
        vmin = variable_cfg.vmin
        vmax = variable_cfg.vmax

    if vmin is None:
        vmin = float(data_var.min())
    if vmax is None:
        vmax = float(data_var.max())

    map_proj = proj or ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": map_proj}, figsize=(10, 6))

    if has_xy:
        mesh = ax.pcolormesh(
            ds["x"].values,
            ds["y"].values,
            data_var.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
    else:
        mesh = ax.pcolormesh(
            data_var.values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )

    ax.coastlines()

    if variable_cfg.extent:
        ax.set_extent(variable_cfg.extent, crs=ccrs.PlateCarree())

    if variable_cfg.pressure_level is not None:
        level_str = f" ({variable_cfg.pressure_level} hPa)"
    elif variable_cfg.level is not None:
        level_str = f" (level {variable_cfg.level})"
    else:
        level_str = ""
    ax.set_title(f"{panel_title}: {var_name}{level_str}")
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


def plot_omb_vs_oma(df: pd.DataFrame, title_suffix: str = "") -> Figure:
    """
    Hexbin of the residual after the analysis (O-A) against the innovation before it (O-B).

    Points below the 1:1 diagonal are observations whose residual shrank, i.e. the
    analysis pulled the model toward them. Axes are symmetric about zero at the
    99th percentile of |O-B|. Values are in the observation's native units.

    Args:
        df: DataFrame with obs, prior_mean and posterior_mean. Filter it by
            quality control flag before calling.
        title_suffix: Optional suffix for the figure title.

    Returns:
        A matplotlib Figure.
    """
    omb = (df["obs"] - df["prior_mean"]).to_numpy(dtype=float)
    oma = (df["obs"] - df["posterior_mean"]).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6.5))

    valid = np.isfinite(omb) & np.isfinite(oma)
    if valid.sum() == 0:
        _empty_axes_message(ax, "No observations with both a prior and a posterior")
    else:
        lim = _symmetric_limit(omb[valid], percentile=99.0)
        ax.hexbin(
            omb[valid],
            oma[valid],
            gridsize=55,
            bins="log",
            cmap="viridis",
            extent=(-lim, lim, -lim, lim),
        )
        ax.plot([-lim, lim], [-lim, lim], "r-", lw=0.8, label="no change")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("O-B (obs - prior mean)")
        ax.set_ylabel("O-A (obs - posterior mean)")
        ax.legend(fontsize=9)

    title = "Innovation before vs after analysis"
    if title_suffix:
        title += f" - {title_suffix}"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    ax.set_title("below the diagonal = analysis pulled toward the observations")

    fig.tight_layout()
    return fig


def plot_innovation_histogram(df: pd.DataFrame, title_suffix: str = "") -> Figure:
    """
    Overlaid step histograms of O-B and O-A on shared bins.

    A working analysis narrows the distribution and pulls its centre toward zero.
    Values are in the observation's native units.

    Args:
        df: DataFrame with obs, prior_mean and posterior_mean. Filter it by
            quality control flag before calling.
        title_suffix: Optional suffix for the figure title.

    Returns:
        A matplotlib Figure.
    """
    omb = (df["obs"] - df["prior_mean"]).to_numpy(dtype=float)
    oma = (df["obs"] - df["posterior_mean"]).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))

    if not np.isfinite(omb).any():
        _empty_axes_message(ax, "No observations with a prior")
    else:
        lim = _symmetric_limit(omb, percentile=99.0)
        bins = np.linspace(-lim, lim, 80)

        omb_finite = omb[np.isfinite(omb)]
        ax.hist(
            omb_finite,
            bins=bins,
            histtype="step",
            lw=1.6,
            label=f"O-B (RMS {np.sqrt(np.mean(omb_finite**2)):.4g})",
        )

        oma_finite = oma[np.isfinite(oma)]
        if oma_finite.size:
            ax.hist(
                oma_finite,
                bins=bins,
                histtype="step",
                lw=1.6,
                label=f"O-A (RMS {np.sqrt(np.mean(oma_finite**2)):.4g})",
            )

        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Innovation")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    title = "Innovation distribution"
    if title_suffix:
        title += f" - {title_suffix}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_obs_space_increment_map(
    df: pd.DataFrame,
    title_suffix: str = "",
    proj: ccrs.Projection | None = None,
) -> Figure:
    """
    Map of the analysis increment (posterior mean - prior mean) at the observation locations.

    Shows where the filter moved the model and in which direction. Colour limits are
    symmetric at the 98th percentile of |increment|, in the observation's native units.

    Args:
        df: DataFrame with longitude, latitude, prior_mean and posterior_mean.
            Filter it by quality control flag before calling.
        title_suffix: Optional suffix for the figure title.
        proj: Cartopy projection for the map. Defaults to PlateCarree.

    Returns:
        A matplotlib Figure.
    """
    if proj is None:
        proj = ccrs.PlateCarree()

    increment = (df["posterior_mean"] - df["prior_mean"]).to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": proj})

    if not np.isfinite(increment).any():
        _empty_axes_message(ax, "No observations with both a prior and a posterior")
    else:
        lim = _symmetric_limit(increment, percentile=98.0)
        scatter = ax.scatter(
            df["longitude"],
            df["latitude"],
            c=increment,
            s=6,
            cmap="RdBu_r",
            vmin=-lim,
            vmax=lim,
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(lw=0.6)
        fig.colorbar(
            scatter,
            ax=ax,
            orientation="horizontal",
            pad=0.05,
            shrink=0.8,
            label="analysis - prior",
        )

    title = "Observation-space analysis increment"
    if title_suffix:
        title += f" - {title_suffix}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig


def plot_cycle_consistency_timeseries(
    df: pd.DataFrame, title_suffix: str = ""
) -> Figure:
    """
    Across-cycle summary of the data assimilation consistency diagnostics.

    Panels, with one line per observation type:
      (0) O-B and O-A RMS, with the RMS reduction on a twin axis
      (1) Spread-skill ratio, with a reference line at 1
      (2) Desroziers observation-error estimate against the assigned error

    Cycles with too few assimilated observations carry NaN metrics and appear as
    gaps, which is intended.

    Args:
        df: Frame from `diagnostics.consistency_metrics_to_dataframe`, one row per
            (cycle, obs_type).
        title_suffix: Optional suffix for the figure title.

    Returns:
        A matplotlib Figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)

    obs_types = sorted(df["obs_type"].unique())

    ax_rms, ax_ratio, ax_sigma = axes
    ax_reduction = ax_rms.twinx()

    for obs_type in obs_types:
        sub = df[df["obs_type"] == obs_type].sort_values("cycle")
        cycles = sub["cycle"]

        (line,) = ax_rms.plot(cycles, sub["omb_rms"], "o-", label=f"{obs_type} O-B RMS")
        ax_rms.plot(
            cycles,
            sub["oma_rms"],
            "s--",
            color=line.get_color(),
            alpha=0.7,
            label=f"{obs_type} O-A RMS",
        )
        ax_reduction.plot(
            cycles,
            sub["rms_reduction_pct"],
            "^:",
            color=line.get_color(),
            alpha=0.4,
        )

        ax_ratio.plot(cycles, sub["spread_skill_ratio"], "o-", label=obs_type)

        (sline,) = ax_sigma.plot(
            cycles, sub["sigma_o_desroziers"], "o-", label=f"{obs_type} Desroziers"
        )
        ax_sigma.plot(
            cycles,
            sub["mean_sigma_o"],
            "s--",
            color=sline.get_color(),
            alpha=0.7,
            label=f"{obs_type} assigned",
        )

    ax_rms.set_ylabel("RMS")
    ax_rms.set_title("Innovation RMS before and after the analysis")
    ax_rms.grid(True, alpha=0.3)
    ax_rms.legend(fontsize=8, loc="upper left")
    ax_reduction.set_ylabel("RMS reduction (%)")
    ax_reduction.axhline(0, color="gray", lw=0.6, ls=":")

    ax_ratio.axhline(
        1.0, color="red", lw=0.8, ls="--", label="consistent (ratio = 1)"
    )
    ax_ratio.set_ylabel("var(O-B) / (σ_o² + σ_B²)")
    ax_ratio.set_title(
        "Spread-skill ratio - below 1 over-specifies the observation error, above 1 under-specifies it"
    )
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.legend(fontsize=8)

    ax_sigma.set_ylabel("σ_o")
    ax_sigma.set_xlabel("Cycle")
    ax_sigma.set_title("Desroziers observation-error estimate vs the assigned error")
    ax_sigma.grid(True, alpha=0.3)
    ax_sigma.legend(fontsize=8)

    title = "Data assimilation consistency across cycles"
    if title_suffix:
        title += f" - {title_suffix}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    fig.tight_layout()
    return fig


def generate_filter_stats_plots(
    df: pd.DataFrame,
    output_dir: Path,
    label: str,
    dpi: int = 150,
    logger=None,
    proj: ccrs.Projection | None = None,
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
        proj: Cartopy projection for the map plots. Defaults to PlateCarree.
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
    df_scatter = df.dropna(subset=["obs", "obs_std"])

    if len(df_scatter) == 0:
        log_warning(f"No valid observations for {label}, skipping")
        return

    # Scatter diagnostics (shows all obs, colored by QC flag)
    fig = plot_filter_scatter_diagnostics(df_scatter, title_suffix=label)
    fig.savefig(output_dir / "diagnostic_scatter.png", dpi=dpi, bbox_inches="tight")
    log_info(f"Saved {output_dir / 'diagnostic_scatter.png'}")
    plt.close(fig)

    # Same but only observations that passed QC
    fig = plot_filter_scatter_diagnostics(
        df_scatter.loc[df_scatter["dart_qc"] == 0], title_suffix=label
    )
    fig.savefig(
        output_dir / "diagnostic_scatter_dartqc_ok.png", dpi=dpi, bbox_inches="tight"
    )
    log_info(f"Saved {output_dir / 'diagnostic_scatter_dartqc_ok.png'}")
    plt.close(fig)

    # Analysis diagnostics: what the filter did to the observations it accepted.
    # These need a posterior, which is absent when the filter only evaluated obs.
    qc_ok = df_scatter.loc[df_scatter["dart_qc"] == 0]
    if len(qc_ok) == 0:
        log_warning(
            f"No QC-passed observations for {label}, skipping analysis diagnostics"
        )
    elif "posterior_mean" not in qc_ok.columns:
        log_warning(
            f"No posterior available for {label}, skipping analysis diagnostics"
        )
    else:
        fig = plot_omb_vs_oma(qc_ok, title_suffix=label)
        fig.savefig(output_dir / "omb_vs_oma.png", dpi=dpi, bbox_inches="tight")
        log_info(f"Saved {output_dir / 'omb_vs_oma.png'}")
        plt.close(fig)

        fig = plot_innovation_histogram(qc_ok, title_suffix=label)
        fig.savefig(
            output_dir / "innovation_histogram.png", dpi=dpi, bbox_inches="tight"
        )
        log_info(f"Saved {output_dir / 'innovation_histogram.png'}")
        plt.close(fig)

        fig = plot_obs_space_increment_map(qc_ok, title_suffix=label, proj=proj)
        fig.savefig(output_dir / "increment_map.png", dpi=dpi, bbox_inches="tight")
        log_info(f"Saved {output_dir / 'increment_map.png'}")
        plt.close(fig)

    # A map of observation locations
    fig, ax = plt.subplots(
        figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.scatter(
        df["longitude"],
        df["latitude"],
        color="black",
        s=10,
        transform=ccrs.PlateCarree(),
    )
    ax.coastlines()

    fig.savefig(output_dir / "observation_locations.png", dpi=dpi, bbox_inches="tight")
    log_info(f"Saved {output_dir / 'observation_locations.png'}")
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
      (0,1) Prior spread vs observation stdev
      (1,0) Posterior mean vs observation (colored by dart_qc)
      (1,1) Posterior spread vs observation stdev
      (2,0) Correlation matrix heatmap
      (2,1) Innovation histogram

    Args:
        df: DataFrame with columns obs, obs_std, prior_mean, prior_spread,
            posterior_mean, posterior_spread, dart_qc.
        title_suffix: Optional suffix for the figure title.

    Returns:
        A matplotlib Figure.
    """
    QC_LABELS = DART_QC_LABELS
    qc_colors = DART_QC_COLORS

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    qc_values = df["dart_qc"].values
    qc_unique = np.unique(qc_values)

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

    # (0,1) Prior spread vs obs stdev
    ax = axes[0, 1]
    ax.scatter(df["obs_std"], df["prior_spread"], alpha=0.5, s=10)
    lims = _common_lims(df["obs_std"], df["prior_spread"])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlabel("Obs StDev")
    ax.set_ylabel("Prior Spread")
    ax.set_title("Prior Spread vs Obs StDev")

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

    # (1,1) Posterior spread vs obs stddev
    ax = axes[1, 1]
    ax.scatter(df["obs_std"], df["posterior_spread"], alpha=0.5, s=10)
    lims = _common_lims(df["obs_std"], df["posterior_spread"])
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlabel("Obs StDev")
    ax.set_ylabel("Posterior Spread")
    ax.set_title("Posterior Spread vs Obs StDev")

    # (2,0) Correlation matrix
    ax = axes[2, 0]
    corr_cols = [
        "obs",
        "obs_std",
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
