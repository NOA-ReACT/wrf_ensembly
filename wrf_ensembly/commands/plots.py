from pathlib import Path

import click
import xarray as xr

from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.config import PlotVariableConfig
from wrf_ensembly.console import logger
from wrf_ensembly.diagnostics import read_obs_seq_nc
from wrf_ensembly.experiment import experiment
from wrf_ensembly.plotting import (
    generate_filter_stats_plots,
    plot_forecast_vs_analysis,
)
from wrf_ensembly.wrf import get_wrf_cartopy_crs


@click.group(name="plots", cls=GroupWithStartEndPrint)
def plots_cli():
    """Commands for generating diagnostic plots"""
    pass


@plots_cli.command()
@click.argument("cycle", type=int)
@click.option(
    "--variables",
    "-v",
    multiple=True,
    help="Plot only these variables (overrides config). Can be specified multiple times.",
)
@click.option(
    "--include-spread/--no-include-spread",
    default=None,
    help="Generate spread comparison plots in addition to mean plots. If not specified, uses config value.",
)
@pass_experiment_path
def forecast_vs_analysis(
    experiment_path: Path, cycle: int, variables: tuple, include_spread: bool | None
):
    """
    Generate three-panel comparison plots (forecast, analysis, difference) for a given cycle.

    Requires that `postprocess run` has been completed for the given cycle.
    Variables to plot are configured in [plots.forecast_vs_analysis] in the config file,
    or can be overridden with the --variables/-v option.

    By default, only mean plots are generated. Use --include-spread to also generate
    spread (standard deviation) comparison plots.
    """

    logger.setup("plots-forecast-vs-analysis", experiment_path)
    exp = experiment.Experiment(experiment_path)
    plot_cfg = exp.cfg.plots.forecast_vs_analysis

    # Determine whether to include spread plots
    if include_spread is None:
        include_spread = plot_cfg.include_spread

    # Locate data files
    forecast_dir = exp.paths.forecast_path(cycle)
    analysis_dir = exp.paths.analysis_path(cycle)

    forecast_mean_file = forecast_dir / f"forecast_mean_cycle_{cycle:03d}.nc"
    analysis_mean_file = analysis_dir / f"analysis_mean_cycle_{cycle:03d}.nc"

    if not forecast_mean_file.exists():
        logger.error(f"Forecast file not found: {forecast_mean_file}")
        return
    if not analysis_mean_file.exists():
        logger.error(f"Analysis file not found: {analysis_mean_file}")
        return

    # Check for spread files if needed
    if include_spread:
        forecast_spread_file = forecast_dir / f"forecast_sd_cycle_{cycle:03d}.nc"
        analysis_spread_file = analysis_dir / f"analysis_sd_cycle_{cycle:03d}.nc"

        if not forecast_spread_file.exists():
            logger.error(f"Forecast spread file not found: {forecast_spread_file}")
            logger.error("Spread files are required when --include-spread is enabled")
            return
        if not analysis_spread_file.exists():
            logger.error(f"Analysis spread file not found: {analysis_spread_file}")
            logger.error("Spread files are required when --include-spread is enabled")
            return

    # Determine which variables to plot
    if variables:
        # CLI override: create default PlotVariableConfig for each
        vars_to_plot = [PlotVariableConfig(name=v) for v in variables]
    elif plot_cfg.variables:
        vars_to_plot = plot_cfg.variables
    else:
        logger.error(
            "No variables configured for plotting. "
            "Set [plots.forecast_vs_analysis] variables in config or use --variables/-v."
        )
        return

    # Get cartopy projection
    try:
        proj = get_wrf_cartopy_crs(exp.cfg.domain_control)
    except NotImplementedError:
        logger.warning(
            "Could not create cartopy projection for this domain, falling back to PlateCarree"
        )
        proj = None

    # Output directory
    output_dir = exp.paths.plots / "forecast_vs_analysis" / f"cycle_{cycle:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open datasets and align forecast time to analysis time
    forecast_mean_ds = xr.open_dataset(forecast_mean_file)
    analysis_mean_ds = xr.open_dataset(analysis_mean_file)

    if include_spread:
        forecast_spread_ds = xr.open_dataset(forecast_spread_file)
        analysis_spread_ds = xr.open_dataset(analysis_spread_file)

    # The forecast file may contain multiple timesteps while the analysis is always
    # a single timestep at the end of the cycle. Select the forecast timestep that
    # matches the analysis time.
    analysis_time = analysis_mean_ds["t"].values[-1]
    analysis_time_str = str(analysis_time)[:19]  # Trim to "YYYY-MM-DDTHH:MM:SS"

    matching = forecast_mean_ds["t"] == analysis_time
    if matching.any():
        forecast_mean_ds = forecast_mean_ds.sel(t=analysis_time)
        analysis_mean_ds = analysis_mean_ds.isel(t=-1)
        if include_spread:
            forecast_spread_ds = forecast_spread_ds.sel(t=analysis_time)
            analysis_spread_ds = analysis_spread_ds.isel(t=-1)
        logger.info(
            f"Selected forecast timestep matching analysis time: {analysis_time}"
        )
    else:
        logger.warning(
            f"No matching timestep found in forecast for analysis time {analysis_time}, "
            "using last forecast timestep"
        )
        forecast_mean_ds = forecast_mean_ds.isel(t=-1)
        analysis_mean_ds = analysis_mean_ds.isel(t=-1)
        if include_spread:
            forecast_spread_ds = forecast_spread_ds.isel(t=-1)
            analysis_spread_ds = analysis_spread_ds.isel(t=-1)

    try:
        for var_cfg in vars_to_plot:
            # Generate mean plots
            if var_cfg.name not in forecast_mean_ds:
                logger.warning(
                    f"Variable '{var_cfg.name}' not found in forecast file, skipping"
                )
                continue
            if var_cfg.name not in analysis_mean_ds:
                logger.warning(
                    f"Variable '{var_cfg.name}' not found in analysis file, skipping"
                )
                continue

            logger.info(f"Plotting {var_cfg.name} (mean)...")
            fig = plot_forecast_vs_analysis(
                forecast_mean_ds,
                analysis_mean_ds,
                var_cfg,
                arrangement=plot_cfg.arrangement,
                proj=proj,
                experiment_name=exp.cfg.metadata.name,
                cycle=cycle,
                time_str=analysis_time_str,
                plot_type="mean",
            )

            suffix = f"_level{var_cfg.level}" if var_cfg.level is not None else ""
            output_path = output_dir / f"{var_cfg.name}{suffix}.png"
            fig.savefig(output_path, dpi=plot_cfg.dpi, bbox_inches="tight")
            logger.info(f"Saved {output_path}")
            fig.clear()
            import matplotlib.pyplot as plt

            plt.close(fig)

            # Generate spread plots if requested
            if include_spread:
                if var_cfg.name not in forecast_spread_ds:
                    logger.warning(
                        f"Variable '{var_cfg.name}' not found in forecast spread file, skipping spread plot"
                    )
                    continue
                if var_cfg.name not in analysis_spread_ds:
                    logger.warning(
                        f"Variable '{var_cfg.name}' not found in analysis spread file, skipping spread plot"
                    )
                    continue

                logger.info(f"Plotting {var_cfg.name} (spread)...")
                fig = plot_forecast_vs_analysis(
                    forecast_spread_ds,
                    analysis_spread_ds,
                    var_cfg,
                    arrangement=plot_cfg.arrangement,
                    proj=proj,
                    experiment_name=exp.cfg.metadata.name,
                    cycle=cycle,
                    time_str=analysis_time_str,
                    plot_type="spread",
                )

                output_path = output_dir / f"{var_cfg.name}{suffix}_spread.png"
                fig.savefig(output_path, dpi=plot_cfg.dpi, bbox_inches="tight")
                logger.info(f"Saved {output_path}")
                fig.clear()
                plt.close(fig)

    finally:
        forecast_mean_ds.close()
        analysis_mean_ds.close()
        if include_spread:
            forecast_spread_ds.close()
            analysis_spread_ds.close()

    logger.info(f"All plots saved to {output_dir}")


@plots_cli.command()
@click.argument("cycle", type=int)
@pass_experiment_path
def cycle_filter_stats(experiment_path: Path, cycle: int):
    """
    Generate diagnostic plots from DART filter output for a given cycle.

    Reads the obs_seq.final NetCDF diagnostics file and produces scatter plots
    and rank histograms. Plots are generated for each observation type
    separately and for all types combined.
    """

    logger.setup("plots-cycle-filter-stats", experiment_path)
    exp = experiment.Experiment(experiment_path)

    diag_file = exp.paths.data_diag / f"cycle_{exp.cycles[cycle].index}.nc"
    if not diag_file.exists():
        logger.error(f"Diagnostics file not found: {diag_file}")
        logger.error("Ensure `ensemble filter` has been run for this cycle.")
        return

    logger.info(f"Reading diagnostics from {diag_file}")
    df = read_obs_seq_nc(diag_file)
    logger.info(f"Loaded {len(df)} observations")

    base_output_dir = exp.paths.plots / "cycle_filter_stats" / f"cycle_{cycle:03d}"

    # Generate plots for all observations combined
    logger.info("Generating plots for all observation types combined...")
    generate_filter_stats_plots(
        df,
        base_output_dir / "all",
        "All Types",
        logger=logger,
    )

    # Generate plots per observation type
    obs_types = df["obs_type"].unique()
    if len(obs_types) > 1:
        for obs_type in sorted(obs_types):
            logger.info(f"Generating plots for {obs_type}...")
            subset = df[df["obs_type"] == obs_type].copy()
            generate_filter_stats_plots(
                subset,
                base_output_dir / obs_type,
                obs_type,
                logger=logger,
            )

    logger.info(f"All plots saved to {base_output_dir}")
