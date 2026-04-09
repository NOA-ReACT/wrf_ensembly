"""
Commands about handling observations in the context of an experiment (adding, retrieving, etc).
"""

import concurrent
import concurrent.futures
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from rich.console import Console
from rich.progress import Progress, track
from rich.table import Table

from wrf_ensembly import experiment, external, observations, wrf
from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.console import console, logger
from wrf_ensembly.observations import plotting
from wrf_ensembly.observations.operations import _build_subtitle
from wrf_ensembly.observations.plotting import plot_observation_locations_on_map
from wrf_ensembly.utils import determine_jobs


@click.group(name="observations", cls=GroupWithStartEndPrint)
def observations_cli():
    """Commands related to handling observations in the context of an experiment"""
    pass


@observations_cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--jobs", "-j", type=int, default=1, help="Number of parallel jobs to use"
)
@pass_experiment_path
def add(experiment_path: Path, files: list[Path], jobs: int):
    """
    Bulk add observations to the experiment
    The files are first spatially and temporally trimmed according to the experiment configuration,
    then added to the experiment's observation database. The first step is done in parallel.
    """

    logger.setup("observations-convert-obs", experiment_path)
    exp = experiment.Experiment(experiment_path)

    files_to_process = []
    for p in files:
        if p.is_dir():
            files_to_process.extend(list(p.glob("*.parquet")))
        else:
            files_to_process.append(p)

    # Compute a temp. output path for each file
    exp.paths.obs_temp.mkdir(exist_ok=True, parents=True)
    for f in exp.paths.obs_temp.glob("*.parquet"):
        f.unlink()
    io_paths = [(f, exp.paths.obs_temp / f.name) for f in files_to_process]

    # Process the files in different processes, using a rich Progress to display a bar
    jobs = determine_jobs(jobs)
    counts = {}
    # Use maxtasksperchild to recycle worker processes and prevent memory accumulation
    with (
        ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as executor,
        Progress() as progress,
    ):
        task = progress.add_task(
            "[cyan]Trimming observation files...", total=len(files_to_process)
        )
        futures = [
            executor.submit(
                exp.obs.trim_observation_file,
                input_path=input_path,
                output_path=output_path,
            )
            for input_path, output_path in io_paths
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                filename, avail_obs, used_obs = future.result()
                counts[filename] = (avail_obs, used_obs)
                progress.advance(task)
                progress.console.print(
                    f"{filename}: Trimmed {avail_obs} -> {used_obs} observations"
                )
            except Exception as e:
                progress.console.print(f"[red]Error processing file: {e}[/red]")
                progress.console.print("[red]Traceback:[/red]")
                progress.console.print(traceback.format_exc())
                logger.error(f"Error processing file: {e}")
                logger.error(traceback.format_exc())
                sys.exit(1)

    # Add the trimmed files to the duckDB
    for _, output_path in track(
        io_paths, description="Adding observations to database..."
    ):
        if output_path.is_file():
            exp.obs.add_observation_file(output_path)

    # Clean up temp files
    for f in exp.paths.obs_temp.glob("*.parquet"):
        f.unlink()

    # Print a summary of what was added
    table = Table(title="Observation Files Added")
    table.add_column("File", style="cyan")
    table.add_column("Observations Available", style="green")
    table.add_column("Observations Added", style="green")
    skipped_counter = 0
    for f, (obs_avail, obs_added) in counts.items():
        if obs_added > 0:
            table.add_row(str(f), str(obs_avail), str(obs_added))
        else:
            skipped_counter += 1

    console.print(table)
    if skipped_counter > 0:
        console.print(f"Skipped {skipped_counter} files that had no observations added")


@observations_cli.command()
@pass_experiment_path
def show(experiment_path: Path):
    """
    Prints two tables, one with every combination of instrument, quantity and how many observations,
    and one with all used files, their time range and instrument.
    """

    exp = experiment.Experiment(experiment_path)

    # Show summary table of instrument/quantity/count
    quantities = exp.obs.get_available_quantities()

    table = Table(title="Available Observation Quantities")
    table.add_column("Instrument", style="cyan", no_wrap=True)
    table.add_column("Quantity", style="cyan", no_wrap=True)
    table.add_column("Count", style="green")
    for info in quantities:
        table.add_row(info["instrument"], info["quantity"], str(info["count"]))
    Console().print(table)

    # Show summary table of instrument/quantity/count
    obs_files = exp.obs.get_available_observations_overview()

    table = Table(title="Available Observation Files")
    table.add_column("Instrument", style="cyan", no_wrap=True)
    table.add_column("Start Time", style="green")
    table.add_column("End Time", style="green")
    table.add_column("Count", style="green")
    table.add_column("Model Values", style="green")
    table.add_column("Filename", style="magenta")

    for obs_file in obs_files:
        table.add_row(
            obs_file["instrument"],
            obs_file["start_time"].strftime("%Y-%m-%d %H:%M"),
            obs_file["end_time"].strftime("%Y-%m-%d %H:%M"),
            str(obs_file["count"]),
            str(obs_file["model_forecasts"]),
            str(obs_file["filename"]),
        )

    Console().print(table)


@observations_cli.command()
@click.argument("filename", type=str)
@pass_experiment_path
def delete(experiment_path: Path, filename: str):
    """
    Delete an observation file from the experiment's observation database.
    """
    exp = experiment.Experiment(experiment_path)
    logger.setup("observations-delete", experiment_path)

    rows = exp.obs.delete_observation_file(filename)
    logger.info(f"Deleted {rows} rows from observation database")


@observations_cli.command()
@click.option(
    "--cycle",
    type=int,
    default=None,
    help="If provided, only convert observations for this cycle (0-indexed)",
)
@click.option(
    "--jobs",
    "-j",
    type=int,
    default=None,
    help="Number of parallel jobs to use",
)
@click.option(
    "--skip-dart",
    is_flag=True,
    default=False,
    help="Skip converting to DART obs_seq format, only write parquet files",
)
@pass_experiment_path
def prepare_cycles(
    experiment_path: Path,
    cycle: int | None = None,
    jobs: int | None = None,
    skip_dart: bool = False,
):
    """
    Prepares observation files for each cycle by extracting relevant observations for that
    cycle's time window and converting them to DART obs_seq format.

    Required for `filter` to be able to use the observations.

    You must build the `wrf_ensembly` observation converter in DART for this to work,
    check the `DART/observations/obs_converters/wrf_ensembly` directory.

    The command will create one parquet and one obs_seq file per cycle in the experiment's
    `obs/` directory, named `cycle_XXX.parquet` and `cycle_XXX.obs_seq` respectively.
    You can skip the obs_seq conversion with `--skip-dart` if you only want the parquet files
    for inspection.
    """

    logger.setup("observations-prepare-cycles", experiment_path)
    exp = experiment.Experiment(experiment_path)

    if not exp.cfg.observations.instruments_to_assimilate:
        logger.info("Assimilating all available instruments")
    else:
        logger.info(
            f"Instruments to assimilate: {', '.join(exp.cfg.observations.instruments_to_assimilate)}"
        )

    if cycle is not None:
        cycles = [exp.cycles[cycle]]
    else:
        cycles = exp.cycles

    commands = []
    for c in cycles:
        # Grab observations from duckDB
        cycle_obs = exp.obs.get_observations_for_cycle(c)

        if cycle_obs is None or cycle_obs.empty:
            logger.info(f"No observations found for cycle {c.index}, skipping")
            continue

        # Also write to parquet for easy inspection later
        parquet_path = exp.paths.obs / f"cycle_{c.index:03d}.parquet"
        observations.io.write_obs(cycle_obs, parquet_path)

        output_path = exp.paths.obs / f"cycle_{c.index:03d}.obs_seq"
        commands.append(
            observations.dart.convert_to_dart_obs_seq(
                dart_path=exp.cfg.directories.dart_root,
                observations=cycle_obs.loc[
                    cycle_obs["qc_flag"] == 0
                ].copy(),  # Only use good quality obs
                output_location=output_path,
            )
        )

    if skip_dart:
        logger.info("Skipping DART obs_seq conversion as per --skip-dart")
        return

    jobs = determine_jobs(jobs)
    for res in external.run_in_parallel(commands, jobs, stop_on_failure=True):
        if res.returncode != 0:
            logger.error(f"Failed command: {res.command}")
            logger.error(res.output)
        else:
            logger.info(f"Converted observations for a cycle to {res.command[-1]}")
        logger.debug(res.output)

    logger.info("Finished converting observations to DART obs_seq format")


@observations_cli.command()
@pass_experiment_path
def cycle_summary(experiment_path: Path):
    """
    Print a table showing observation counts for all cycles.

    Columns: cycle number, total observations available, observations to be assimilated.
    """

    logger.setup("observations-cycle-summary", experiment_path)
    exp = experiment.Experiment(experiment_path)

    df = exp.obs.get_cycle_summary(exp.cycles)

    table = Table(title="Observations per Cycle")
    table.add_column("Cycle", style="cyan", justify="right")
    table.add_column("Total", style="white", justify="right")
    table.add_column("To Assimilate", style="green", justify="right")

    for _, row in df.iterrows():
        table.add_row(
            str(int(row["cycle_index"])),
            str(row["total"]),
            str(row["to_assimilate"]),
        )

    console.print(table)


@observations_cli.command()
@click.argument("cycle", type=int, required=True, default=None)
@click.option("--as_json", is_flag=True, default=False, help="Output in JSON format")
@pass_experiment_path
def cycle_info(experiment_path: Path, cycle: int, as_json: bool):
    """
    Print a summary of the observations for a specific cycle.

    Specifically: Number of observations per file, how many were assimilated, with percentages.
    """

    logger.setup("observations-cycle-info", experiment_path)
    exp = experiment.Experiment(experiment_path)

    cycle_info = exp.cycles[cycle]
    file_stats = exp.obs.get_cycle_file_info(cycle_info)

    if file_stats.empty:
        logger.error(
            f"Cycle {cycle_info.index} has no observations in the time window {cycle_info.start} to {cycle_info.end}"
        )
        return

    file_stats["pct"] = file_stats["assimilated"] / file_stats["total"] * 100
    total = int(file_stats["total"].sum())
    total_assimilated = int(file_stats["assimilated"].sum())

    if as_json:
        out = {
            "cycle": cycle_info.index,
            "time_window": {"start": str(cycle_info.start), "end": str(cycle_info.end)},
            "total": total,
            "assimilated": total_assimilated,
            "files": [
                {
                    "filename": row["orig_filename"],
                    "instrument": row["instrument"],
                    "total": int(row["total"]),
                    "assimilated": int(row["assimilated"]),
                    "pct": round(float(row["pct"]), 2),
                }
                for _, row in file_stats.iterrows()
            ],
        }
        print(json.dumps(out, indent=2))
        return

    console.print(f"[bold]Observation stats for cycle {cycle_info.index}:[/bold]")
    console.print(f"Time window: {cycle_info.start} to {cycle_info.end}")
    pct_str = f" ({total_assimilated / total * 100:.1f}%)" if total > 0 else ""
    console.print(
        f"Total observations: {total} | Assimilated: {total_assimilated}{pct_str}"
    )

    files_table = Table(title="Observations by file")
    files_table.add_column("File", style="cyan")
    files_table.add_column("Instrument", style="magenta", no_wrap=True)
    files_table.add_column("Total", style="white", justify="right")
    files_table.add_column("Assimilated", style="green", justify="right")
    files_table.add_column("%", style="green", justify="right")
    for _, row in file_stats.iterrows():
        files_table.add_row(
            row["orig_filename"],
            row["instrument"],
            str(int(row["total"])),
            str(int(row["assimilated"])),
            f"{row['pct']:.1f}%",
        )
    console.print(files_table)


@observations_cli.command()
@click.argument("cycle", type=int, required=True, default=None)
@pass_experiment_path
def plot_cycle_locations(experiment_path: Path, cycle: int):
    """
    Plot the locations of observations for a specific cycle on a map.

    The plot will be saved to the plot subdirectory in the experiment directory, named
    `obs_cycle_XXX_locations.png`.

    Args:
        cycle: The cycle index to plot observations for.
    """

    logger.setup("observations-plot-cycle-locations", experiment_path)
    exp = experiment.Experiment(experiment_path)

    cycle_info = exp.cycles[cycle]

    parquet_path = exp.paths.obs / f"cycle_{cycle_info.index:03d}.parquet"
    if not parquet_path.is_file():
        logger.error(
            f"Cycle {cycle_info.index} parquet file {parquet_path} does not exist, run `wrf-ensembly obs prepare-cycles` first or there are no observations for this cycle"
        )
        return
    obs = observations.io.read_obs(parquet_path)
    if obs is None or obs.empty:
        logger.error(f"Cycle {cycle_info.index} has no observations, cannot plot")
        return

    # Find a wrfinput file
    if not exp.cfg.data.per_member_meteorology:
        wrfinput_path = exp.paths.data_icbc / "wrfinput_d01_cycle_0"
    else:
        wrfinput_path = exp.paths.data_icbc / "member_00" / "wrfinput_d01_cycle_0"

    if wrfinput_path.exists():
        bounds = wrf.get_spatial_domain_bounds(wrfinput_path)
    else:
        logger.warning("No wrfinput file found, cannot set map bounds")
        bounds = None

    fig = observations.plotting.plot_observation_locations_on_map(
        obs,
        proj=wrf.get_wrf_cartopy_crs(exp.cfg.domain_control),
        domain_bounds=bounds,
    )
    fig.suptitle(f"Observation Locations for Cycle {cycle_info.index}")

    output_path = exp.paths.plots / f"obs_locations_cycle_{cycle_info.index:03d}.png"
    output_path.parent.mkdir(exist_ok=True, parents=True)

    fig.savefig(output_path)
    logger.info(
        f"Saved observation locations plot for cycle {cycle_info.index} to {output_path}"
    )


@observations_cli.command()
@click.argument("cycle", type=int, required=True)
@click.option(
    "--center-lat",
    type=float,
    default=None,
    help="Latitude to center the zoom on. Defaults to domain center.",
)
@click.option(
    "--center-lon",
    type=float,
    default=None,
    help="Longitude to center the zoom on. Defaults to domain center.",
)
@click.option(
    "--window-size",
    type=int,
    default=15,
    help="Number of grid points to show in each direction from center (default 15, giving ~30x30).",
)
@click.option(
    "--instrument",
    type=str,
    default=None,
    help="Filter observations to this instrument.",
)
@click.option(
    "--quantity",
    type=str,
    default=None,
    help="Filter observations to this quantity.",
)
@click.option(
    "--keep-only-good-qc",
    is_flag=True,
    help="Keep only observations with good quality control (qc_flag = 0).",
)
@pass_experiment_path
def plot_compare_obs_to_grid(
    experiment_path: Path,
    cycle: int,
    center_lat: float | None,
    center_lon: float | None,
    window_size: int,
    instrument: str | None,
    quantity: str | None,
    keep_only_good_qc: bool,
):
    """
    Plot observation locations overlaid on the WRF model grid, zoomed in to compare
    observation density with grid resolution. Useful for determining optimal superobbing
    bin sizes.

    The plot shows grid points as grey squares connected by lines, with observations
    as colored dots on top. By default it zooms to ~30x30 grid points around the domain
    center.
    """

    import numpy as np
    import xarray as xr

    logger.setup("observations-plot-compare-obs-to-grid", experiment_path)
    exp = experiment.Experiment(experiment_path)

    cycle_info = exp.cycles[cycle]

    # Load observations
    obs = exp.obs.get_observations_for_cycle(cycle_info)
    if obs is None or obs.empty:
        logger.error(f"Cycle {cycle_info.index} has no observations, cannot plot")
        return

    # Filter by instrument/quantity if requested
    if instrument is not None:
        obs = obs[obs["instrument"] == instrument]
    if quantity is not None:
        obs = obs[obs["quantity"] == quantity]
    if obs.empty:
        logger.error("No observations remaining after filtering")
        return

    # Keep only good QC if requested
    if keep_only_good_qc:
        obs = obs[obs["qc_flag"] == 0]
        if obs.empty:
            logger.error("No observations remaining after filtering for good QC")
            return

    # Load grid coordinates from wrfinput
    wrfinput_path = exp.paths.ic_path(0, 0)
    if not wrfinput_path.exists():
        logger.error(f"wrfinput file not found at {wrfinput_path}")
        return

    with xr.open_dataset(wrfinput_path) as ds:
        grid_lat = ds["XLAT"].isel(Time=0).values
        grid_lon = ds["XLONG"].isel(Time=0).values

    # Determine center grid index
    center_idx = None
    if center_lat is not None and center_lon is not None:
        # Find nearest grid point to the given lat/lon
        dist = (grid_lat - center_lat) ** 2 + (grid_lon - center_lon) ** 2
        center_idx = np.unravel_index(np.argmin(dist), dist.shape)
        logger.info(
            f"Centering on grid index {center_idx} (nearest to {center_lat}, {center_lon})"
        )

    proj = wrf.get_wrf_cartopy_crs(exp.cfg.domain_control)
    fig = observations.plotting.plot_obs_vs_grid(
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        observations=obs,
        proj=proj,
        center_idx=center_idx,
        window_size=window_size,
    )

    title = f"Observations vs Grid - Cycle {cycle_info.index}"
    if instrument or quantity:
        filters = [f for f in [instrument, quantity] if f is not None]
        title += f" ({', '.join(filters)})"
    fig.suptitle(title)

    output_path = exp.paths.plots / f"obs_vs_grid_cycle_{cycle_info.index:03d}.png"
    output_path.parent.mkdir(exist_ok=True, parents=True)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved obs vs grid plot to {output_path}")


@observations_cli.command()
@click.argument("filename", type=str)
@click.option("--dpi", type=int, default=200, help="DPI for output file")
@click.option("--vmin", type=float, default=None, help="Override colorbar minimum")
@click.option("--vmax", type=float, default=None, help="Override colorbar maximum")
@click.option(
    "--ylim",
    type=(float, float),
    default=(None, None),
    help="Override y-axis limits (low high)",
)
@click.option("--qc", type=int, default=None, help="Filter to specific QC flag value")
@click.option("--no-robust", is_flag=True, help="Disable robust color scaling")
@click.option(
    "--with-model",
    is_flag=True,
    help="Create 3-panel plot with observation, model equivalent, and O-B departure",
)
@pass_experiment_path
def plot(
    experiment_path: Path,
    filename: str,
    dpi: int = 200,
    vmin: float | None = None,
    vmax: float | None = None,
    ylim: tuple[float | None, float | None] = (None, None),
    qc: int | None = None,
    no_robust: bool = False,
    with_model: bool = False,
):
    """
    Plot observations from the experiment's duckdb database, filtered by their original filename.

    FILENAME is the orig_filename value used to filter observations (not a full path).
    Plots are saved to the experiment's plots directory.
    """

    logger.setup("observations-plot-file", experiment_path)
    exp = experiment.Experiment(experiment_path)

    t = time.process_time()
    logger.info(f"Reading observations for filename '{filename}'...")
    df = exp.obs.get_observations_by_filename(filename)
    logger.info(f"Done in {time.process_time() - t:.2f}s")

    if df is None or df.empty:
        logger.error(f"No observations found with orig_filename = '{filename}'")
        return

    plot_kwargs: dict[str, Any] = {}
    if vmin is not None:
        plot_kwargs["vmin"] = vmin
    if vmax is not None:
        plot_kwargs["vmax"] = vmax
    if no_robust:
        plot_kwargs["robust"] = False

    output_dir = exp.paths.plots / filename
    output_dir.mkdir(exist_ok=True, parents=True)

    stem = Path(filename).stem
    instrument_quantity = (df["instrument"] + "." + df["quantity"]).unique()
    print(instrument_quantity)
    for iq in instrument_quantity:
        logger.info(f"Plotting for {iq}")
        group = df[df["instrument"] + "." + df["quantity"] == iq]

        # Build list of (model_column, suffix) pairs to plot
        model_plots: list[tuple[str, str]] = []
        if with_model:
            for col, suffix in [
                ("model_forecast", "_vs_forecast"),
                ("model_analysis", "_vs_analysis"),
            ]:
                if col in group.columns and not group[col].isna().all():
                    model_plots.append((col, suffix))
                else:
                    logger.warning(f"No {col} data for {iq}, skipping")

            if not model_plots:
                logger.warning(f"No model data for {iq}, falling back to single panel")

        # Always plot observations-only
        figures: list[tuple[Figure, str]] = []
        fig = plotting.plot_observations(
            group, keep_only_qc_flag=qc, plot_kwargs=plot_kwargs
        )
        figures.append((fig, ""))

        for col, suffix in model_plots:
            fig = plotting.plot_observations_vs_model(
                group,
                model_column=col,
                keep_only_qc_flag=qc,
                plot_kwargs=plot_kwargs,
            )
            figures.append((fig, suffix))

        for fig, suffix in figures:
            if ylim != (None, None):
                for ax in fig.get_axes():
                    if not isinstance(ax, Axes):
                        continue
                    if ax.get_label() == "<colorbar>":
                        continue
                    ax.set_ylim(*ylim)

            fig.text(0.5, 1.0, stem, ha="center", va="top", fontsize=9, color="0.4")
            subtitle = _build_subtitle(group, qc=qc)
            fig.text(
                0.5, 0.0, subtitle, ha="center", va="bottom", fontsize=7, color="0.5"
            )

            output_path = output_dir / f"{stem}_{iq}{suffix}.png"
            fig.tight_layout()
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.3)
            plt.close(fig)
        logger.info(f"Saved plot to {output_path}")

    logger.info("Plotting map...")
    fig = plot_observation_locations_on_map(df, None)
    fig.tight_layout()
    map_path = output_dir / f"{stem}_map.png"
    fig.savefig(map_path, dpi=dpi, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    logger.info(f"Saved map plot to {map_path}")
