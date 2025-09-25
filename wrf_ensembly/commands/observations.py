"""
Commands about handling observations in the context of an experiment (adding, retrieving, etc).
"""

import concurrent
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import json
from pathlib import Path
import sys

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, track

from wrf_ensembly import experiment, external, observations, wrf
from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.console import logger, console
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
    with ProcessPoolExecutor(max_workers=jobs) as executor, Progress() as progress:
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
    table.add_column("Path", style="magenta")

    for obs_file in obs_files:
        table.add_row(
            obs_file["instrument"],
            obs_file["start_time"].strftime("%Y-%m-%d %H:%M"),
            obs_file["end_time"].strftime("%Y-%m-%d %H:%M"),
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
@pass_experiment_path
def superorbing(experiment_path: Path):
    """
    Downsample observations using superorbing, according to the experiment configuration.

    Superorbing groups observations that are close in space and time, and combines them into
    a single "superobservation". This can help reduce the number of observations, improving
    performance and avoiding biases from over-represented areas.

    The configuration for superorbing should be specified in the experiment's config file
    under `observations.superorbing`. See the documentation for details on the configuration
    options.

    Note: This will remove any previously downsampled observations from the database before
    applying superorbing again.
    """

    logger.setup("observations-superorbing", experiment_path)
    exp = experiment.Experiment(experiment_path)
    exp.obs.apply_superorbing()


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
                observations=cycle_obs,
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
@click.argument("cycle", type=int, required=True, default=None)
@click.option("--as_json", is_flag=True, default=False, help="Output in JSON format")
@pass_experiment_path
def cycle_stats(experiment_path: Path, cycle: int, as_json: bool):
    """
    Print a summary of the observations for a specific cycle.

    Specifically: Number of observations, instruments, obs per instrument, original files.
    """

    logger.setup("observations-cycle-stats", experiment_path)
    exp = experiment.Experiment(experiment_path)

    cycle_info = exp.cycles[cycle]

    parquet_path = exp.paths.obs / f"cycle_{cycle_info.index:03d}.parquet"
    if not parquet_path.is_file():
        logger.error(
            f"Cycle {cycle_info.index} parquet file {parquet_path} does not exist, run `wrf-ensembly obs prepare-cycles` first"
        )
        return
    obs = observations.io.read_obs(parquet_path)
    if obs is None or obs.empty:
        logger.error(f"Cycle {cycle_info.index} has no observations, cannot show stats")
        return

    stats = {}
    stats["count"] = len(obs)
    stats["instruments"] = {
        instr: {
            "count": obs[obs["instrument"] == instr].shape[0],
            "files": obs[obs["instrument"] == instr]["orig_filename"]
            .value_counts()
            .to_dict(),
            "qc": {
                int(qc): int((obs[obs["instrument"] == instr]["qc_flag"] == qc).sum())
                for qc in obs[obs["instrument"] == instr]["qc_flag"].unique()
            },
        }
        for instr in obs["instrument"].unique()
    }
    stats["quantities"] = obs["quantity"].value_counts().to_dict()
    stats["qc"] = obs["qc_flag"].value_counts().to_dict()

    # If JSON, dump the `stats` dict. Otherwise print some nice tables with rich
    if as_json:
        print(json.dumps(stats, indent=2))
        return

    console.print(f"[bold]Observation stats for cycle {cycle_info.index}:[/bold]")
    console.print(f"Time window: {cycle_info.start} to {cycle_info.end}")
    console.print(f"Total observations: {stats['count']}")
    console.print(
        f"Valid observations (qc_flag=0): {stats['qc'].get(0, 0)} ({stats['qc'].get(0, 0) / stats['count'] * 100:.2f}%)"
    )

    instr_table = Table(title="Instruments")
    instr_table.add_column("Instrument", style="cyan", no_wrap=True)
    instr_table.add_column("Count", style="green")
    instr_table.add_column("Valid QC (qc_flag=0)", style="green")
    for instr, info in stats["instruments"].items():
        instr_table.add_row(instr, str(info["count"]), str(info["qc"].get(0, 0)))
    console.print(instr_table)

    quantities_table = Table(title="Quantities")
    quantities_table.add_column("Quantity", style="cyan", no_wrap=True)
    quantities_table.add_column("Count", style="green")
    for qty, count in stats["quantities"].items():
        quantities_table.add_row(qty, str(count))
    console.print(quantities_table)

    # Print files per instrument
    for instr, info in stats["instruments"].items():
        files_table = Table(title=f"Files for instrument {instr}")
        files_table.add_column("File", style="cyan")
        files_table.add_column("Count", style="green")
        for fname, count in info["files"].items():
            files_table.add_row(fname, str(count))
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
