"""
Commands about handling observations in the context of an experiment (adding, retrieving, etc).
"""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from wrf_ensembly import experiment, external, observations
from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.console import logger
from wrf_ensembly.utils import determine_jobs


@click.group(name="observations", cls=GroupWithStartEndPrint)
def observations_cli():
    """Commands related to handling observations in the context of an experiment"""
    pass


@observations_cli.command()
@click.argument("files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@pass_experiment_path
def add(experiment_path: Path, files: list[Path]):
    """Bulk add observations to the experiment"""

    logger.setup("observations-convert-obs", experiment_path)
    exp = experiment.Experiment(experiment_path)

    for file_path in files:
        exp.obs.add_observation_file(file_path)


@observations_cli.command()
@pass_experiment_path
def show(experiment_path: Path):
    exp = experiment.Experiment(experiment_path)
    obs_files = exp.obs.get_available_observation_files()

    table = Table(title="Available Observation Files")
    table.add_column("Instrument", style="cyan", no_wrap=True)
    table.add_column("Start Time", style="green")
    table.add_column("End Time", style="green")
    table.add_column("Path", style="magenta")

    for obs_file in obs_files:
        table.add_row(
            obs_file.instrument,
            obs_file.start_time.strftime("%Y-%m-%d %H:%M"),
            obs_file.end_time.strftime("%Y-%m-%d %H:%M"),
            str(obs_file.path),
        )

    Console().print(table)


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
def convert_to_dart(
    experiment_path: Path,
    cycle: int | None = None,
    jobs: int | None = None,
    skip_dart: bool = False,
):
    """
    Converts the experiment's observation files to DART obs_seq format.
    Required for `filter` to be able to use the observations.

    You must build the `wrf_ensembly` observation converter in DART for this to work,
    check the `DART/observations/obs_converters/wrf_ensembly` directory.
    """

    logger.setup("observations-convert-to-dart", experiment_path)
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
