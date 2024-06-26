import sys
from pathlib import Path
from typing import Optional

import click
import netCDF4

from wrf_ensembly import experiment, external, nco, wrf
from wrf_ensembly.click_utils import pass_experiment_path
from wrf_ensembly.console import logger


@click.group(name="postprocess")
def postprocess_cli():
    pass


@postprocess_cli.command()
@click.option(
    "--cycle",
    type=int,
    help="Cycle to compute statistics for. Will compute for all current cycle if missing.",
)
@click.option(
    "--jobs",
    type=click.IntRange(min=0, max=None),
    default=4,
    help="How many NCO commands to execute in parallel",
)
@pass_experiment_path
def statistics(
    experiment_path: Path,
    cycle: Optional[int],
    jobs: int,
):
    """
    Calculates the ensemble mean and standard deviation from the forecast/analysis files of given cycle.
    """

    logger.setup("postprocess-statistics", experiment_path)
    exp = experiment.Experiment(experiment_path)

    if cycle is None:
        cycle = exp.current_cycle_i

    logger.info(f"Cycle: {exp.cycles[cycle]}")

    # An array to collect all commands to run
    commands = []

    # Compute analysis statistics
    scratch_analysis_dir = exp.paths.scratch_analysis_path(cycle)
    analysis_files = list(scratch_analysis_dir.rglob("member_*/wrfout*"))
    if len(analysis_files) != 0:
        analysis_mean_file = scratch_analysis_dir / f"{analysis_files[0].name}_mean"
        analysis_mean_file.unlink(missing_ok=True)
        commands.append(nco.average(analysis_files, analysis_mean_file))

        analysis_sd_file = scratch_analysis_dir / f"{analysis_files[0].name}_sd"
        analysis_sd_file.unlink(missing_ok=True)
        commands.append(nco.standard_deviation(analysis_files, analysis_sd_file))
    else:
        logger.warning("No analysis files found!")

    # Compute forecast statistics
    scratch_forecast_dir = exp.paths.scratch_forecasts_path(cycle)
    forecast_filenames = [
        x.name for x in scratch_forecast_dir.rglob("member_00/wrfout*")
    ]
    for name in forecast_filenames:
        logger.info(f"Computing statistics for {name}")

        forecast_files = list(scratch_forecast_dir.rglob(f"member_*/{name}"))
        if len(forecast_files) == 0:
            logger.warning(f"No forecast files found for {name}!")
            continue

        forecast_mean_file = scratch_forecast_dir / f"{name}_mean"
        forecast_mean_file.unlink(missing_ok=True)
        commands.append(nco.average(forecast_files, forecast_mean_file))

        forecast_sd_file = scratch_forecast_dir / f"{name}_sd"
        forecast_sd_file.unlink(missing_ok=True)
        commands.append(nco.standard_deviation(forecast_files, forecast_sd_file))

    # Execute commands
    failure = False
    logger.info(
        f"Executing {len(commands)} nco commands in parallel, using {jobs} jobs"
    )
    for res in external.run_in_parallel(commands, jobs):
        if res.returncode != 0:
            logger.error(f"nco command failed with exit code {res.returncode}")
            logger.error(res.output)
            failure = True

    if failure:
        logger.error("One or more nco commands failed, exiting")
        sys.exit(1)


@postprocess_cli.command()
@click.option(
    "--cycle",
    type=int,
    help="Cycle to compute statistics for. Will compute for all current cycle if missing.",
)
@click.option(
    "--jobs",
    type=click.IntRange(min=0, max=None),
    default=4,
    help="How many NCO commands to execute in parallel",
)
@pass_experiment_path
def concatenate(
    experiment_path: Path,
    cycle: Optional[int],
    jobs: int,
):
    """
    Concatenates all output files (mean and standard deviation) into two files, one for analysis
    and one for forecast. It uses the `_mean` and `_sd` files created by the `statistics` command.
    """

    logger.setup("postprocess-statistics", experiment_path)
    exp = experiment.Experiment(experiment_path)

    if cycle is None:
        cycle = exp.current_cycle_i

    commands = []

    # Prepare compression related arguments
    cmp_args = []
    if len(exp.cfg.postprocess.ppc_filter) > 0:
        cmp_args = ["--ppc", exp.cfg.postprocess.ppc_filter]
    if len(exp.cfg.postprocess.compression_filters) > 0:
        cmp_args.append(f"--cmp={exp.cfg.postprocess.compression_filters}")

    if len(cmp_args) == 0:
        logger.warning("No compression filters set, output files will be uncompressed")
    else:
        logger.info(f"Using compression filters: {' '.join(cmp_args)}")

    # Find all forecast files
    forecast_dir = exp.paths.forecast_path(cycle)
    scratch_forecast_dir = exp.paths.scratch_forecasts_path(cycle)
    forecast_files = sorted(scratch_forecast_dir.rglob("*_mean"))
    if len(forecast_files) > 0:
        commands.append(
            nco.concatenate(
                forecast_files,
                forecast_dir / f"forecast_mean_cycle_{cycle:03d}.nc",
                cmp_args,
            )
        )
    forecast_files = sorted(scratch_forecast_dir.rglob("*_sd"))
    if len(forecast_files) > 0:
        commands.append(
            nco.concatenate(
                forecast_files,
                forecast_dir / f"forecast_sd_cycle_{cycle:03d}.nc",
                cmp_args,
            )
        )

    # Find all analysis files
    analysis_dir = exp.paths.analysis_path(cycle)
    scratch_analysis_dir = exp.paths.scratch_analysis_path(cycle)
    analysis_files = sorted(scratch_analysis_dir.rglob("*_mean"))
    if len(analysis_files) > 0:
        commands.append(
            nco.concatenate(
                analysis_files,
                analysis_dir / f"analysis_mean_cycle_{cycle:03d}.nc",
                cmp_args,
            )
        )
    analysis_files = sorted(scratch_analysis_dir.rglob("*_sd"))
    if len(analysis_files) > 0:
        commands.append(
            nco.concatenate(
                analysis_files,
                analysis_dir / f"analysis_sd_cycle_{cycle:03d}.nc",
                cmp_args,
            )
        )

    failure = False
    logger.info(
        f"Executing {len(commands)} nco commands in parallel, using {jobs} jobs"
    )
    for res in external.run_in_parallel(commands, jobs):
        if res.returncode != 0:
            logger.error(f"nco command failed with exit code {res.returncode}")
            logger.error(res.output)
            failure = True

    if failure:
        logger.error("One or more nco commands failed, exiting")
        sys.exit(1)


@postprocess_cli.command()
@click.option(
    "--cycle",
    type=int,
    help="Cycle to clean up. Will clean for current cycle if missing.",
)
@click.option(
    "--remove-wrfout",
    default=True,
    is_flag=True,
    help="Remove the raw wrfout files",
)
@pass_experiment_path
def clean(experiment_path: Path, cycle: Optional[int], remove_wrfout: bool):
    """
    Clean up the scratch directory for the given cycle. Use after running the other
    postprocessing commands to save disk space.
    """

    logger.setup("postprocess-statistics", experiment_path)
    exp = experiment.Experiment(experiment_path)

    if cycle is None:
        cycle = exp.current_cycle_i

    logger.info(f"Cleaning scratch for cycle {cycle}")

    scratch_dirs = [
        exp.paths.scratch_forecasts_path(cycle),
        exp.paths.scratch_analysis_path(cycle),
    ]
    for dir in scratch_dirs:
        if remove_wrfout:
            for f in dir.rglob("wrfout*"):
                logger.info(f"Removing {f}")
                f.unlink()
