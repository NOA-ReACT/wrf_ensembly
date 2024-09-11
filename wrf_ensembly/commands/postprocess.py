import concurrent.futures
import sys
from itertools import chain
from pathlib import Path
from typing import Optional

import click

from wrf_ensembly import experiment, external, nco, postprocess, utils
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
    help="How many files to process in parallel",
)
@pass_experiment_path
def wrf_post(experiment_path: Path, cycle: Optional[int], jobs: int):
    """
    Does some basic postprocessing on the wrfout files:
    - Make units pint friendly
    - Rename dimensions to (t, x, y, z)
    - Destagger variables
    - Compute derived variables such as air temperature and earth-relative wind speed
    - Computes X and Y arrays in the model's projection for interpolation purposes
    Essentially uses the excellent [xwrf](https://github.com/xarray-contrib/xwrf) to make the files a bit more CF-compliant.

    This function is applied to the wrfout files after statistics (mean and standard deviation) have been computed,
    but in principle could be applies to a single wrfout file as well. We do it this way to cut down on time and I/O.
    """

    logger.setup("postprocess-wrf_post", experiment_path)
    exp = experiment.Experiment(experiment_path)

    if cycle is None:
        cycle = exp.current_cycle_i

    logger.info(f"Cycle: {exp.cycles[cycle]}")

    files_to_process = []

    # Find all forecast files
    scratch_forecast_dir = exp.paths.scratch_forecasts_path(cycle)
    forecast_files = chain(
        scratch_forecast_dir.glob("wrfout*_mean"),
        scratch_forecast_dir.glob("wrfout*_sd"),
    )
    for f in forecast_files:
        output_path = scratch_forecast_dir / f"{f.name}_post"
        if output_path.exists():
            output_path.unlink()
        files_to_process.append((f, output_path))

    # Find all analysis files
    scratch_analysis_dir = exp.paths.scratch_analysis_path(cycle)
    analysis_files = chain(
        scratch_analysis_dir.glob("wrfout*_mean"),
        scratch_analysis_dir.glob("wrfout*_sd"),
    )
    for f in analysis_files:
        output_path = scratch_analysis_dir / f"{f.name}_post"
        if output_path.exists():
            output_path.unlink()
        files_to_process.append((f, output_path))

    # Execute commands
    logger.info(
        f"Processing {len(files_to_process)} files in parallel, using {jobs} jobs"
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        executor.map(lambda args: postprocess.xwrf_post(*args), files_to_process)

    # Move files to the correct location
    for old, new in files_to_process:
        logger.debug(f"Moving {new} to {old}")

        old.unlink()
        new.rename(old)


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
    help="How many commands to execute in parallel",
)
@click.option(
    "--keep-temp",
    is_flag=True,
    help="Keep temporary files after processing (scratch/postprocessing)",
)
@pass_experiment_path
def apply_scripts(
    experiment_path: Path, cycle: Optional[int], jobs: int, keep_temp: bool
):
    """
    Apply postprocessing scripts to the output files.
    The scripts are defined in the `PostprocessConfig:scripts` variable of the
    configuration file.
    """

    logger.setup("postprocess-wrf_post", experiment_path)
    exp = experiment.Experiment(experiment_path)

    if cycle is None:
        cycle = exp.current_cycle_i

    logger.info(f"Cycle: {exp.cycles[cycle]}")

    # Gather all the files for processing
    scratch_forecast_dir = exp.paths.scratch_forecasts_path(cycle)
    scratch_analysis_dir = exp.paths.scratch_analysis_path(cycle)
    files_to_process = list(scratch_analysis_dir.glob("wrfout*")) + list(
        scratch_forecast_dir.glob("wrfout*")
    )
    logger.info(f"Found {len(files_to_process)} files to process")

    # Create a scratch dir. for temporary files
    scratch_dir = exp.paths.scratch / "postprocess"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Prepare commands and a directory for each file
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        for i, f in enumerate(files_to_process):
            workdir = scratch_dir / f"{i}"
            workdir.mkdir(exist_ok=True)

            results.append(
                executor.submit(
                    postprocess.apply_scripts_to_file,
                    exp.cfg.postprocess.scripts,
                    f,
                    workdir,
                )
            )

        # If finished successfully, move the files back
        results = [x for x in concurrent.futures.as_completed(results)]

    # Only move files if all were successful
    # It would throw an error is any of the commands failed, so by this point we know all were successful
    try:
        for res in results:
            orig_path, target_path = res.result()
            logger.debug(f"Moving {target_path} to {orig_path}")
            orig_path.unlink()
            target_path.rename(orig_path)
    except external.ExternalProcessFailed:
        logger.error(
            "At least one of the postprocessing scripts failed, files are NOT modified!"
        )

    # Clean up the scratch directory
    if not keep_temp:
        logger.debug(f"Removing temp. directory {scratch_dir}")
        utils.rm_tree(scratch_dir)


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
