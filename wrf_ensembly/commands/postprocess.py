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
    default=None,
    help="Cycle to extract the variables from. If not provided, will extract from the current cycle.",
)
@pass_experiment_path
def extract_vars(experiment_path: Path, cycle: Optional[int]):
    """
    Extract a set of desired variables from forecast/analysis wrfout files to create
    smaller files.

    The extracted variables are defined in the configuration file (`postprocess.extract_vars`).

    For each wrfout file in the scratch directory, a new file will be created with the
    `_small` prefix that only contains the desired variables.
    """

    logger.setup("postprocess-extract-vars", experiment_path)
    exp = experiment.Experiment(experiment_path)

    # Sanity checking in regards to the cycle
    if cycle is None:
        cycle = exp.current_cycle_i

    if cycle > exp.current_cycle_i:
        logger.error(f"Cycle {cycle} is in the future, cannot extract variables.")
        sys.exit(1)
    if cycle < 0 or cycle >= len(exp.cycles):
        logger.error(f"Cycle {cycle} is out of bounds, cannot extract variables.")
        sys.exit(1)
    if cycle == exp.current_cycle_i and not exp.all_members_advanced:
        logger.error("Not all members have advanced, cannot extract variables.")
        sys.exit(1)

    # Create set of required variables based on config
    required_vars = set(exp.cfg.postprocess.extract_vars) | wrf.ESSENTIAL_VARIABLES

    # Find all files to process in the given cycle
    fc_dir = exp.paths.scratch_forecasts_path(cycle)
    an_dir = exp.paths.scratch_analysis_path(cycle)
    files = list(fc_dir.rglob("**/wrfout*")) + list(an_dir.rglob("**/wrfout*"))
    for in_path in files:
        member_i = in_path.parent.name.split("_")[-1]
        out_path = in_path.parent / f"{in_path.stem}_small"
        logger.info(
            f"Extracting variables from {in_path} (member {member_i}) into {out_path}"
        )

        with (
            netCDF4.Dataset(in_path, "r") as ds_in,  # type: ignore
            netCDF4.Dataset(out_path, "w") as ds_out,  # type: ignore
        ):
            # Add some metadata to the attributes
            ds_out.setncattr("wrf_ensembly_experiment_name", exp.cfg.metadata.name)
            ds_out.setncattr("wrf_ensembly_cycle", cycle)
            ds_out.setncattr("wrf_ensembly_member", member_i)

            # Copy global attributes
            ds_out.setncatts({k: ds_in.getncattr(k) for k in ds_in.ncattrs()})

            # Copy dimensions
            for name, dim in ds_in.dimensions.items():
                ds_out.createDimension(
                    name, len(dim) if not dim.isunlimited() else None
                )

            # Copy variables
            for name, var in ds_in.variables.items():
                if name in required_vars:
                    out_var = ds_out.createVariable(name, var.datatype, var.dimensions)
                    out_var.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                    out_var[:] = var[:]


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
    It uses the `_small` files created by the `extract_vars` command.
    """

    logger.setup("postprocess-statistics", experiment_path)
    exp = experiment.Experiment(experiment_path)
    if not exp.all_members_advanced:
        logger.error(
            "Not all members have advanced to the next cycle, cannot run statistics without at least forecasts!"
        )
        sys.exit(1)

    if cycle is None:
        cycle = exp.current_cycle_i

    logger.info(f"Cycle: {exp.cycles[cycle]}")

    # An array to collect all commands to run
    commands = []

    # Compute analysis statistics
    scratch_analysis_dir = exp.paths.scratch_analysis_path(cycle)
    analysis_dir = exp.paths.analysis_path(cycle)
    analysis_files = list(scratch_analysis_dir.rglob("member_*/wrfout*_small"))
    if len(analysis_files) != 0:
        analysis_mean_file = analysis_dir / f"{analysis_files[0].name}_mean"
        analysis_mean_file.unlink(missing_ok=True)
        commands.append(nco.average(analysis_files, analysis_mean_file))

        analysis_sd_file = analysis_dir / f"{analysis_files[0].name}_sd"
        analysis_sd_file.unlink(missing_ok=True)
        commands.append(nco.standard_deviation(analysis_files, analysis_sd_file))
    else:
        logger.warning("No analysis files found!")

    # Compute forecast statistics
    scratch_forecast_dir = exp.paths.scratch_forecasts_path(cycle)
    forecast_dir = exp.paths.forecast_path(cycle)
    forecast_filenames = [
        x.name for x in scratch_forecast_dir.rglob("member_00/wrfout*_small")
    ]
    for name in forecast_filenames:
        logger.info(f"Computing statistics for {name}")

        forecast_files = list(scratch_forecast_dir.rglob(f"member_*/{name}"))
        if len(forecast_files) == 0:
            logger.warning(f"No forecast files found for {name}!")
            continue

        forecast_mean_file = forecast_dir / f"{name}_mean"
        forecast_mean_file.unlink(missing_ok=True)
        commands.append(nco.average(forecast_files, forecast_mean_file))

        forecast_sd_file = forecast_dir / f"{name}_sd"
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
def concat(
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

    # Find all forecast files
    forecast_dir = exp.paths.forecast_path(cycle)
    forecast_files = sorted(forecast_dir.rglob("*_mean"))
    if len(forecast_files) > 0:
        commands.append(
            nco.concatenate(
                forecast_files, forecast_dir / f"forecast_mean_cycle_{cycle:03d}.nc"
            )
        )
    forecast_files = sorted(forecast_dir.rglob("*_sd"))
    if len(forecast_files) > 0:
        commands.append(
            nco.concatenate(
                forecast_files, forecast_dir / f"forecast_sd_cycle_{cycle:03d}.nc"
            )
        )

    # Find all analysis files
    analysis_dir = exp.paths.analysis_path(cycle)
    analysis_files = sorted(analysis_dir.rglob("*_mean"))
    if len(analysis_files) > 0:
        commands.append(
            nco.concatenate(
                analysis_files, analysis_dir / f"analysis_mean_cycle_{cycle:03d}.nc"
            )
        )
    analysis_files = sorted(analysis_dir.rglob("*_sd"))
    if len(analysis_files) > 0:
        commands.append(
            nco.concatenate(
                analysis_files, analysis_dir / f"analysis_sd_cycle_{cycle:03d}.nc"
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
@click.option(
    "--remove-small",
    default=True,
    is_flag=True,
    help="Remove the small wrfout files (_small)",
)
@pass_experiment_path
def clean(
    experiment_path: Path, cycle: Optional[int], remove_wrfout: bool, remove_small: bool
):
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
        if remove_small:
            for f in dir.rglob("wrfout*_small"):
                logger.info(f"Removing {f}")
                f.unlink()
