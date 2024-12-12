import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import click
import netCDF4

from wrf_ensembly import experiment, utils
from wrf_ensembly.click_utils import pass_experiment_path
from wrf_ensembly.console import logger


@click.group(name="ensemble")
def ensemble_cli():
    pass


@ensemble_cli.command()
@pass_experiment_path
def setup(experiment_path: Path):
    """
    Copies initial/boundary conditions for each member.
    """

    logger.setup("ensemble-setup", experiment_path)
    exp = experiment.Experiment(experiment_path)

    first_cycle = exp.cycles[0]
    logger.info(f"Configuring members for cycle 0: {str(first_cycle)}")

    for i in range(exp.cfg.assimilation.n_members):
        member_dir = exp.paths.member_path(i)

        # First check if there are member-specific IC/BC files, otherwise use the same
        # for all members
        ic_file = (
            exp.paths.data_icbc
            / f"member_{i:02d}"
            / f"wrfinput_d01_member_{i:02d}_cycle_0"
        )
        if ic_file.exists():
            logger.info(f"Member {i}: Using member-specific IC file {ic_file}")
        else:
            ic_file = exp.paths.data_icbc / "wrfinput_d01_cycle_0"

        bc_file = (
            exp.paths.data_icbc
            / f"member_{i:02d}"
            / f"wrfbdy_d01_member_{i:02d}_cycle_0"
        )
        if bc_file.exists():
            logger.info(f"Member {i}: Using member-specific BC file {bc_file}")
        else:
            bc_file = exp.paths.data_icbc / "wrfbdy_d01_cycle_0"

        # Copy initial and boundary conditions
        utils.copy(ic_file, member_dir / "wrfinput_d01")
        logger.info(f"Member {i}: Copied wrfinput_d01")

        utils.copy(bc_file, member_dir / "wrfbdy_d01")
        logger.info(f"Member {i}: Copied wrfbdy_d01_cycle_0")


@ensemble_cli.command()
@click.option(
    "--jobs",
    type=click.IntRange(min=0, max=None),
    help="How many files to process in parallel",
)
@pass_experiment_path
def generate_perturbations(experiment_path: Path, jobs: Optional[int]):
    """Generates perturbations for all experiment cycles"""

    logger.setup("ensemble-generate-perturbations", experiment_path)
    exp = experiment.Experiment(experiment_path)

    jobs = utils.determine_jobs(jobs)
    logger.info(f"Using {jobs} jobs")
    logger.info(
        f"Applying perturbations every cycle: {exp.cfg.perturbations.apply_perturbations_every_cycle}"
    )

    if exp.cfg.perturbations.apply_perturbations_every_cycle:
        logger.info("Generating perturbations for all cycles...")
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            res = executor.map(exp.generate_perturbations, range(len(exp.cycles)))
            for _ in res:
                pass
    else:
        logger.info("Generating perturbations for first cycle only...")
        exp.generate_perturbations(0)


@ensemble_cli.command()
@click.option(
    "--jobs",
    type=click.IntRange(min=0, max=None),
    help="How many files to process in parallel",
)
@pass_experiment_path
def apply_perturbations(experiment_path: Path, jobs: Optional[int]):
    """
    Applies perturbations to the initial conditions of the current cycle.
    Make sure to update the boundary conditions afterwards!
    """

    logger.setup("ensemble-apply-perturbations", experiment_path)
    exp = experiment.Experiment(experiment_path)

    jobs = utils.determine_jobs(jobs)
    logger.info(f"Using {jobs} jobs")

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        res = executor.map(
            exp.apply_perturbations, range(exp.cfg.assimilation.n_members)
        )
        for _ in res:
            pass


@ensemble_cli.command()
@click.option(
    "--jobs",
    type=click.IntRange(min=0, max=None),
    help="How many files to process in parallel",
)
@pass_experiment_path
def update_bc(experiment_path: Path, jobs: Optional[int]):
    """
    Runs `update_wrf_bc` for all members to update boundary conditions.
    Use this after you have modified the initial conditions (perts or cycling).
    """

    logger.setup("ensemble-update-bc", experiment_path)
    exp = experiment.Experiment(experiment_path)
    exp.set_dart_environment()

    jobs = utils.determine_jobs(jobs)
    logger.info(f"Using {jobs} jobs")

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        results = executor.map(
            exp.update_bc,
            range(exp.cfg.assimilation.n_members),
        )

        for _ in results:
            pass


@ensemble_cli.command()
@click.option("--member", required=True, type=int, help="Which member to advance")
@click.option(
    "--cores",
    type=int,
    help="Number of cores to use for wrf.exe. ",
)
@pass_experiment_path
def advance_member(
    experiment_path: Path,
    member: int,
    cores: int,
):
    """
    Advances the given MEMBER 1 cycle by running the model

    You can control how many cores to use with --cores. If omitted, will check for
    `SLURM_NTASKS` in the environment and use that. If missing, will use 1 core.
    """

    logger.setup(f"ensemble-advance-member_{member}", experiment_path)
    exp = experiment.Experiment(experiment_path)
    exp.set_wrf_environment()
    if member < 0 or member >= exp.cfg.assimilation.n_members:
        logger.error(f"Member {member} does not exist")
        sys.exit(1)

    # Determine number of cores
    if cores is None:
        if "SLURM_NTASKS" in os.environ:
            cores = int(os.environ["SLURM_NTASKS"])
        else:
            cores = 1
            logger.warning("No --cores no SLURM_NTASKS, will use 1 core!")
    logger.info(f"Using {cores} cores for wrf.exe")

    # Run WRF!
    success = exp.advance_member(member, cores=cores)
    if not success:
        sys.exit(1)


@ensemble_cli.command()
@pass_experiment_path
def filter(experiment_path: Path):
    """
    Runs the assimilation filter for the current cycle
    """

    logger.setup("ensemble-filter", experiment_path)
    exp = experiment.Experiment(experiment_path)
    exp.set_dart_environment()
    if exp.filter():
        exp.write_status()


@ensemble_cli.command()
@pass_experiment_path
def analysis(experiment_path: Path):
    """
    Combines the DART output files and the forecast to create the analysis.
    """

    logger.setup("ensemble-analysis", experiment_path)
    exp = experiment.Experiment(experiment_path)

    if not exp.filter_run:
        logger.error("Filter has not been run, cannot run analysis!")
        sys.exit(1)

    cycle_i = exp.current_cycle_i
    cycle = exp.current_cycle

    forecast_dir = exp.paths.scratch_forecasts_path(cycle_i)
    analysis_dir = exp.paths.scratch_analysis_path(cycle_i)
    dart_out_dir = exp.paths.scratch_dart_path(cycle_i)

    # Postprocess analysis files
    for member in range(exp.cfg.assimilation.n_members):
        # Copy forecasts to analysis directory
        wrfout_name = "wrfout_d01_" + cycle.end.strftime("%Y-%m-%d_%H:%M:%S")
        forecast_file = forecast_dir / f"member_{member:02d}" / wrfout_name
        analysis_file = analysis_dir / f"member_{member:02d}" / wrfout_name
        utils.copy(forecast_file, analysis_file)

        dart_file = dart_out_dir / f"dart_member_{member:02d}.nc"
        if not dart_file.exists():
            logger.error(f"Member {member}: {dart_file} does not exist")
            sys.exit(1)

        # Copy the state variables from the dart file to the analysis file
        logger.info(f"Member {member}: Copying state variables from {dart_file}")
        with (
            netCDF4.Dataset(dart_file, "r") as nc_dart,  # type: ignore
            netCDF4.Dataset(analysis_file, "r+") as nc_analysis,  # type: ignore
        ):
            for name in exp.cfg.assimilation.state_variables:
                if name not in nc_dart.variables:
                    logger.warning(f"Member {member}: {name} not in dart file")
                    continue
                logger.info(f"Member {member}: Copying {name}")
                nc_analysis[name][:] = nc_dart[name][:]

            # Add experiment name and current cycle information to attributes
            # TODO Standardize this somehow? We must add metadata to all files!
            nc_analysis.experiment_name = exp.cfg.metadata.name
            nc_analysis.current_cycle = cycle_i
            nc_analysis.cycle_start = cycle.start.strftime("%Y-%m-%d_%H:%M:%S")
            nc_analysis.cycle_end = cycle.end.strftime("%Y-%m-%d_%H:%M:%S")

    # Update experiment status
    exp.analysis_run = True
    exp.write_status()


@ensemble_cli.command()
@click.option(
    "--use-forecast",
    is_flag=True,
    help="Cycle with the latest forecast instead of the analysis",
)
@click.option(
    "--jobs",
    type=click.IntRange(min=0, max=None),
    help="How many files to process in parallel",
)
@pass_experiment_path
def cycle(experiment_path: Path, use_forecast: bool, jobs: Optional[int]):
    """
    Prepares the experiment for the next cycle by copying the cycled variables from the analysis
    to the initial conditions and preparing the namelist.
    """

    logger.setup("cycle", experiment_path)
    exp = experiment.Experiment(experiment_path)

    if not exp.all_members_advanced:
        logger.error("Not all members have advanced to the next cycle, cannot cycle!")
        sys.exit(1)
    if not use_forecast and not exp.analysis_run:
        logger.error(
            "Analysis step is not done for this cycle, either run it or use --use-forecast to cycle w/ the latest forecast"
        )
        sys.exit(1)

    if use_forecast:
        logger.warning("Cycling using the latest forecast")

    cycle_i = exp.current_cycle_i
    next_cycle_i = cycle_i + 1

    if next_cycle_i >= len(exp.cycles):
        logger.error(f"Experiment is finished! No cycle {next_cycle_i}")
        sys.exit(1)

    # Determine job count
    jobs = utils.determine_jobs(jobs)
    logger.info(f"Using {jobs} jobs")

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        results = executor.map(
            exp.cycle_member,
            range(exp.cfg.assimilation.n_members),
            [use_forecast] * exp.cfg.assimilation.n_members,
        )

        for _ in results:
            pass

    # Update experiment status
    exp.set_next_cycle()
    exp.write_status()


@ensemble_cli.command()
@pass_experiment_path
def status(experiment_path: Path):
    """Prints the current status of all members (i.e., which cycles have been completed)"""

    logger.setup("experiment-status", experiment_path)
    exp = experiment.Experiment(experiment_path)

    logger.info(f"Current cycle: {exp.current_cycle_i}")
    logger.info(f"Members: {exp.cfg.assimilation.n_members}")
    logger.info(f"Filter run: {exp.filter_run}")
    logger.info(f"Analysis run: {exp.analysis_run}")

    for member in exp.members:
        logger.info(f"Member {member.i} advanced: {member.advanced}")
