import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click
import netCDF4

from wrf_ensembly import config, experiment, perturbations, utils
from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.console import logger
from wrf_ensembly.experiment import ExperimentStateError, StateTransition


@click.group(name="ensemble", cls=GroupWithStartEndPrint)
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

        # Copy initial and boundary conditions
        utils.copy(exp.paths.ic_path(i, 0), member_dir / "wrfinput_d01")
        exp.prepare_member_boundaries(i, 0)
        logger.info(f"Member {i}: Copied initial and boundary conditions")


@ensemble_cli.command()
@click.argument(
    "other-experiment", type=click.Path(dir_okay=True, file_okay=False, path_type=Path)
)
@click.option(
    "--cycle", type=int, required=True, help="Which cycle to use for initialisation"
)
@pass_experiment_path
def setup_from_other_experiment(
    experiment_path: Path, other_experiment: Path, cycle: int
):
    """
    Setup the ensemble using a forecast from another experiment.

    The usecase for this command is having a control experiment that starts earlier, for
    spin-up reasons. You can initialise a second experiment from a mid-point and work
    forwards.

    The other experiment must have the same cycle setup (start/end dates, output interval,
    boundary conditions interval), domain and cycling mode. The forecast files for the
    requested cycle must be available in the scratch directory. In restart mode, its
    restart files at the end of the cycle are copied too, so they must still exist (see
    `assimilation.keep_restart_files_for_cycles`).
    """

    logger.setup("ensemble-setup-from-other-experiment", experiment_path)
    exp = experiment.Experiment(experiment_path)

    logger.info(f"Opening second experiment at {other_experiment}")
    other_exp = experiment.Experiment(other_experiment.resolve())

    # A restart-mode experiment continues from restart files, which a wrfinput-mode one
    # doesn't write. The other way around, there are no wrfinput files for later cycles.
    mode = exp.cfg.assimilation.cycling_mode
    other_mode = other_exp.cfg.assimilation.cycling_mode
    if mode != other_mode:
        logger.error(
            f'This experiment has cycling_mode = "{mode}" but the other one "{other_mode}", '
            "they must be the same"
        )
        sys.exit(1)

    # Check if the domain and time control setups are the same
    res = exp.cfg.domain_control.is_equal(other_exp.cfg.domain_control)
    if type(res) is str:
        logger.error(f"Field is not equal in [domain_control]: {res}")
        sys.exit(1)
    res = exp.cfg.time_control.is_equal(other_exp.cfg.time_control)
    if type(res) is str:
        logger.error(f"Field is not equal in [time_control]: {res}")
        sys.exit(1)
    if exp.cfg.time_control.cycles != {} or other_exp.cfg.time_control.cycles != {}:
        logger.warning(
            "[time_control.cycles] is defined in at least one of the experiments. Be careful!"
        )
    if exp.cfg.assimilation.n_members != other_exp.cfg.assimilation.n_members:
        logger.error("Experiments do not have the same amount of members")
        sys.exit(1)

    # In restart mode the members continue from the other experiment's restart files at
    # the end of the cycle. Check they are all there before changing anything.
    cycle_end = other_exp.cycles[cycle].end
    restart_files = []
    if mode == "restart":
        restart_files = [
            other_exp.paths.scratch_restart_path(cycle, i)
            / f"wrfrst_d01_{cycle_end:%Y-%m-%d_%H:%M:%S}"
            for i in range(exp.cfg.assimilation.n_members)
        ]
        missing = [f for f in restart_files if not f.exists()]
        if missing:
            logger.error(
                f"{len(missing)} restart file(s) of cycle {cycle} are missing from the other "
                f"experiment, e.g. {missing[0]}. `cycle` deletes old restart files unless "
                f"the cycle is in assimilation.keep_restart_files_for_cycles (or "
                f"keep_restart_files is set)"
            )
            sys.exit(1)

    # Namelist differences are allowed (e.g. emission tuning), but a restart takes the
    # model state from the other experiment, so physics or chemistry changes may not
    # work. Point them out.
    differences = config.wrf_namelist_differences(exp.cfg, other_exp.cfg)
    if differences:
        shown = differences[:20]
        more = (
            f"\n  ... and {len(differences) - 20} more" if len(differences) > 20 else ""
        )
        logger.warning(
            "The WRF namelist differs from the other experiment's (this / other):\n  "
            + "\n  ".join(shown)
            + more
        )

    # Link other experiments IC/BC directory to current exp.
    icbc_dir = exp.paths.data_icbc
    logger.info(
        f"Removing current IC/BC directory {icbc_dir} and linking to {other_exp.paths.data_icbc}"
    )
    if icbc_dir.is_dir():
        icbc_dir.rmdir()
    elif icbc_dir.is_symlink():
        icbc_dir.unlink()
    icbc_dir.symlink_to(other_exp.paths.data_icbc, target_is_directory=True)

    # Check if the required forecasts exist in the scratch directory & link them in the current experiment
    required_wrfout_filename = f"wrfout_d01_{cycle_end:%Y-%m-%d_%H:%M:%S}"
    logger.info(f"Required forecast filename: {required_wrfout_filename}")

    for i in range(exp.cfg.assimilation.n_members):
        target_scratch = other_exp.paths.scratch_forecasts_path(cycle=cycle, member=i)
        required_wrfout_file = (target_scratch / required_wrfout_filename).resolve()
        if not required_wrfout_file.exists():
            logger.error(f"Forecast doesn't exist at {required_wrfout_file}")
            sys.exit(1)

        new_dir = exp.paths.scratch_forecasts_path(cycle=cycle, member=i)
        new_dir.mkdir(exist_ok=True, parents=True)

        symlink_loc = new_dir / required_wrfout_filename
        if symlink_loc.exists() and not symlink_loc.is_symlink():
            logger.error(
                f"File already exists at {symlink_loc} and is not a symlink. Too scared to replace."
            )
            sys.exit(1)
        symlink_loc.unlink(missing_ok=True)
        logger.info(f"Linking {symlink_loc} to {required_wrfout_file}")
        symlink_loc.symlink_to(required_wrfout_file)

        # The restart file is copied, not linked, so the other experiment can delete
        # its own and this one can modify its copy
        if restart_files:
            target = exp.paths.scratch_restart_path(cycle, i) / restart_files[i].name
            logger.info(f"Copying {restart_files[i]} to {target}")
            utils.copy(restart_files[i], target)

        # Set member as advanced
        exp.members[i].advanced = True
        exp.state.set_member_advanced(cycle, i)

    # Update experiment status & metadata
    exp.current_cycle_i = cycle

    logger.info(f"Linked to {other_exp}, cycle = {cycle}.")
    logger.info("Use the `cycle` command to advance this experiment to the next cycle")


@ensemble_cli.command()
@click.option(
    "--jobs",
    type=click.IntRange(min=0, max=None),
    help="How many files to process in parallel",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force regenerating perturbations even if already done",
)
@pass_experiment_path
def generate_perturbations(experiment_path: Path, jobs: int | None, force: bool):
    """
    Generates perturbations for all experiment cycles.

    This operation is tracked per-cycle to prevent regeneration.
    Use --force to override if needed.
    """

    logger.setup("ensemble-generate-perturbations", experiment_path)
    exp = experiment.Experiment(experiment_path)

    operation_name = "generate_perturbations"

    jobs = utils.determine_jobs(jobs)
    logger.info(f"Using {jobs} jobs")

    # Check if perturbations for cycle 0 have already been generated
    cycle_0_done = exp.state.is_optional_operation_complete(0, operation_name)

    if cycle_0_done and not force:
        logger.warning("Perturbations already generated for all cycles")
        logger.warning("Use --force to regenerate them")
        sys.exit(1)

    if cycle_0_done and force:
        logger.warning("Forcing regeneration of perturbations")

    # First, generate the perturbations for the first cycle, because they might be needed
    # for the other cycles too (`kind = "parameter"` in wrfinput cycling mode)
    logger.info("Generating perturbations for first cycle...")
    exp.generate_perturbations(0)

    # Mark cycle 0 as done
    exp.state.mark_optional_operation_complete(0, operation_name)

    # Now, if required, generate perturbations for all cycles. Every cycle after the first
    # perturbs the same variables.
    if len(exp.cycles) > 1 and perturbations.applied_at_cycle(exp.cfg, 1):
        logger.info("Generating perturbations for all cycles...")
        for cycle_i in exp.generate_perturbations_for_cycles(
            range(1, len(exp.cycles)), jobs
        ):
            exp.state.mark_optional_operation_complete(cycle_i, operation_name)
    else:
        # Nothing is perturbed after the first cycle. Remove files left from an earlier
        # configuration, since `apply-perturbations` runs whenever a cycle has one.
        for cycle_i in range(1, len(exp.cycles)):
            pert_file = (
                exp.paths.data_diag / "perturbations" / f"perts_cycle_{cycle_i}.nc"
            )
            if pert_file.exists() or pert_file.is_symlink():
                logger.info(
                    f"Removing {pert_file}, nothing is perturbed at cycle {cycle_i}"
                )
                pert_file.unlink()

    logger.info("Perturbation generation complete")


@ensemble_cli.command()
@click.option(
    "--jobs",
    type=click.IntRange(min=0, max=None),
    help="How many files to process in parallel",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force applying perturbations even if already done for this cycle",
)
@pass_experiment_path
def apply_perturbations(experiment_path: Path, jobs: int | None, force: bool):
    """
    Applies perturbations to the initial conditions of the current cycle.
    Make sure to update the boundary conditions afterwards!

    This operation is tracked per-cycle to prevent double-perturbation.
    Use --force to override if needed.
    """

    logger.setup("ensemble-apply-perturbations", experiment_path)
    exp = experiment.Experiment(experiment_path)

    # Check if perturbations have already been applied for this cycle
    operation_name = "apply_perturbations"
    already_done = exp.state.is_optional_operation_complete(
        exp.current_cycle_i, operation_name
    )

    if already_done and not force:
        logger.warning(f"Perturbations already applied for cycle {exp.current_cycle_i}")
        logger.warning(
            "Use --force to apply them again (may cause double-perturbation)"
        )
        sys.exit(1)

    if already_done and force:
        logger.warning("Forcing re-application of perturbations")

    jobs = utils.determine_jobs(jobs)
    logger.info(f"Using {jobs} jobs")

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        res = executor.map(
            exp.apply_perturbations, range(exp.cfg.assimilation.n_members)
        )
        for _ in res:
            pass

    # Mark as completed
    exp.state.mark_optional_operation_complete(exp.current_cycle_i, operation_name)

    logger.info(f"Perturbations applied for cycle {exp.current_cycle_i}")


@ensemble_cli.command()
@click.option(
    "--jobs",
    type=click.IntRange(min=0, max=None),
    help="How many files to process in parallel",
)
@pass_experiment_path
def update_bc(experiment_path: Path, jobs: int | None):
    """
    Updates the boundary conditions of all members to match their initial conditions.
    Use this after you have modified the initial conditions (perts or cycling).
    """

    logger.setup("ensemble-update-bc", experiment_path)
    exp = experiment.Experiment(experiment_path)

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

    can_advance, error = exp.state_machine.can_advance_member(
        exp.current_cycle_i, member
    )
    if not can_advance:
        logger.error(f"Cannot advance member {member}: {error}")
        logger.info(
            f"Next required action: {exp.state_machine.current_cycle.get_required_actions()}"
        )
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

    # Nothing to transition: the cycle state is derived from the member files, so it
    # becomes MEMBERS_ADVANCED on its own once the last member writes its own file.
    n_advanced = exp.state.count_advanced(exp.current_cycle_i)
    logger.info(f"{n_advanced}/{exp.cfg.assimilation.n_members} members advanced")
    if n_advanced >= exp.cfg.assimilation.n_members:
        logger.info("All members advanced - ready for filter")


@ensemble_cli.command()
@pass_experiment_path
def filter(experiment_path: Path):
    """
    Runs the assimilation filter for the current cycle
    """

    logger.setup("ensemble-filter", experiment_path)
    exp = experiment.Experiment(experiment_path)
    exp.set_dart_environment()

    try:
        success = exp.filter()
    except ExperimentStateError as e:
        logger.error(str(e))
        sys.exit(1)

    if success:
        logger.info("Filter complete - ready for analysis")


@ensemble_cli.command()
@pass_experiment_path
def analysis(experiment_path: Path):
    """
    Combines the DART output files and the forecast to create the analysis.
    """

    logger.setup("ensemble-analysis", experiment_path)
    exp = experiment.Experiment(experiment_path)

    can_run, error = exp.state_machine.can_run_analysis(exp.current_cycle_i)
    if not can_run:
        logger.error(error)
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

    exp.state_machine.current_cycle.transition(StateTransition.ANALYSIS_COMPLETE)
    logger.info("Analysis complete - ready to cycle")


@ensemble_cli.command()
@click.option(
    "--jobs",
    type=click.IntRange(min=0, max=None),
    help="How many files to process in parallel",
)
@pass_experiment_path
def cycle(experiment_path: Path, jobs: int | None):
    """
    Prepares the experiment for the next cycle by copying the cycled variables from the analysis
    to the initial conditions and preparing the namelist.
    """

    logger.setup("cycle", experiment_path)
    exp = experiment.Experiment(experiment_path)

    if not exp.wait_for_all_members():
        n_advanced = exp.state.count_advanced(exp.current_cycle_i)
        logger.error(
            f"Only {n_advanced}/{exp.cfg.assimilation.n_members} members have advanced, "
            "cannot cycle!"
        )
        sys.exit(1)

    can_cycle, use_forecast, error = exp.state_machine.can_cycle_to_next(
        exp.current_cycle_i
    )
    if not can_cycle:
        logger.error(error)
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

    # Do the cycling work
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        results = executor.map(
            exp.cycle_member,
            range(exp.cfg.assimilation.n_members),
            [use_forecast] * exp.cfg.assimilation.n_members,
        )

        for _ in results:
            pass

    exp.state_machine.current_cycle.transition(StateTransition.CYCLE_COMPLETE)

    # Move the experiment pointer to the next cycle, which starts out empty
    exp.set_next_cycle()
    logger.info(f"Cycled to cycle {next_cycle_i}")

    exp.clean_restart_files()


@ensemble_cli.command()
@click.option(
    "--cycle",
    type=int,
    help="Which cycle to reset (defaults to current cycle)",
)
@pass_experiment_path
def reset_cycle(experiment_path: Path, cycle: int | None):
    """
    Reset the cycle state to INITIALIZED.

    This command allows you to restart a cycle from the beginning by resetting
    the state machine back to INITIALIZED. Use this when you need to re-run a
    cycle after fixing issues.

    WARNING: This does not clean up any files or undo any work. It only resets
    the state tracking. You may need to manually clean up files before re-running
    the cycle.
    """

    logger.setup("ensemble-reset-cycle", experiment_path)
    exp = experiment.Experiment(experiment_path)

    cycle_idx = cycle if cycle is not None else exp.current_cycle_i

    if cycle_idx < 0 or cycle_idx >= len(exp.cycles):
        logger.error(f"Invalid cycle index {cycle_idx}")
        sys.exit(1)

    cycle_machine = exp.state_machine.get_cycle(cycle_idx)
    logger.warning(
        f"Resetting cycle {cycle_idx} from state {cycle_machine.current_state.value} to INITIALIZED"
    )

    # Removes the cycle's markers and member files, affecting only this cycle
    cycle_machine.reset()

    logger.info(f"Cycle {cycle_idx} reset to INITIALIZED state")
    logger.info("You may need to manually clean up files before re-running this cycle")
