import os
import sys
from pathlib import Path
from typing import Optional

import click
import netCDF4
import numpy as np

from wrf_ensembly import experiment, pertubations, update_bc, utils
from wrf_ensembly.click_utils import pass_experiment_path
from wrf_ensembly.console import logger


@click.group(name="ensemble")
def ensemble_cli():
    pass


@ensemble_cli.command()
@pass_experiment_path
def setup(experiment_path: Path):
    """
    Generates namelists and copies initial/boundary conditions for each member.
    """

    logger.setup("ensemble-setup", experiment_path)
    exp = experiment.Experiment(experiment_path)

    first_cycle = exp.cycles[0]
    logger.info(f"Configuring members for cycle 0: {str(first_cycle)}")

    for i in range(exp.cfg.assimilation.n_members):
        member_dir = exp.paths.member_path(i)

        # Copy initial and boundary conditions
        utils.copy(
            exp.paths.data_icbc / f"wrfinput_d01_cycle_0",
            member_dir / "wrfinput_d01",
        )
        logger.info(f"Member {i}: Copied wrfinput_d01")

        utils.copy(
            exp.paths.data_icbc / f"wrfbdy_d01_cycle_0",
            member_dir / "wrfbdy_d01",
        )
        logger.info(f"Member {i}: Copied wrfbdy_d01_cycle_0")


@ensemble_cli.command()
@pass_experiment_path
def apply_pertubations(
    experiment_path: Path,
):
    """
    Applies the configured pertubations to the initial conditions of each ensemble member
    """

    logger.setup("ensemble-apply-pertubations", experiment_path)
    exp = experiment.Experiment(experiment_path)
    cfg = exp.cfg

    if len(cfg.pertubations.variables) == 0:
        logger.info("No pertubations configured.")
        return 0

    if cfg.pertubations.seed is not None:
        logger.warning(f"Setting numpy random seed to {cfg.pertubations.seed}")
        np.random.seed(cfg.pertubations.seed)

    perts_nc_path = exp.paths.data_diag / "pertubations.nc"
    perts_nc_path.unlink(missing_ok=True)
    with netCDF4.Dataset(perts_nc_path, "w") as perts_nc:  # type: ignore
        perts_nc.createDimension("member", cfg.assimilation.n_members)

        for i in range(cfg.assimilation.n_members):
            member_dir = exp.paths.member_path(i)
            wrfinput_path = member_dir / "wrfinput_d01"
            wrfbdy_path = member_dir / "wrfbdy_d01"

            wrfinput_copy_path = member_dir / "wrfinput_d01_copy"
            wrfinput_copy_path.unlink(missing_ok=True)
            wrfbdy_copy_path = member_dir / "wrfbdy_d01_copy"
            wrfbdy_copy_path.unlink(missing_ok=True)

            # Copy wrfinput and wrfbdy to a temporary file
            utils.copy(wrfinput_path, wrfinput_copy_path)
            utils.copy(wrfbdy_path, wrfbdy_copy_path)

            # Modify wrfinput accoarding to pertubation configuration
            logger.info(f"Member {i}: Applying pertubations to {wrfinput_path}")
            with netCDF4.Dataset(wrfinput_copy_path, "r+") as ds:  # type: ignore
                # Check if pertubations have already been applied
                if "wrf_ensembly_perts" in ds.ncattrs():
                    logger.warning(f"Pertubations already applied, skipping file")
                    continue
                ds.wrf_ensembly_perts = "True"

                for variable, pertubation in cfg.pertubations.variables.items():
                    logger.info(f"Member {i}: Perturbing {variable} by {pertubation}")
                    var = ds[variable]

                    field = pertubations.generate_pertubation_field(
                        var.shape, pertubation.mean, pertubation.sd, pertubation.rounds
                    )
                    ds[variable][:] *= field
                    ds[variable].perts = str(pertubation)

                    ## Store pertubation field in netcdf file
                    # Copy dimensions if they don't exist
                    for dim in var.dimensions:
                        if dim not in perts_nc.dimensions:
                            perts_nc.createDimension(dim, ds.dimensions[dim].size)

                    # Create variable to store pertubation field
                    if f"{variable}_pert" in perts_nc.variables:
                        field_var = perts_nc.variables[f"{variable}_pert"]
                    else:
                        field_var = perts_nc.createVariable(
                            f"{variable}_pert", var.dtype, ["member", *var.dimensions]
                        )
                        field_var.units = var.units
                        field_var.description = (
                            f"wrf-ensembly: Pertubation field for {variable}"
                        )
                        field_var.mean = pertubation.mean
                        field_var.sd = pertubation.sd
                        field_var.rounds = pertubation.rounds
                    field_var[i, :] = field

            # Update BC to match
            logger.info("Updating boundary conditions...")
            res = update_bc.update_wrf_bc(
                cfg,
                wrfinput_path,
                wrfbdy_path,
                log_filename=f"da_update_bc_member_{i}.log",
            )
            if (
                res.returncode != 0
                or "update_wrf_bc Finished successfully" not in res.output
            ):
                logger.error(
                    f"Member {i}: bc_update.exe failed with exit code {res.returncode}"
                )
                sys.exit(1)
            logger.info(f"Member {i}: bc_update.exe finished successfully")

            # Move temporary file to original file
            wrfinput_path.unlink()
            utils.copy(wrfinput_copy_path, wrfinput_path)
            wrfbdy_path.unlink()
            utils.copy(wrfbdy_copy_path, wrfbdy_path)

    logger.info("Finished applying pertubations")
    return 0


@ensemble_cli.command()
@click.argument(
    "perturbations_file", type=click.Path(exists=True, readable=True, path_type=Path)
)
@pass_experiment_path
def apply_pertubations_from_file(
    experiment_path: Path,
    perturbations_file: Path,
):
    """
    Apply pertubations from a `pertubations.nc` file, generated by a previous run of the
    `apply_pertubations` command.
    """

    logger.setup("ensemble-apply-pertubations-from-file", experiment_path)
    exp = experiment.Experiment(experiment_path)
    cfg = exp.cfg

    perts_nc_path = perturbations_file
    with netCDF4.Dataset(perts_nc_path, "r") as perts_nc:  # type: ignore
        for i in range(cfg.assimilation.n_members):
            member_dir = exp.paths.member_path(i)
            wrfinput_path = member_dir / "wrfinput_d01"
            wrfbdy_path = member_dir / "wrfbdy_d01"

            wrfinput_copy_path = member_dir / "wrfinput_d01_copy"
            wrfinput_copy_path.unlink(missing_ok=True)
            wrfbdy_copy_path = member_dir / "wrfbdy_d01_copy"
            wrfbdy_copy_path.unlink(missing_ok=True)

            # Copy wrfinput and wrfbdy to a temporary file
            utils.copy(wrfinput_path, wrfinput_copy_path)
            utils.copy(wrfbdy_path, wrfbdy_copy_path)

            # Modify wrfinput accoarding to pertubation configuration
            logger.info(f"Member {i}: Applying pertubations to {wrfinput_path}")
            with netCDF4.Dataset(wrfinput_copy_path, "r+") as ds:  # type: ignore
                ds.wrf_ensembly_perts = "True"

                for variable, pertubation in cfg.pertubations.variables.items():
                    logger.info(f"Member {i}: Perturbing {variable} by {pertubation}")

                    # Check of pert file exists in perts_nc
                    if f"{variable}_pert" not in perts_nc.variables:
                        logger.error(
                            f"Variable {variable}_perf not found in {perts_nc_path}"
                        )
                        sys.exit(1)

                    # Apply pertubation field
                    field_var = perts_nc.variables[f"{variable}_pert"]
                    field = field_var[i, :]
                    ds[variable][:] *= field
                    ds[variable].perts = str(pertubation)

            # Update BC to match
            logger.info("Updating boundary conditions...")
            res = update_bc.update_wrf_bc(
                cfg,
                wrfinput_path,
                wrfbdy_path,
                log_filename=f"da_update_bc_member_{i}.log",
            )
            if (
                res.returncode != 0
                or "update_wrf_bc Finished successfully" not in res.output
            ):
                logger.error(
                    f"Member {i}: bc_update.exe failed with exit code {res.returncode}"
                )
                sys.exit(1)
            logger.info(f"Member {i}: bc_update.exe finished successfully")

            # Move temporary file to original file
            wrfinput_path.unlink()
            utils.copy(wrfinput_copy_path, wrfinput_path)
            wrfbdy_path.unlink()
            utils.copy(wrfbdy_copy_path, wrfbdy_path)


@ensemble_cli.command()
@click.argument("member", type=int)
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

        dart_file = dart_out_dir / f"dart_analysis_member_{member:02d}.nc"
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
@pass_experiment_path
def cycle(experiment_path: Path, use_forecast: bool):
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
    cycle = exp.current_cycle
    next_cycle_i = cycle_i + 1

    if next_cycle_i >= len(exp.cycles):
        logger.error(f"Experiment is finished! No cycle {next_cycle_i}")
        sys.exit(1)

    if use_forecast:
        analysis_dir = exp.paths.scratch_forecasts_path(cycle_i)
    else:
        analysis_dir = exp.paths.scratch_analysis_path(cycle_i)

    # Prepare namelist contents, same for all members
    logger.info(
        f"Configuring members for cycle {next_cycle_i}: {str(exp.cycles[next_cycle_i])}"
    )

    # Combine initial condition file w/ analysis by copying the cycled variables, for each member
    for member in exp.members:
        member_path = exp.paths.member_path(member.i)

        # Copy the initial & boundary condition files for the next cycle, as is
        icbc_file = exp.paths.data_icbc / f"wrfinput_d01_cycle_{next_cycle_i}"
        bdy_file = exp.paths.data_icbc / f"wrfbdy_d01_cycle_{next_cycle_i}"

        icbc_target_file = member_path / "wrfinput_d01"
        bdy_target_file = member_path / "wrfbdy_d01"

        utils.copy(icbc_file, icbc_target_file)
        utils.copy(bdy_file, bdy_target_file)

        # Copy the cycled variables from the analysis file to the initial condition file
        wrfout_name = "wrfout_d01_" + cycle.end.strftime("%Y-%m-%d_%H:%M:%S")
        analysis_file = analysis_dir / f"member_{member.i:02d}" / wrfout_name

        logger.info(
            f"Copying cycled variables from {analysis_file} to {icbc_target_file}"
        )
        with (
            netCDF4.Dataset(analysis_file, "r") as nc_analysis,  # type: ignore
            netCDF4.Dataset(icbc_target_file, "r+") as nc_icbc,  # type: ignore
        ):
            for name in exp.cfg.assimilation.cycled_variables:
                if name not in nc_analysis.variables:
                    logger.warning(f"Member {member.i}: {name} not in analysis file")
                    continue
                logger.info(f"Member {member.i}: Copying {name}")
                nc_icbc[name][:] = nc_analysis[name][:]

            # Add experiment name to attributes
            nc_icbc.experiment_name = exp.cfg.metadata.name

        # Update the boundary conditions to match the new initial conditions
        logger.info(f"Member {member.i}: Updating boundary conditions")
        res = update_bc.update_wrf_bc(
            exp.cfg,
            icbc_target_file,
            bdy_target_file,
            log_filename=f"da_update_bc_analysis_member_{member.i}.log",
        )
        if (
            res.returncode != 0
            or "update_wrf_bc Finished successfully" not in res.output
        ):
            logger.error(
                f"Member {member.i}: bc_update.exe failed with exit code {res.returncode}"
            )
            logger.error(res.output)
            sys.exit(1)

        # Remove forecast files, log files
        logger.info(f"Cleaning up member directory {member_path}")
        for f in member_path.glob("wrfout*"):
            logger.debug(f"Removing forecast file {f}")
            f.unlink()
        for f in member_path.glob("rsl*"):
            logger.debug(f"Removing log file {f}")
            f.unlink()

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
