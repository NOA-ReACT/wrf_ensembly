import datetime
import sys
from pathlib import Path
import os
from typing import Optional

import click
import netCDF4
import numpy as np
from rich.console import Console
from rich.table import Table

from wrf_ensembly import (
    experiment,
    external,
    member_info,
    nco,
    observations,
    pertubations,
    update_bc,
    utils,
    wrf,
)
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

    # Generate namelists for 1st cycle
    for member in exp.members:
        member_dir = exp.paths.member_path(member.i)
        wrf.generate_wrf_namelist(
            exp, cycle=0, chem_in_opt=True, paths=member_dir, member=member.i
        )

    # Create member info files
    for member in exp.members:
        member.current_cycle_i = 0
        member.current_cycle = member_info.CycleSection(
            runtime=None,
            walltime_s=None,
            advanced=False,
            filter=False,
            analysis=False,
        )
        member.write_minfo()


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
                    ds[variable][:] += field
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
                    ds[variable][:] += field
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

    member_dir = exp.paths.member_path(member)
    minfo = exp.members[member]
    if minfo.current_cycle.advanced:
        logger.info(
            f"Member {member} is already advanced to cycle {minfo.current_cycle_i}"
        )
        return 0

    # Determine number of cores
    if cores is None:
        if "SLURM_NTASKS" in os.environ:
            cores = int(os.environ["SLURM_NTASKS"])
        else:
            cores = 1
            logger.warning("No --cores no SLURM_NTASKS, will use 1 core!")
    logger.info(f"Using {cores} cores for wrf.exe")

    logger.info(f"Advancing member {member} to cycle {minfo.current_cycle_i + 1}")

    wrf_exe_path = (member_dir / "wrf.exe").resolve()
    cmd = [exp.cfg.slurm.mpirun_command, "-n", str(cores), wrf_exe_path]

    start_time = datetime.datetime.now()
    res = external.runc(cmd, member_dir, log_filename=f"wrf.log")
    end_time = datetime.datetime.now()

    for f in member_dir.glob("rsl.*"):
        logger.add_log_file(f)

    rsl_file = member_dir / "rsl.out.0000"
    if not rsl_file.exists():
        logger.error(f"Member {member}: rsl.out.0000 does not exist")
        sys.exit(1)
    rsl_content = rsl_file.read_text()

    if "SUCCESS COMPLETE WRF" not in rsl_content:
        logger.error(f"Member {member}: wrf.exe failed with exit code {res.returncode}")
        sys.exit(1)

    # Copy wrfout to the forecasts directory
    forecasts_dir = exp.paths.forecast_path(minfo.current_cycle_i, member)
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    for wrfout in member_dir.glob("wrfout*"):
        logger.info(f"Member {member}: Moving {wrfout} to {forecasts_dir}")
        wrfout.rename(forecasts_dir / wrfout.name)

    minfo.current_cycle = member_info.CycleSection(
        runtime=start_time,
        walltime_s=int((end_time - start_time).total_seconds()),
        advanced=True,
        filter=False,
        analysis=False,
    )
    minfo.write_minfo()


@ensemble_cli.command()
@pass_experiment_path
def filter(experiment_path: Path):
    """
    Runs the assimilation filter for the current cycle
    """

    logger.setup("ensemble-filter", experiment_path)
    exp = experiment.Experiment(experiment_path)
    dart_dir = exp.cfg.directories.dart_root / "models" / "wrf" / "work"
    dart_dir = dart_dir.resolve()

    # Establish which cycle we are running and that all member priors are pre-processed
    exp.ensure_same_cycle()
    exp.ensure_current_cycle_state({"advanced": True})

    current_cycle = exp.members[0].current_cycle_i
    cycle_info = exp.cycles[current_cycle]

    # Grab observations if they exist for this cycle
    obs_seq = dart_dir / "obs_seq.out"
    obs_seq.unlink(missing_ok=True)

    obs_file = exp.paths.obs / f"cycle_{current_cycle}.obs_seq"
    if not obs_file.exists():
        logger.warning(
            f"No observations found for cycle {current_cycle} ({obs_file}), skipping filter!"
        )
        sys.exit(0)
    else:
        utils.copy(obs_file, obs_seq)
        logger.info(f"Added observations!")

    # Write input/output file lists
    # For each member, we need the latest forecast only!
    wrfout_name = "wrfout_d01_" + cycle_info.end.strftime("%Y-%m-%d_%H:%M:%S")
    priors = list(
        (exp.paths.forecast_path(current_cycle)).glob(f"member_*/{wrfout_name}")
    )
    dart_output = [
        exp.paths.dart_path(current_cycle) / f"dart_analysis_{prior.parent.name}.nc"
        for prior in priors
    ]
    exp.paths.dart_path(current_cycle).mkdir(parents=True, exist_ok=True)

    dart_input_txt = dart_dir / "input_list.txt"
    dart_input_txt.write_text("\n".join([str(prior.resolve()) for prior in priors]))
    logger.info(f"Wrote input_list.txt")

    dart_output_txt = dart_dir / "output_list.txt"
    dart_output_txt.write_text("\n".join([str(f.resolve()) for f in dart_output]))
    logger.info(f"Wrote output_list.txt")

    # Run filter
    if exp.cfg.assimilation.filter_mpi_tasks == 1:
        cmd = ["./filter"]
    else:
        logger.info(
            f"Using MPI to run filter, n={exp.cfg.assimilation.filter_mpi_tasks}"
        )
        cmd = [
            exp.cfg.slurm.mpirun_command,
            "-n",
            str(exp.cfg.assimilation.filter_mpi_tasks),
            "./filter",
        ]
    res = external.runc(cmd, dart_dir, log_filename="filter.log")
    if res.returncode != 0 or "Finished ... at" not in res.output:
        logger.error(f"filter failed with exit code {res.returncode}")
        sys.exit(1)

    # Keep obs_seq.final, convert to netcdf
    obs_seq_final = dart_dir / "obs_seq.final"
    utils.copy(
        obs_seq_final,
        exp.paths.data_diag / f"cycle_{current_cycle}.obs_seq.final",
    )
    obs_seq_nc = exp.paths.data_diag / f"cycle_{current_cycle}.nc"
    observations.obs_seq_to_nc(exp, obs_seq_final, obs_seq_nc)

    # Mark filter as completed
    for m in exp.members:
        m.cycles[current_cycle].filter = True
    exp.write_all_member_info()


@ensemble_cli.command()
@pass_experiment_path
def analysis(experiment_path: Path):
    """
    Combines the DART output files and the forecast to create the analysis.
    Also creates the mean and standard deviation analysis files.
    """

    logger.setup("ensemble-analysis", experiment_path)
    exp = experiment.Experiment(experiment_path)

    # Establish which cycle we are running and that all member priors are pre-processed
    exp.ensure_same_cycle()
    exp.ensure_current_cycle_state({"advanced": True, "filter": True})

    current_cycle = exp.members[0].current_cycle_i
    cycle_info = exp.cycles[current_cycle]

    forecast_dir = exp.paths.forecast_path(current_cycle)
    analysis_dir = exp.paths.analysis_path(current_cycle)
    dart_out_dir = exp.paths.dart_path(current_cycle)

    # Postprocess analysis files
    for member in range(exp.cfg.assimilation.n_members):
        # Copy forecasts to analysis directory
        wrfout_name = "wrfout_d01_" + cycle_info.end.strftime("%Y-%m-%d_%H:%M:%S")
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
            nc_analysis.current_cycle = current_cycle
            nc_analysis.cycle_start = cycle_info.start.strftime("%Y-%m-%d_%H:%M:%S")
            nc_analysis.cycle_end = cycle_info.end.strftime("%Y-%m-%d_%H:%M:%S")

        # Update member info
        exp.members[member].current_cycle.analysis = True
        exp.members[member].write_minfo()


@ensemble_cli.command()
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
@click.option(
    "--remove-member-forecasts",
    is_flag=True,
    help="Remove the individual member forecast files after computing the statistics",
)
@click.option(
    "--remove-member-analysis",
    is_flag=True,
    help="Remove the individual member analysis files after computing the statistics",
)
@pass_experiment_path
def statistics(
    experiment_path: Path,
    cycle: Optional[int],
    jobs: int,
    remove_member_forecasts: bool,
    remove_member_analysis: bool,
):
    """
    Calculates the mean and standard deviation of the analysis files
    """

    logger.setup("statistics", experiment_path)
    exp = experiment.Experiment(experiment_path)
    exp.ensure_same_cycle()

    if cycle is None:
        cycle = exp.members[0].current_cycle_i

    logger.info(f"Cycle: {exp.cycles[cycle]}")

    # An array to collect all commands to run
    commads = []

    # Compute analysis statistics
    analysis_dir = exp.paths.analysis_path(cycle)
    analysis_files = list(analysis_dir.rglob("member_*/wrfout*"))
    if len(analysis_files) != 0:
        analysis_mean_file = analysis_dir / f"{analysis_files[0].name}_mean"
        analysis_mean_file.unlink(missing_ok=True)
        commads.append(nco.average(analysis_files, analysis_mean_file))

        analysis_sd_file = analysis_dir / f"{analysis_files[0].name}_sd"
        analysis_sd_file.unlink(missing_ok=True)
        commads.append(nco.standard_deviation(analysis_files, analysis_sd_file))
    else:
        logger.warning("No analysis files found!")

    # Compute forecast statistics
    forecast_dir = exp.paths.forecast_path(cycle)
    forecast_filenames = [x.name for x in forecast_dir.rglob("member_00/wrfout*")]
    for name in forecast_filenames:
        logger.info(f"Computing statistics for {name}")

        forecast_files = list(forecast_dir.rglob(f"member_*/{name}"))
        if len(forecast_files) == 0:
            logger.warning(f"No forecast files found for {name}!")
            continue

        forecast_mean_file = forecast_dir / f"{name}_mean"
        forecast_mean_file.unlink(missing_ok=True)
        commads.append(nco.average(forecast_files, forecast_mean_file))

        forecast_sd_file = forecast_dir / f"{name}_sd"
        forecast_sd_file.unlink(missing_ok=True)
        commads.append(nco.standard_deviation(forecast_files, forecast_sd_file))

    # Execute commands
    failure = False
    logger.info(f"Executing {len(commads)} nco commands in parallel, using {jobs} jobs")
    for res in external.run_in_parallel(commads, jobs):
        str_cmd = " ".join(res.command)
        if res.returncode != 0:
            logger.error(f"nco command failed with exit code {res.returncode}")
            logger.error(res.output)
            failure = True

    if failure:
        logger.error("One or more nco commands failed, exiting")
        sys.exit(1)

    # Remove files if required
    if remove_member_forecasts:
        logger.info(f"Removing member forecasts for cycle {cycle}")
        for f in forecast_dir.rglob("member_*/wrfout*"):
            f.unlink()
        for d in forecast_dir.glob("member_*"):
            d.rmdir()
    if remove_member_analysis:
        logger.info(f"Removing member analysis for cycle {cycle}")
        for f in analysis_dir.rglob("member_*/wrfout*"):
            f.unlink()
        for d in analysis_dir.glob("member_*"):
            d.rmdir()


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

    exp.ensure_same_cycle()

    # Establish which cycle we are running and that all member have the analysis prepared
    exp.ensure_current_cycle_state({"advanced": True})
    try:
        exp.ensure_current_cycle_state({"filter": True, "analysis": True})
    except ValueError:
        if not use_forecast:
            logger.error("Not all members have completed the analysis step")
            logger.error(
                "Either run the analysis or use `--use-forecast` to cycle w/ the latest forecast"
            )
            sys.exit(1)

        if use_forecast:
            logger.warning(
                "Not all members have completed the analysis step, using forecasts for cycling"
            )

    current_cycle = exp.members[0].current_cycle_i
    cycle_info = exp.cycles[current_cycle]
    next_cycle = current_cycle + 1

    if next_cycle >= len(exp.cycles):
        logger.error(f"Experiment is finished! No cycle {next_cycle}")
        sys.exit(1)

    if use_forecast:
        analysis_dir = exp.paths.forecast_path(current_cycle)
    else:
        analysis_dir = exp.paths.analysis_path(current_cycle)

    # Prepare namelist contents, same for all members
    cycle = exp.cycles[next_cycle]
    logger.info(f"Configuring members for cycle {next_cycle}: {str(cycle)}")

    # Update namelists
    for member in exp.members:
        member_dir = exp.paths.member_path(member.i)
        wrf.generate_wrf_namelist(
            exp, cycle=next_cycle, chem_in_opt=True, paths=member_dir, member=member.i
        )

    # Combine initial condition file w/ analysis by copying the cycled variables, for each member
    for member in exp.members:
        # Copy the initial & boundary condition files for the next cycle, as is
        icbc_file = exp.paths.data_icbc / f"wrfinput_d01_cycle_{next_cycle}"
        bdy_file = exp.paths.data_icbc / f"wrfbdy_d01_cycle_{next_cycle}"

        icbc_target_file = member.path / "wrfinput_d01"
        bdy_target_file = member.path / "wrfbdy_d01"

        utils.copy(icbc_file, icbc_target_file)
        utils.copy(bdy_file, bdy_target_file)

        # Copy the cycled variables from the analysis file to the initial condition file
        wrfout_name = "wrfout_d01_" + cycle_info.end.strftime("%Y-%m-%d_%H:%M:%S")
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
                    logger.warning(f"Member {member}: {name} not in analysis file")
                    continue
                logger.info(f"Member {member}: Copying {name}")
                nc_icbc[name][:] = nc_analysis[name][:]

            # Add experiment name to attributes
            nc_icbc.experiment_name = exp.cfg.metadata.name

        # Update the boundary conditions to match the new initial conditions
        logger.info(f"Member {member}: Updating boundary conditions")
        res = update_bc.update_wrf_bc(
            exp.cfg,
            icbc_target_file,
            bdy_target_file,
            log_filename=f"da_update_bc_analysis_member_{member}.log",
        )
        if (
            res.returncode != 0
            or "update_wrf_bc Finished successfully" not in res.output
        ):
            logger.error(
                f"Member {member}: bc_update.exe failed with exit code {res.returncode}"
            )
            logger.error(res.output)
            sys.exit(1)

        # Remove forecast files
        logger.info(f"Removing forecast files from member directory {member.path}")
        for f in member.path.glob("wrfout*"):
            logger.debug(f"Removing forecast file {f}")
            f.unlink()

        # Update member info
        member.current_cycle_i += 1
        member.current_cycle = member_info.CycleSection(
            runtime=None,
            walltime_s=None,
            advanced=False,
            filter=False,
            analysis=False,
        )
    exp.write_all_member_info()


@ensemble_cli.command()
@pass_experiment_path
def status(experiment_path: Path):
    """Prints the current status of all members (i.e., which cycles have been completed)"""

    logger.setup("experiment-status", experiment_path)
    exp = experiment.Experiment(experiment_path)

    table = Table(
        "Member",
        "Current cycle",
        "advanced",
        "filter",
        "analysis",
        title="Experiment status",
    )

    for i, member in enumerate(exp.members):
        table.add_row(
            str(i),
            str(member.current_cycle_i),
            utils.bool_to_console_str(member.current_cycle.advanced),
            utils.bool_to_console_str(member.current_cycle.filter),
            utils.bool_to_console_str(member.current_cycle.analysis),
        )

    Console().print(table)
