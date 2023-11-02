from pathlib import Path
import datetime
from typing import Optional
from typing_extensions import Annotated

import typer
import netCDF4
import numpy as np

app = typer.Typer()

from wrf_ensembly.console import logger
from wrf_ensembly import (
    config,
    cycling,
    namelist,
    wrf,
    utils,
    pertubations,
    member_info,
    update_bc,
    nco,
)


@app.command()
def setup(experiment_path: Path):
    """
    Generates namelists and copies initial/boundary conditions for each member.
    """

    logger.setup("ensemble-setup", experiment_path)
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")
    data_path = (
        experiment_path / cfg.directories.output_sub / "initial_boundary"
    )  # Where the wrfinput/wrfbdy are stored

    # WRF namelist for the first cycle
    cycles = cycling.get_cycle_information(cfg)
    first_cycle = cycles[0]
    logger.info(f"Configuring members for cycle 0: {str(first_cycle)}")

    history_interval = cfg.time_control.output_interval
    if first_cycle.output_interval is not None:
        history_interval = cycle.output_interval
    wrf_namelist = {
        "time_control": {
            **wrf.timedelta_to_namelist_items(first_cycle.end - first_cycle.start),
            **wrf.datetime_to_namelist_items(first_cycle.start, "start"),
            **wrf.datetime_to_namelist_items(first_cycle.end, "end"),
            "interval_seconds": cfg.time_control.boundary_update_interval * 60,
            "history_interval": history_interval,
        },
        "domains": {
            "e_we": cfg.domain_control.xy_size[0],
            "e_sn": cfg.domain_control.xy_size[1],
            "dx": cfg.domain_control.xy_resolution[0] * 1000,
            "dy": cfg.domain_control.xy_resolution[1] * 1000,
            "grid_id": 1,
            "parent_id": 0,
            "max_dom": 1,
        },
    }
    for name, group in cfg.wrf_namelist.items():
        if name in wrf_namelist:
            wrf_namelist[name] |= group
        else:
            wrf_namelist[name] = group

    for i in range(cfg.assimilation.n_members):
        member_dir = cfg.get_member_dir(i)
        member_dir.mkdir(parents=True, exist_ok=True)

        namelist_path = member_dir / "namelist.input"
        namelist.write_namelist(wrf_namelist, namelist_path)
        logger.info(f"Member {i}: Wrote namelist to {namelist_path}")

        # Copy initial and boundary conditions
        utils.copy(
            data_path / f"wrfinput_d01_cycle_0",
            member_dir / "wrfinput_d01",
        )
        logger.info(f"Member {i}: Copied wrfinput_d01")

        utils.copy(
            data_path / f"wrfbdy_d01_cycle_0",
            member_dir / "wrfbdy_d01",
        )
        logger.info(f"Member {i}: Copied wrfbdy_d01_cycle_0")

        # Create member info file
        minfo = member_info.MemberInfo(
            metadata=cfg.metadata, member={"i": i, "current_cycle": 0}
        )
        for c in cycles:
            minfo.cycle[c.index] = member_info.CycleSection(
                runtime=None,
                walltime_s=None,
                advanced=False,
                filter=False,
                analysis=False,
            )
        toml_path = member_info.write_member_info(experiment_path, minfo)
        logger.info(f"Member {i}: Wrote info file to {toml_path}")


@app.command()
def apply_pertubations(
    experiment_path: Path,
):
    """
    Applies the configured pertubations to the initial conditions of each ensemble member
    """

    logger.setup("ensemble-apply-pertubations", experiment_path)
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")

    if len(cfg.pertubations.variables) == 0:
        logger.info("No pertubations configured.")
        return 0

    if cfg.pertubations.seed is not None:
        logger.warning(f"Setting numpy random seed to {cfg.pertubations.seed}")
        np.random.seed(cfg.pertubations.seed)

    for i in range(cfg.assimilation.n_members):
        member_dir = cfg.get_member_dir(i)
        wrfinput_path = member_dir / "wrfinput_d01"
        wrfbdy_path = member_dir / "wrfbdy_d01"

        # Modify wrfinput accoarding to pertubation configuration
        logger.info(f"Member {i}: Applying pertubations to {wrfinput_path}")
        with netCDF4.Dataset(wrfinput_path, "r+") as ds:
            for variable, pertubation in cfg.pertubations.variables.items():
                logger.info(f"Member {i}: Perturbing {variable} by {pertubation}")
                var = ds[variable]
                field = pertubations.generate_pertubation_field(
                    var.shape, pertubation.mean, pertubation.sd, pertubation.rounds
                )
                ds[variable][:] += field

                # Store pertubation field in netcdf file
                if f"{variable}_pert" in ds.variables:
                    field_var = ds[f"{variable}_pert"]
                else:
                    field_var = ds.createVariable(
                        f"{variable}_pert", var.dtype, var.dimensions
                    )
                field_var[:] = field
                field_var.units = var.units
                field_var.description = (
                    f"wrf-ensembly: Pertubation field for {variable}"
                )
                field_var.mean = pertubation.mean
                field_var.sd = pertubation.sd
                field_var.rounds = pertubation.rounds

        # Update BC to match
        res = update_bc.update_wrf_bc(
            cfg,
            wrfinput_path,
            wrfbdy_path,
            log_filename=f"da_update_bc_member_{i}.log",
        )
        if not res.success or "update_wrf_bc Finished successfully" not in res.stdout:
            logger.error(
                f"Member {i}: bc_update.exe failed with exit code {res.returncode}"
            )
            raise typer.Exit(1)
        logger.info(f"Member {i}: bc_update.exe finished successfully")

    logger.info("Finished applying pertubations")
    return 0


@app.command()
def advance_member(
    experiment_path: Path,
    member: int,
    skip_wrf: Annotated[Optional[bool], typer.Option()] = False,
):
    """
    Advances the given member 1 cycle by running the model

    Args:
        experiment_path: Path to the experiment directory
        member: The member to advance
        skip_wrf: If True, skips the WRF run and assumes it has already been run.
                  Useful when something goes wrong with wrf-ensembly but you know
                  everything is OK with the model.
    """

    logger.setup(f"ensemble-advance-member_{member}", experiment_path)
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")

    member_dir = cfg.get_member_dir(member)
    minfo = member_info.read_member_info(experiment_path, member)
    if minfo.get_current_cycle().advanced:
        logger.info(
            f"Member {member} is already advanced to cycle {minfo.member.current_cycle}"
        )
        return 0

    logger.info(f"Advancing member {member} to cycle {minfo.member.current_cycle + 1}")

    wrf_exe_path = (member_dir / "wrf.exe").resolve()
    cmd = [cfg.slurm.mpirun_command, wrf_exe_path]

    start_time = datetime.datetime.now()
    if skip_wrf:
        res = utils.ExternalProcessResult(0, True, "", "")
    else:
        res = utils.call_external_process(cmd, member_dir, log_filename=f"wrf.log")
    end_time = datetime.datetime.now()

    for f in member_dir.glob("rsl.*"):
        logger.add_log_file(f)

    rsl_file = member_dir / "rsl.out.0000"
    if not rsl_file.exists():
        logger.error(f"Member {member}: rsl.out.0000 does not exist")
        return typer.Exit(1)
    rsl_content = rsl_file.read_text()

    if "SUCCESS COMPLETE WRF" not in rsl_content:
        logger.error(f"Member {member}: wrf.exe failed with exit code {res.returncode}")
        return typer.Exit(1)

    # Copy wrfout to the forecasts directory
    data_dir = experiment_path / cfg.directories.output_sub
    current_cycle = minfo.member.current_cycle

    forecasts_dir = (
        data_dir / "forecasts" / f"cycle_{current_cycle}" / f"member_{member:02d}"
    )
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    for wrfout in member_dir.glob("wrfout*"):
        # TODO move files instead of copying, this is just to debug this command in dev
        logger.info(f"Member {member}: Copying {wrfout} to {forecasts_dir}")
        utils.copy(wrfout, forecasts_dir / wrfout.name)

    minfo.set_current_cycle(
        member_info.CycleSection(
            runtime=start_time,
            walltime_s=(end_time - start_time).total_seconds(),
            advanced=True,
            filter=False,
            analysis=False,
        )
    )
    member_info.write_member_info(experiment_path, minfo)


@app.command()
def filter(experiment_path: Path):
    """
    Runs the assimilation filter for the current cycle
    """

    logger.setup("ensemble-filter", experiment_path)
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")
    data_dir = experiment_path / cfg.directories.output_sub
    dart_dir = cfg.directories.dart_root / "models" / "wrf" / "work"
    dart_dir = dart_dir.resolve()

    # Establish which cycle we are running and that all member priors are pre-processed
    minfos = member_info.read_all_member_info(experiment_path)
    member_info.ensure_same_cycle(minfos)
    member_info.ensure_current_cycle_state(minfos, {"advanced": True})

    current_cycle = minfos[0].member.current_cycle
    cycle_info = cycling.get_cycle_information(cfg)[current_cycle]

    # Grab observations if they exist for this cycle
    obs_seq = dart_dir / "obs_seq.out"
    obs_seq.unlink(missing_ok=True)

    obs_file = experiment_path / "obs" / f"cycle_{current_cycle}.obs_seq"
    if not obs_file.exists():
        logger.warning(
            f"No observations found for cycle {current_cycle} ({obs_file}), skipping filter!"
        )
        raise typer.Exit(0)
    else:
        utils.copy(obs_file, obs_seq)
        logger.info(f"Added observations!")

    # Write input/output file lists
    # For each member, we need the latest forecast only!
    wrfout_name = "wrfout_d01_" + cycle_info.end.strftime("%Y-%m-%d_%H:%M:%S")
    priors = list(
        (data_dir / "forecasts" / f"cycle_{current_cycle}").glob(
            f"member_*/{wrfout_name}"
        )
    )
    dart_output = [
        data_dir
        / "dart"
        / f"cycle_{current_cycle}"
        / f"dart_analysis_{prior.parent.name}.nc"
        for prior in priors
    ]
    (data_dir / "dart" / f"cycle_{current_cycle}").mkdir(parents=True, exist_ok=True)

    dart_input_txt = dart_dir / "input_list.txt"
    dart_input_txt.write_text("\n".join([str(prior) for prior in priors]))
    logger.info(f"Wrote input_list.txt")

    dart_output_txt = dart_dir / "output_list.txt"
    dart_output_txt.write_text("\n".join([str(f) for f in dart_output]))
    logger.info(f"Wrote output_list.txt")

    # Run filter
    if cfg.assimilation.filter_mpi_tasks == 1:
        cmd = ["./filter"]
    else:
        logger.info(f"Using MPI to run filter, n={cfg.assimilation.filter_mpi_tasks}")
        cmd = [
            cfg.slurm.mpirun_command,
            "-n",
            str(cfg.assimilation.filter_mpi_tasks),
            "./filter",
        ]
    res = utils.call_external_process(cmd, dart_dir, log_filename="filter.log")
    if not res.success or "Finished ... at" not in res.stdout:
        logger.error(f"filter failed with exit code {res.returncode}")
        raise typer.Exit(1)

    # Mark filter as completed
    for minfo in minfos.values():
        minfo.cycle[current_cycle].filter = True
        member_info.write_member_info(experiment_path, minfo)


@app.command()
def analysis(experiment_path: Path):
    """
    Combines the DART output files and the forecast to create the analysis.
    Also creates the mean and standard deviation analysis files.
    """

    logger.setup("ensemble-analysis", experiment_path)
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")
    data_dir = experiment_path / cfg.directories.output_sub

    # Establish which cycle we are running and that all member priors are pre-processed
    minfos = member_info.read_all_member_info(experiment_path)
    member_info.ensure_current_cycle_state(minfos, {"advanced": True, "filter": True})

    current_cycle = minfos[0].member.current_cycle
    cycle_info = cycling.get_cycle_information(cfg)[current_cycle]

    forecast_dir = data_dir / "forecasts" / f"cycle_{current_cycle}"
    analysis_dir = data_dir / "analysis" / f"cycle_{current_cycle}"
    dart_out_dir = data_dir / "dart" / f"cycle_{current_cycle}"

    # Postprocess analysis files
    for member in range(cfg.assimilation.n_members):
        # Copy forecasts to analysis directory
        wrfout_name = "wrfout_d01_" + cycle_info.end.strftime("%Y-%m-%d_%H:%M:%S")
        forecast_file = forecast_dir / f"member_{member:02d}" / wrfout_name
        analysis_file = analysis_dir / f"member_{member:02d}" / wrfout_name
        utils.copy(forecast_file, analysis_file)

        dart_file = dart_out_dir / f"dart_analysis_member_{member:02d}.nc"
        if not dart_file.exists():
            logger.error(f"Member {member}: {dart_file} does not exist")
            return typer.Exit(1)

        # Copy the state variables from the dart file to the analysis file
        logger.info(f"Member {member}: Copying state variables from {dart_file}")
        with (
            netCDF4.Dataset(dart_file, "r") as nc_dart,
            netCDF4.Dataset(analysis_file, "r+") as nc_analysis,
        ):
            for name in cfg.assimilation.state_variables:
                if name not in nc_dart.variables:
                    logger.warning(f"Member {member}: {name} not in dart file")
                    continue
                logger.info(f"Member {member}: Copying {name}")
                nc_analysis[name][:] = nc_dart[name][:]

            # Add experiment name and current cycle information to attributes
            # TODO Standardize this somehow? We must add metadata to all files!
            nc_analysis.experiment_name = cfg.metadata.name
            nc_analysis.current_cycle = current_cycle
            nc_analysis.cycle_start = cycle_info.start.strftime("%Y-%m-%d_%H:%M:%S")
            nc_analysis.cycle_end = cycle_info.end.strftime("%Y-%m-%d_%H:%M:%S")

        # Update member info
        minfos[member].cycle[current_cycle].analysis = True
        member_info.write_member_info(experiment_path, minfos[member])


@app.command()
def statistics(
    experiment_path: Path,
    cycle: Annotated[
        Optional[int],
        typer.Argument(
            ..., help="Cycle to compute statistics for. Current cycle if not set"
        ),
    ] = None,
    remove_member_forecasts: Annotated[
        Optional[bool],
        typer.Option(
            ...,
            help="Remove the individual member forecast files after computing the statistics",
        ),
    ] = False,
    remove_member_analysis: Annotated[
        Optional[bool],
        typer.Option(
            ...,
            help="Remove the individual member analysis files after computing the statistics",
        ),
    ] = False,
):
    """
    Calculates the mean and standard deviation of the analysis files
    """

    logger.setup("statistics", experiment_path)
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")

    data_dir = experiment_path / cfg.directories.output_sub

    if cycle is None:
        minfos = member_info.read_all_member_info(experiment_path)
        cycle = minfos[0].member.current_cycle
        member_info.ensure_current_cycle_state(minfos, {"analysis": True})

    cycle = cycling.get_cycle_information(cfg)[cycle]

    # Compute analysis statistics
    logger.info(f"Computing analysis statistics for {cycle}")
    analysis_dir = data_dir / "analysis" / f"cycle_{cycle.index}"
    analysis_files = list(analysis_dir.rglob("member_*/wrfout*"))
    if len(analysis_files) != 0:
        analysis_mean_file = (
            data_dir
            / "analysis"
            / f"cycle_{cycle.index}"
            / f"{analysis_files[0].name}_mean"
        )
        analysis_mean_file.unlink(missing_ok=True)
        nco.average(analysis_files, analysis_mean_file)

        analysis_sd_file = (
            data_dir
            / "analysis"
            / f"cycle_{cycle.index}"
            / f"{analysis_files[0].name}_sd"
        )
        analysis_sd_file.unlink(missing_ok=True)
        nco.standard_deviation(analysis_files, analysis_sd_file)
    else:
        logger.warning(f"No analysis files found for {cycle}!")

    # Compute forecast statistics
    logger.info(f"Computing forecast statistics for {cycle}")
    forecast_dir = data_dir / "forecasts" / f"cycle_{cycle.index}"
    forecast_filenames = [x.name for x in forecast_dir.rglob("member_00/wrfout*")]
    for name in forecast_filenames:
        logger.info(f"Computing statistics for {name}")

        forecast_files = list(forecast_dir.rglob(f"member_*/{name}"))
        if len(forecast_files) == 0:
            logger.warning(f"No forecast files found for {name}!")
            continue

        forecast_mean_file = (
            data_dir / "forecasts" / f"cycle_{cycle.index}" / f"{name}_mean"
        )
        forecast_mean_file.unlink(missing_ok=True)
        nco.average(forecast_files, forecast_mean_file)

        forecast_sd_file = (
            data_dir / "forecasts" / f"cycle_{cycle.index}" / f"{name}_sd"
        )
        forecast_sd_file.unlink(missing_ok=True)
        nco.standard_deviation(forecast_files, forecast_sd_file)

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


@app.command()
def cycle(experiment_path: Path, use_forecast: bool = False):
    """
    Prepares the experiment for the next cycle by copying the cycled variables from the analysis
    to the initial conditions and preparing the namelist.
    """

    logger.setup("cycle", experiment_path)
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")
    data_dir = experiment_path / cfg.directories.output_sub
    icbc_dir = data_dir / "initial_boundary"

    # Establish which cycle we are running and that all member have the analysis prepared
    minfos = member_info.read_all_member_info(experiment_path)
    member_info.ensure_current_cycle_state(minfos, {"advanced": True})
    try:
        member_info.ensure_current_cycle_state(minfos, {"analysis": True})
    except ValueError:
        if not use_forecast:
            logger.error("Not all members have completed the analysis step")
            logger.error(
                "Either run the analysis or use `--use-forecast` to cycle w/ the latest forecast"
            )
            raise typer.Exit(1)

        if use_forecast:
            logger.warning(
                "Not all members have completed the analysis step, using forecasts for cycling"
            )

    current_cycle = minfos[0].member.current_cycle
    cycle_info = cycling.get_cycle_information(cfg)[current_cycle]
    next_cycle = current_cycle + 1

    if use_forecast:
        analysis_dir = data_dir / "forecasts" / f"cycle_{current_cycle}"
    else:
        analysis_dir = data_dir / "analysis" / f"cycle_{current_cycle}"

    # Prepare namelist contents, same for all members
    cycles = cycling.get_cycle_information(cfg)
    cycle = cycles[next_cycle]
    logger.info(f"Configuring members for cycle {next_cycle}: {str(cycle)}")

    history_interval = cfg.time_control.output_interval
    if cycle.output_interval is not None:
        history_interval = cycle.output_interval
    wrf_namelist = {
        "time_control": {
            **wrf.timedelta_to_namelist_items(cycle.end - cycle.start),
            **wrf.datetime_to_namelist_items(cycle.start, "start"),
            **wrf.datetime_to_namelist_items(cycle.end, "end"),
            "interval_seconds": cfg.time_control.boundary_update_interval * 60,
            "history_interval": history_interval,
        },
        "domains": {
            "e_we": cfg.domain_control.xy_size[0],
            "e_sn": cfg.domain_control.xy_size[1],
            "dx": cfg.domain_control.xy_resolution[0] * 1000,
            "dy": cfg.domain_control.xy_resolution[1] * 1000,
            "grid_id": 1,
            "parent_id": 0,
            "max_dom": 1,
        },
    }
    for name, group in cfg.wrf_namelist.items():
        if name in wrf_namelist:
            wrf_namelist[name] |= group
        else:
            wrf_namelist[name] = group

    # Combine initial condition file w/ analysis by copying the cycled variables, for each member
    for member in range(cfg.assimilation.n_members):
        member_dir = cfg.get_member_dir(member)

        # Copy the initial & boundary condition files for the next cycle, as is
        icbc_file = icbc_dir / f"wrfinput_d01_cycle_{next_cycle}"
        bdy_file = icbc_dir / f"wrfbdy_d01_cycle_{next_cycle}"

        icbc_target_file = member_dir / "wrfinput_d01"
        bdy_target_file = member_dir / "wrfbdy_d01"

        utils.copy(icbc_file, icbc_target_file)
        utils.copy(bdy_file, bdy_target_file)

        # Copy the cycled variables from the analysis file to the initial condition file
        wrfout_name = "wrfout_d01_" + cycle_info.end.strftime("%Y-%m-%d_%H:%M:%S")
        analysis_file = analysis_dir / f"member_{member:02d}" / wrfout_name

        logger.info(
            f"Copying cycled variables from {analysis_file} to {icbc_target_file}"
        )
        with (
            netCDF4.Dataset(analysis_file, "r") as nc_analysis,
            netCDF4.Dataset(icbc_target_file, "r+") as nc_icbc,
        ):
            for name in cfg.assimilation.cycled_variables:
                if name not in nc_analysis.variables:
                    logger.warning(f"Member {member}: {name} not in analysis file")
                    continue
                logger.info(f"Member {member}: Copying {name}")
                nc_icbc[name][:] = nc_analysis[name][:]

            # Add experiment name to attributes
            nc_icbc.experiment_name = cfg.metadata.name

        # Update the boundary conditions to match the new initial conditions
        logger.info(f"Member {member}: Updating boundary conditions")
        res = update_bc.update_wrf_bc(
            cfg,
            icbc_target_file,
            bdy_target_file,
            log_filename=f"da_update_bc_analysis_member_{member}.log",
        )
        if not res.success or "update_wrf_bc Finished successfully" not in res.stdout:
            logger.error(
                f"Member {member}: bc_update.exe failed with exit code {res.returncode}"
            )
            logger.error(res.stdout)
            # TODO raise exception?
            continue

        # Write namelist
        namelist_path = member_dir / "namelist.input"
        namelist.write_namelist(wrf_namelist, namelist_path)
        logger.info(f"Member {member}: Wrote namelist to {namelist_path}")

        # Remove forecast files
        logger.info(f"Removing forecast files from member directory {member_dir}")
        for f in member_dir.glob("wrfout*"):
            logger.debug(f"Removing forecast file {f}")
            f.unlink()

        # Update member info
        minfos[member].member.current_cycle += 1
        minfos[member].cycle[next_cycle] = member_info.CycleSection(
            runtime=None,
            walltime_s=None,
            advanced=False,
            filter=False,
            analysis=False,
        )
        member_info.write_member_info(experiment_path, minfos[member])
