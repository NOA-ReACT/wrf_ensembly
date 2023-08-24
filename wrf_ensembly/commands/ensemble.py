from pathlib import Path
import datetime
from typing import Optional
from typing_extensions import Annotated

import typer
import netCDF4

app = typer.Typer()

from wrf_ensembly.console import console, get_logger, LoggerConfig
from wrf_ensembly import (
    config,
    cycling,
    namelist,
    wrf,
    utils,
    pertubations,
    member_info,
    templates,
    update_bc,
)


@app.command()
def setup(experiment_path: Path):
    """
    Generates namelists and copies initial/boundary conditions for each member.
    """

    logger, _ = get_logger(LoggerConfig(experiment_path, "ensemble-setup"))
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")
    data_path = (
        experiment_path
        / cfg.directories.work_sub
        / "preprocessing"
        / "initial_boundary"
    )  # Where the wrfinput/wrfbdy are stored

    # WRF namelist for the first cycle
    cycles = cycling.get_cycle_information(cfg)
    first_cycle = cycles[0]
    logger.info(f"Configuring members for cycle 0: {str(first_cycle)}")

    wrf_namelist = {
        "time_control": {
            **wrf.timedelta_to_namelist_items(first_cycle.end - first_cycle.start),
            **wrf.datetime_to_namelist_items(first_cycle.start, "start"),
            **wrf.datetime_to_namelist_items(first_cycle.end, "end"),
            "interval_seconds": cfg.time_control.boundary_update_interval * 60,
            "history_interval": cfg.time_control.output_interval,
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
            minfo.cycle[c.i] = member_info.CycleSection(
                runtime=None,
                walltime_s=None,
                advanced=False,
                prior_postprocessed=False,
                filter=False,
                posterior_postprocessed=False,
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

    logger, log_dir = get_logger(
        LoggerConfig(experiment_path, "ensemble-apply-pertubations")
    )
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")

    ensemble_dir = experiment_path / cfg.directories.work_sub / "ensemble"

    if len(cfg.pertubations) == 0:
        logger.info("No pertubations configured.")
        return 0

    for i in range(cfg.assimilation.n_members):
        wrfinput_path = ensemble_dir / f"member_{i}" / "wrfinput_d01"
        wrfbdy_path = ensemble_dir / f"member_{i}" / "wrfbdy_d01"

        # Modify wrfinput accoarding to pertubation configuration
        logger.info(f"Member {i}: Applying pertubations to {wrfinput_path}")
        with netCDF4.Dataset(wrfinput_path, "r+") as ds:
            for variable, pertubation in cfg.pertubations.items():
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
        res = update_bc.update_wrf_bc(cfg, logger, wrfinput_path, wrfbdy_path)
        (log_dir / f"da_update_bc_member_{i}.log").write_text(res.stdout)
        if not res.success or "update_wrf_bc Finished successfully" not in res.stdout:
            logger.error(
                f"Member {i}: bc_update.exe failed with exit code {res.returncode}"
            )
            return 1
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
        skip_wrf: If True, skips the WRF run and assumes it has already been run. Useful when something goes wrong with wrf-ensembly but you know everything is OK with the model.
    """

    logger, log_dir = get_logger(
        LoggerConfig(experiment_path, f"ensemble-advance-member_{member}")
    )
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
    cmd = ["mpirun", wrf_exe_path]  # TODO make slurm configurable here

    start_time = datetime.datetime.now()
    if skip_wrf:
        res = utils.ExternalProcessResult(0, True, "", "")
    else:
        res = utils.call_external_process(cmd, member_dir, logger)
    end_time = datetime.datetime.now()

    for log_file in member_dir.glob("rsl.*"):
        utils.copy(log_file, log_dir / log_file.name)
    (log_dir / f"wrf.log").write_text(res.stdout)

    rsl_file = member_dir / "rsl.out.0000"
    if not rsl_file.exists():
        logger.error(f"Member {member}: rsl.out.0000 does not exist")
        return 1
    rsl_content = rsl_file.read_text()

    if "SUCCESS COMPLETE WRF" not in rsl_content:
        logger.error(f"Member {member}: wrf.exe failed with exit code {res.returncode}")
        return 1

    minfo.set_current_cycle(
        member_info.CycleSection(
            runtime=start_time,
            walltime_s=(end_time - start_time).total_seconds(),
            advanced=True,
            prior_postprocessed=False,
            filter=False,
            posterior_postprocessed=False,
        )
    )
    member_info.write_member_info(experiment_path, minfo)


@app.command()
def advance_members_slurm(
    experiment_path: Path,
):
    """
    Creates a SLURM jobfile to advance each member 1 cycle
    """

    logger, _ = get_logger(
        LoggerConfig(experiment_path, f"ensemble-advance-members-slurm")
    )
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")

    jobfile_directory = experiment_path / "jobfiles"
    jobfile_directory.mkdir(parents=True, exist_ok=True)

    minfos = member_info.read_all_member_info(experiment_path)
    member_info.ensure_same_cycle(minfos)

    slurm_args = cfg.slurm
    env_modules = []
    if "env_modules" in slurm_args:
        env_modules = slurm_args["env_modules"]
        del slurm_args["env_modules"]
    for i in range(cfg.assimilation.n_members):
        jobfile = jobfile_directory / f"advance_member_{i}.job.sh"

        jobfile.write_text(
            templates.generate(
                "slurm_job.sh.j2",
                slurm_args=slurm_args
                | {"job-name": f"{cfg.metadata.name}_wrf_member_{i}"},
                env_modules=env_modules,
                commands=[
                    f"conda run -n wrf python -m wrf_ensembly ensemble advance-member {experiment_path.resolve()} {i}"
                ],
            )
        )
        logger.info(f"Wrote jobfile for member {i} to {jobfile}")


@app.command()
def postprocess_prior(experiment_path: Path, member: int, force: bool = False):
    """
    After a forward run has completed, converts the WRF output files into input ones
    by copying cycling variables into the next initial condition files.
    """

    logger, log_dir = get_logger(
        LoggerConfig(experiment_path, f"postprocess-prior-member_{member}")
    )
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")
    data_dir = experiment_path / cfg.directories.output_sub

    member_dir = cfg.get_member_dir(member)
    minfo = member_info.read_member_info(experiment_path, member)
    current_cycle = minfo.member.current_cycle
    next_cycle = current_cycle + 1

    minfo_cycle = minfo.cycle[current_cycle]
    if not force and minfo_cycle.prior_postprocessed:
        logger.info(
            f"Member {member} is already postprocessed for cycle {current_cycle}"
        )
        return 0

    cycle_info = cycling.get_cycle_information(cfg)[current_cycle]

    # Copy wrfout to the forecasts directory
    forecasts_dir = (
        data_dir / "forecasts" / f"cycle_{current_cycle}" / f"member_{member}"
    )
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    for wrfout in member_dir.glob("wrfout*"):
        # TODO move files instead of copying
        logger.info(f"Member {member}: Copying {wrfout} to {forecasts_dir}")
        utils.copy(wrfout, forecasts_dir / wrfout.name)

    # Copy initial/boundary for the next cycle to the prior directory, then copy the
    # cycled variables
    prior_dir = data_dir / "prior" / f"cycle_{current_cycle}" / f"member_{member}"
    prior_dir.mkdir(parents=True, exist_ok=True)

    initial_c = data_dir / "initial_boundary" / f"wrfinput_d01_cycle_{next_cycle}"
    boundary_c = data_dir / "initial_boundary" / f"wrfbdy_d01_cycle_{next_cycle}"
    logger.info(f"Member {member}: Copying {initial_c} to {prior_dir}")
    utils.copy(initial_c, prior_dir / "wrfinput_d01")
    logger.info(f"Member {member}: Copying {boundary_c} to {prior_dir}")
    utils.copy(boundary_c, prior_dir / "wrfbdy_d01")

    wrfout_name = "wrfout_d01_" + cycle_info.end.strftime("%Y-%m-%d_%H:%M:%S")

    logger.info("Copying cycled variables to initial conditions")
    with (
        netCDF4.Dataset(prior_dir / "wrfinput_d01", "r+") as nc_prior_initial,
        netCDF4.Dataset(forecasts_dir / wrfout_name, "r") as nc_prior_wrfout,
    ):
        for name in cfg.assimilation.cycled_variables:
            logger.info(f"Member {member}: Copying {name} from {wrfout_name}")
            nc_prior_initial[name][:] = nc_prior_wrfout[name][:]

    # Update boundary conditions to match
    res = update_bc.update_wrf_bc(cfg, logger, initial_c, boundary_c)
    (log_dir / f"da_update_bc_member_{member}.log").write_text(res.stdout)
    if not res.success or "update_wrf_bc Finished successfully" not in res.stdout:
        logger.error(
            f"Member {member}: bc_update.exe failed with exit code {res.returncode}"
        )
        return 1
    logger.info(f"Member {member}: bc_update.exe finished successfully")

    minfo.cycle[current_cycle].prior_postprocessed = True
    member_info.write_member_info(member_dir / "member_info.toml", minfo)


@app.command()
def filter(experiment_path: Path):
    """
    Runs the assimilation filter for the current cycle
    """

    logger, log_dir = get_logger(LoggerConfig(experiment_path, f"filter"))
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")
    data_dir = experiment_path / cfg.directories.output_sub

    # Establish which cycle we are running and that all member priors are pre-processed
    minfos = member_info.read_all_member_info(experiment_path)
    member_info.ensure_same_cycle(minfos)
    member_info.ensure_current_cycle_state(
        minfos, {"advanced": True, "prior_postprocessed": True}
    )

    current_cycle = minfos[0].member.current_cycle

    # Write input/output file lists
    priors = list(
        (data_dir / "prior" / f"cycle_{current_cycle}").glob("member_*/wrfinput_d01")
    )
    analysis = [
        data_dir
        / "analysis"
        / f"cycle_{current_cycle}"
        / f"analysis_{prior.parent.name}.nc"
        for prior in priors
    ]
    (data_dir / "analysis" / f"cycle_{current_cycle}").mkdir(
        parents=True, exist_ok=True
    )

    dart_dir = cfg.directories.dart_root / "models" / "wrf" / "work"
    dart_dir = dart_dir.resolve()
    dart_input_txt = dart_dir / "input_list.txt"
    dart_input_txt.write_text("\n".join([str(prior) for prior in priors]))
    logger.info(f"Wrote input_list.txt")

    dart_output_txt = dart_dir / "output_list.txt"
    dart_output_txt.write_text("\n".join([str(f) for f in analysis]))
    logger.info(f"Wrote output_list.txt")

    # Run filter
    cmd = ["./filter"]
    res = utils.call_external_process(cmd, dart_dir, logger)
    (log_dir / f"filter.log").write_text(res.stdout)
    if not res.success or "Finished ... at" not in res.stdout:
        logger.error(f"filter failed with exit code {res.returncode}")
        return 1

    # Copy output to posterior directory
    posterior_dir = data_dir / "posterior" / f"cycle_{current_cycle}"
    posterior_dir.mkdir(parents=True, exist_ok=True)
    for f in dart_dir.glob("dart_member_*.nc"):
        utils.copy(f, posterior_dir / f.name)
        logger.info(f"Copied {f} to {posterior_dir}")


@app.command()
def postprocess_analysis(experiment_path: Path):
    """
    Postprocesses the analysis files from the filter, by running bc_update again
    and moving them to the appropriate directory to re-run the model.
    """

    logger, log_dir = get_logger(LoggerConfig(experiment_path, f"postprocess_analysis"))
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")
    data_dir = experiment_path / cfg.directories.output_sub
    icbc_dir = data_dir / "initial_boundary"

    # Establish which cycle we are running and that all member priors are pre-processed
    minfos = member_info.read_all_member_info(experiment_path)
    member_info.ensure_current_cycle_state(
        minfos, {"advanced": True, "prior_postprocessed": True, "filter": True}
    )

    current_cycle = minfos[0].member.current_cycle
    next_cycle = current_cycle + 1
    analysis_dir = data_dir / "analysis" / f"cycle_{current_cycle}"

    # Prepare namelist contents
    cycles = cycling.get_cycle_information(cfg)
    cycle = cycles[next_cycle]
    logger.info(f"Configuring members for cycle {next_cycle}: {str(cycle)}")

    wrf_namelist = {
        "time_control": {
            **wrf.timedelta_to_namelist_items(cycle.end - cycle.start),
            **wrf.datetime_to_namelist_items(cycle.start, "start"),
            **wrf.datetime_to_namelist_items(cycle.end, "end"),
            "interval_seconds": cfg.time_control.boundary_update_interval * 60,
            "history_interval": cfg.time_control.output_interval,
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

    # Postprocess analysis files
    for member in range(cfg.assimilation.n_members):
        member_dir = cfg.get_member_dir(member)
        analysis_file = analysis_dir / f"analysis_member_{member}.nc"
        if not analysis_file.exists():
            logger.error(f"Member {member}: {analysis_file} does not exist")
            return 1

        target_wrfinput = member_dir / "wrfinput_d01"
        target_wrfbdy = member_dir / "wrfbdy_d01"

        utils.copy(icbc_dir / f"wrfinput_d01_cycle_{next_cycle}", target_wrfinput)
        utils.copy(icbc_dir / f"wrfbdy_d01_cycle_{next_cycle}", target_wrfbdy)

        logger.info(f"Member {member}: Copying cycled variables to initial conditions")
        with (
            netCDF4.Dataset(analysis_file, "r") as nc_analysis,
            netCDF4.Dataset(target_wrfinput, "r+") as nc_wrfinput,
        ):
            for name in cfg.assimilation.cycled_variables:
                if name not in nc_analysis.variables:
                    logger.warning(f"Member {member}: {name} not in analysis file")
                    continue
                logger.info(f"Member {member}: Copying {name}")
                nc_wrfinput[name][:] = nc_analysis[name][:]

        logger.info(f"Member {member}: Postprocessing analysis file")
        res = update_bc.update_wrf_bc(cfg, logger, target_wrfinput, target_wrfbdy)
        (log_dir / f"da_update_bc_analysis_member_{member}.log").write_text(res.stdout)
        if not res.success or "update_wrf_bc Finished successfully" not in res.stdout:
            logger.error(
                f"Member {member}: bc_update.exe failed with exit code {res.returncode}"
            )
            logger.error(res.stdout)
            continue

        # Write namelist
        namelist_path = member_dir / "namelist.input"
        namelist.write_namelist(wrf_namelist, namelist_path)
        logger.info(f"Member {member}: Wrote namelist to {namelist_path}")

        # Update member info
        minfos[member].cycle[current_cycle].posterior_postprocessed = True
        minfos[member].member.current_cycle = next_cycle
        member_info.write_member_info(
            experiment_path
            / cfg.directories.work_sub
            / "ensemble"
            / f"member_{member}"
            / "member_info.toml",
            minfos[member],
        )

        # Remove forecast files
        logger.info(f"Removing forecast files from member directory {member_dir}")
        for f in member_dir.glob("wrfout*"):
            logger.debug(f"Removing forecast file {f}")
            f.unlink()
