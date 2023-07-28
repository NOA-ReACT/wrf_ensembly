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
        member_dir = (
            experiment_path / cfg.directories.work_sub / "ensemble" / f"member_{i}"
        )
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
        member_info.write_member_info(member_dir / "member_info.toml", minfo)
        logger.info(f"Member {i}: Wrote info to {member_dir / 'member_info.toml'}")


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

    member_dir = (
        experiment_path / cfg.directories.work_sub / "ensemble" / f"member_{member}"
    )
    minfo = member_info.read_member_info(member_dir / "member_info.toml")
    if (
        minfo.member.current_cycle in minfo.cycle
        and minfo.cycle[minfo.member.current_cycle].advanced
    ):
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

    minfo.cycle[minfo.member.current_cycle] = member_info.CycleSection(
        runtime=start_time,
        walltime_s=(end_time - start_time).total_seconds(),
        advanced=True,
        prior_postprocessed=False,
        filter=False,
        posterior_postprocessed=False,
    )
    member_info.write_member_info(member_dir / "member_info.toml", minfo)


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

    minfos = [
        member_info.read_member_info(
            experiment_path
            / cfg.directories.work_sub
            / "ensemble"
            / f"member_{i}"
            / "member_info.toml"
        )
        for i in range(cfg.assimilation.n_members)
    ]

    current_cycle = [minfo.member.current_cycle for minfo in minfos]
    for c in current_cycle[1:]:
        if c != current_cycle[0]:
            logger.error(
                f"Member {current_cycle.index(c)} has a different current cycle than member 0"
            )
            return 1

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

    member_dir = (
        experiment_path / cfg.directories.work_sub / "ensemble" / f"member_{member}"
    )
    minfo = member_info.read_member_info(member_dir / "member_info.toml")
    current_cycle = minfo.member.current_cycle
    next_cycle = current_cycle + 1
    if current_cycle not in minfo.cycle:
        logger.error(f"Member {member} has not been advanced to cycle {current_cycle}")
        logger.error("Did `advance` run successfully?")
        return 1
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
    minfos = [
        member_info.read_member_info(
            experiment_path
            / cfg.directories.work_sub
            / "ensemble"
            / f"member_{i}"
            / "member_info.toml"
        )
        for i in range(cfg.assimilation.n_members)
    ]
    current_cycle = minfos[0].member.current_cycle
    for minfo in minfos:
        if minfo.member.current_cycle != current_cycle:
            logger.error(
                f"Member {minfo.member.i} has a different current cycle than member 0"
            )
            return 1
        if current_cycle not in minfo.cycle:
            logger.error(
                f"Member {minfo.member.i} has not been advanced to cycle {current_cycle}"
            )
            return 1
        if minfo.cycle[current_cycle].prior_postprocessed is False:
            logger.error(
                f"Member {minfo.member.i} has not been postprocessed for cycle {current_cycle}"
            )
            return 1

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


# @app.command()
# def compute_cycle_statistics(experiment_path: Path):
#     logger, _ = get_logger(
#         LoggerConfig(experiment_path, f"ensemble-advance-members-slurm")
#     )
#     experiment_path = experiment_path.resolve()
#     cfg = config.read_config(experiment_path / "config.toml")
#     ensemble_dir = experiment_path / cfg.directories.work_sub / "ensemble"

#     # Verify that all members are at the same point in time (cycle)
#     minfos = [
#         member_info.read_member_info(ensemble_dir / f"member_{i}" / "member_info.toml")
#         for i in range(cfg.assimilation.n_members)
#     ]
#     current_cycle = [minfo.member.current_cycle for minfo in minfos]
#     for c in current_cycle[1:]:
#         if c != current_cycle[0]:
#             logger.error(
#                 f"Member {current_cycle.index(c)} has a different current cycle than member 0"
#             )
#             return 1

#     # Use member 0 as a reference to check how many files we should have
#     files = set([f.name for f in (ensemble_dir / "member_0").glob("wrfout_d01_*")])
#     for i in range(1, cfg.assimilation.n_members):
#         for f in (ensemble_dir / f"member_{i}").glob("wrfout_d01_*"):
#             if f.name not in files:
#                 logger.error(
#                     f"Member 0 is missing file {f.name} that exists in member {i}"
#                 )
#                 return 1

#     # Move files to analysis directory
#     analysis_dir = experiment_path / "analysis" / f"cycle_{current_cycle[0] - 1}"
#     analysis_dir.mkdir(parents=True, exist_ok=True)

#     prior_dir = analysis_dir / "prior"
#     prior_dir.mkdir(parents=True, exist_ok=True)
#     # for i in range(cfg.assimilation.n_members):
#     #     member_dir = prior_dir / f"member_{i}"
#     #     member_dir.mkdir(exist_ok=True)
#     #     for f in files:
#     #         source = ensemble_dir / f"member_{i}" / f
#     #         logger.info(f"Moving {source} to {member_dir}")
#     #         shutil.copy(source, member_dir)  # TODO Change to move?

#     # Compute statistics
#     times = [f.replace("wrfout_d01_", "") for f in files]
#     times = sorted(times)
#     times = [times[-1]]
#     for t in times:
#         logger.info(f"Computing ensemble mean/SD for {t}...")
#         with netCDF4.Dataset(prior_dir / "wrfout_ensemble", "w") as nc_ensemble:
#             member_ncs = [
#                 netCDF4.Dataset(prior_dir / f"member_{i}" / f"wrfout_d01_{t}", "r")
#                 for i in range(cfg.assimilation.n_members)
#             ]
#             for name, dim in member_ncs[0].dimensions.items():
#                 nc_ensemble.createDimension(name, dim.size)

#             for v in member_ncs[0].variables:
#                 if v == "Times":
#                     nc_ensemble.createVariable(
#                         "Times",
#                         "S1",
#                         member_ncs[0].variables["Times"].dimensions,
#                     )
#                     nc_ensemble.variables["Times"][:] = member_ncs[0]["Times"][:]
#                     for attr in member_ncs[0].variables["Times"].ncattrs():
#                         nc_ensemble.variables["Times"].setncattr(
#                             attr,
#                             getattr(member_ncs[0].variables["Times"], attr),
#                         )
#                     continue

#                 if v not in nc_ensemble.variables:
#                     nc_ensemble.createVariable(
#                         f"{v}_mean",
#                         member_ncs[0].variables[v].dtype,
#                         member_ncs[0].variables[v].dimensions,
#                     )
#                     nc_ensemble.createVariable(
#                         f"{v}_sd",
#                         member_ncs[0].variables[v].dtype,
#                         member_ncs[0].variables[v].dimensions,
#                     )
#                     # Copy attributes
#                     for attr in member_ncs[0].variables[v].ncattrs():
#                         nc_ensemble.variables[f"{v}_mean"].setncattr(
#                             attr,
#                             getattr(member_ncs[0].variables[v], attr),
#                         )
#                         nc_ensemble.variables[f"{v}_sd"].setncattr(
#                             attr,
#                             getattr(member_ncs[0].variables[v], attr),
#                         )
#                 stack = np.stack([nc.variables[v][:] for nc in member_ncs])
#                 nc_ensemble.variables[f"{v}_mean"][:] = stack.mean(axis=0)
#                 nc_ensemble.variables[f"{v}_sd"][:] = stack.std(axis=0)

#             for nc in member_ncs:
#                 nc.close()
