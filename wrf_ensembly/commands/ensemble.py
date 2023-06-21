from pathlib import Path
import shutil

import typer
import netCDF4
import numpy as np

app = typer.Typer()

from wrf_ensembly.console import console, get_logger, LoggerConfig
from wrf_ensembly import config, cycling, namelist, wrf, utils, pertubations


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
        / "initial_boundary_nc"
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
        shutil.copy(
            data_path / f"wrfinput_d01",
            member_dir / "wrfinput_d01",
        )
        logger.info(f"Member {i}: Copied wrfinput_d01")

        shutil.copy(
            data_path / f"wrfbdy_d01_cycle_0",
            member_dir / "wrfbdy_d01",
        )
        logger.info(f"Member {i}: Copied wrfbdy_d01_cycle_0")


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
    wrfda_dir = cfg.directories.wrfda_root
    cwd = experiment_path / cfg.directories.work_sub / "update_bc"
    cwd.mkdir(parents=True, exist_ok=True)

    (cwd / "da_update_bc.exe").unlink()
    (cwd / "da_update_bc.exe").symlink_to(wrfda_dir / "var" / "da" / "da_update_bc.exe")
    logger.info("Linked da_update_bc.exe")

    if len(cfg.pertubations) == 0:
        logger.info("No pertubations configured.")
        return 0

    for i in range(cfg.assimilation.n_members):
        member_dir = (
            experiment_path / cfg.directories.work_sub / "ensemble" / f"member_{i}"
        )

        wrfinput_path = member_dir / "wrfinput_d01"

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

        # Run bc_update.exe to update the boundary conditions file so that there are
        # no discontinuities between the initial and boundary conditions
        logger.info(f"Member {i}: Running bc_update.exe")
        bc_update_namelist = {
            "control_param": {
                "da_file": wrfinput_path.resolve(),
                "wrf_bdy_file": (member_dir / "wrfbdy_d01").resolve(),
                "domain_id": 1,
                "debug": True,
                "update_lateral_bdy": True,
                "update_low_bdy": False,
                "update_lsm": False,
                "iswater": 16,
                "var4d_lbc": False,
            }
        }
        namelist.write_namelist(bc_update_namelist, cwd / "parame.in")
        logger.info(f"Member {i}: Wrote da_update_bc namelist to {cwd / 'parame.in'}")

        cmd = [str((cwd / "da_update_bc.exe").resolve())]
        res = utils.call_external_process(cmd, cwd, logger)
        (log_dir / f"da_update_bc_member_{i}.log").write_text(res.stdout)
        if not res.success or "Update_bc completed successfully" not in res.stdout:
            logger.error(
                f"Member {i}: bc_update.exe failed with exit code {res.returncode}"
            )
            return 1
        logger.info(f"Member {i}: bc_update.exe finished successfully")

    logger.info("Finished applying pertubations")
    return 0
