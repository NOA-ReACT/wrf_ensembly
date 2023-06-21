from pathlib import Path
import shutil
import datetime
from itertools import chain

import typer
import netCDF4
import numpy as np

from wrf_ensembly.console import console, get_logger, LoggerConfig
from wrf_ensembly import config, cycling, namelist, wrf, utils

app = typer.Typer()


@app.command()
def wps_namelist(experiment_path: Path):
    """Generates the WPS namelist for the whole experiment period."""

    logger, _ = get_logger(LoggerConfig(experiment_path, "preprocess-wps-namelist"))
    cfg = config.read_config(experiment_path / "config.toml")

    wps_namelist = {
        "share": {
            "wrf_core": "ARW",
            "max_dom": 1,
            "start_date": cfg.time_control.start.strftime("%Y-%m-%d_%H:%M:%S"),
            "end_date": cfg.time_control.end.strftime("%Y-%m-%d_%H:%M:%S"),
            "interval_seconds": cfg.time_control.boundary_update_interval * 60,
        },
        "geogrid": {
            "parent_id": 1,
            "parent_grid_ratio": 1,
            "i_parent_start": 1,
            "j_parent_start": 1,
            "e_we": cfg.domain_control.xy_size[0],
            "e_sn": cfg.domain_control.xy_size[1],
            "geog_data_res": "default",
            "dx": cfg.domain_control.xy_resolution[0] * 1000,
            "dy": cfg.domain_control.xy_resolution[1] * 1000,
            "map_proj": cfg.domain_control.projection,
            "ref_lat": cfg.domain_control.ref_lat,
            "ref_lon": cfg.domain_control.ref_lon,
            "truelat1": cfg.domain_control.truelat1,
            "truelat2": cfg.domain_control.truelat2,
            "stand_lon": cfg.domain_control.stand_lon,
            "geog_data_path": cfg.data.wps_geog.resolve(),
        },
        "ungrib": {
            "out_format": "WPS",
            "prefix": "FILE",
        },
        "metgrid": {
            "fg_name": "FILE",
            "io_form_metgrid": 2,
        },
    }

    preprocess_dir = experiment_path / cfg.directories.work_sub / "preprocessing"
    namelist_path = preprocess_dir / "namelist.wps"
    namelist.write_namelist(wps_namelist, namelist_path)

    logger.info(f"Wrote WPS namelist to {namelist_path}")


@app.command()
def wrf_namelist(experiment_path: Path):
    """Generates the WRF namelist for the whole experiment period, for use by real.exe
    for creating initial and boundary conditions."""

    logger, _ = get_logger(LoggerConfig(experiment_path, "preprocess-wrf-namelist"))
    cfg = config.read_config(experiment_path / "config.toml")
    wrf_namelist = {
        "time_control": {
            **wrf.timedelta_to_namelist_items(
                cfg.time_control.end - cfg.time_control.start
            ),
            **wrf.datetime_to_namelist_items(cfg.time_control.start, "start"),
            **wrf.datetime_to_namelist_items(cfg.time_control.end, "end"),
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

    preprocess_dir = experiment_path / cfg.directories.work_sub / "preprocessing"
    namelist_path = preprocess_dir / "namelist.input"
    namelist.write_namelist(wrf_namelist, namelist_path)

    logger.info(f"Wrote WRF namelist to {namelist_path}")


@app.command()
def setup(experiment_path: Path):
    """
    Setups the preprocessing environment by copying WRF/WPS to the correct places and
    generating their namelists.
    """

    logger, _ = get_logger(LoggerConfig(experiment_path, "preprocess-setup"))
    cfg = config.read_config(experiment_path / "config.toml")
    preprocess_dir = experiment_path / cfg.directories.work_sub / "preprocessing"

    shutil.copytree(
        experiment_path / cfg.directories.work_sub / "WRF",
        preprocess_dir / "WRF",
    )
    logger.info(f"Copied WRF to {preprocess_dir / 'WRF'}")
    shutil.copytree(
        experiment_path / cfg.directories.work_sub / "WPS",
        preprocess_dir / "WPS",
    )
    logger.info(f"Copied WPS to {preprocess_dir / 'WPS'}")

    wps_namelist(experiment_path)
    wrf_namelist(experiment_path)

    logger.info("Preprocessing ready to run")


@app.command()
def geogrid(experiment_path: Path, force=False):
    """
    Runs geogrid.exe for the experiment.
    """

    logger, log_dir = get_logger(LoggerConfig(experiment_path, "preprocess-geogrid"))
    cfg = config.read_config(experiment_path / "config.toml")
    preprocessing_dir = experiment_path / cfg.directories.work_sub / "preprocessing"
    wps_dir = preprocessing_dir / "WPS"

    if (wps_dir / "geo_em.d01.nc").exists() and not force:
        logger.warning("geo_em.d01.nc already exists, skipping geogrid.exe")
        return 0

    shutil.copy(
        preprocessing_dir / "namelist.wps",
        wps_dir / "namelist.wps",
    )

    geogrid_path = wps_dir / "geogrid.exe"
    if not geogrid_path.is_file():
        console.log("Could not find geogrid.exe at {geogrid_path}")
        return 1

    cmd = [geogrid_path]
    res = utils.call_external_process(cmd, wps_dir, logger)
    (log_dir / "geogrid.log").write_text(res.stdout)
    if not res.success:
        logger.error("Error is fatal")
        return 1

    logger.info("Geogrid finished successfully!")
    logger.debug(f"stdout:\n{res.stdout.strip()}")

    return 0


@app.command()
def ungrib(experiment_path: Path, force=False):
    """
    Runs ungrib.exe for the experiment, after linking the grib files into the WPS directory
    """

    logger, log_dir = get_logger(LoggerConfig(experiment_path, "preprocess-ungrib"))
    cfg = config.read_config(experiment_path / "config.toml")
    preprocessing_dir = experiment_path / cfg.directories.work_sub / "preprocessing"
    wps_dir = preprocessing_dir / "WPS"
    data_dir = cfg.data.meteorology.resolve()

    for f in chain(wps_dir.glob("FILE:*"), wps_dir.glob("PFILE:*")):
        logger.debug(f"Removing old WPS intermediate file {f}")
        f.unlink()

    # Add namelist
    shutil.copy(
        preprocessing_dir / "namelist.wps",
        wps_dir / "namelist.wps",
    )

    # Link Vtable
    if cfg.data.meteorology_vtable.is_absolute():
        vtable_path = cfg.data.meteorology_vtable
    else:
        vtable_path = (
            wps_dir / "ungrib" / "Variable_Tables" / cfg.data.meteorology_vtable
        )
    if not vtable_path.is_file() or vtable_path.is_symlink():
        logger.error(f"Vtable {vtable_path} does not exist")
        return 1
    console.print(f"[green]Linking Vtable[/green] {vtable_path}")
    (wps_dir / "Vtable").unlink(missing_ok=True)
    (wps_dir / "Vtable").symlink_to(vtable_path)

    # Make symlinks for grib files
    for i, grib_file in enumerate(data_dir.glob(cfg.data.meteorology_glob)):
        link_path = wps_dir / f"GRIBFILE.{utils.int_to_letter_numeral(i + 1)}"
        if not link_path.is_symlink():
            if link_path.exists():
                link_path.unlink()
            link_path.symlink_to(grib_file)

            logger.debug(f"Created symlink for {grib_file} at {link_path}")
    logger.info(f"Linked {i+1} GRIB files to {wps_dir} from {data_dir}")

    # Run ungrib.exe
    ungrib_path = wps_dir / "ungrib.exe"
    if not ungrib_path.is_file():
        console.log("Could not find ungrib.exe at {ungrib_path}")
        return 1

    cmd = [ungrib_path]
    res = utils.call_external_process(cmd, wps_dir, logger)
    (log_dir / "ungrib.log").write_text(res.stdout)
    if not res.success or "Successful completion of ungrib" not in res.stdout:
        logger.error("Ungrib could not finish successfully")
        logger.error("Check the `ungrib.log` file for more info.")
        return 1

    logger.info("Ungrib finished successfully!")
    return 0


@app.command()
def metgrid(experiment_path: Path, force=False):
    """
    Run metgrid.exe to produce the `met_em*.nc` files.
    """

    logger, log_dir = get_logger(LoggerConfig(experiment_path, "preprocess-metgrid"))
    cfg = config.read_config(experiment_path / "config.toml")
    preprocessing_dir = experiment_path / cfg.directories.work_sub / "preprocessing"
    wps_dir = preprocessing_dir / "WPS"

    if len(list(wps_dir.glob("met_em*.nc"))) > 0 and not force:
        logger.warning("met_em files seem to exist, skipping metgrid.exe")
        return 0

    shutil.copy(
        preprocessing_dir / "namelist.wps",
        wps_dir / "namelist.wps",
    )

    metgrid_path = wps_dir / "metgrid.exe"
    if not metgrid_path.is_file():
        console.log("Could not find metgrid.exe at {metgrid_path}")
        return 1

    cmd = [metgrid_path]
    res = utils.call_external_process(cmd, wps_dir, logger)
    (log_dir / "metgrid.log").write_text(res.stdout)
    if not res.success or "Successful completion of metgrid" not in res.stdout:
        logger.error("Metgrid could not finish successfully")
        logger.error("Check the `metgrid.log` file for more info.")
        return 1

    logger.info("Metgrid finished successfully!")

    return 0


@app.command()
def real(experiment_path: Path, force=False):
    """
    Run real.exe to produce the initial (wrfinput) and boundary (wrfbdy) conditions.
    These are produces for the whole experiment duration (all cycles).
    You should split the wrfbdy file for each cycle using `split-wrfbdy`.
    """

    logger, log_dir = get_logger(LoggerConfig(experiment_path, "preprocess-real"))
    cfg = config.read_config(experiment_path / "config.toml")
    preprocessing_dir = experiment_path / cfg.directories.work_sub / "preprocessing"
    wps_dir = preprocessing_dir / "WPS"
    wrf_dir = preprocessing_dir / "WRF"

    if (
        (wrf_dir / "wrfinput_d01").exists()
        and (wrf_dir / "wrfbdy_d01").exists()
        and not force
    ):
        logger.warning("wrfinput and wrfbdy files seem to exist, skipping real.exe")
        return 0

    # Clean WRF dir from old met_em files
    for p in wrf_dir.glob("met_em*nc"):
        p.unlink()

    shutil.copy(
        preprocessing_dir / "namelist.input",
        wrf_dir / "namelist.input",
    )

    # Link met_em files to WRF directory
    count = 0
    for p in wps_dir.glob("met_em*nc"):
        count += 1
        target = wrf_dir / p.name
        target.symlink_to(p)
        logger.debug(f"Created symlink for {p} at {target}")

    if count == 0:
        logger.error("No met_em files found")
        return 1

    logger.info(f"Linked {count} met_em files to {wrf_dir}")

    # Run real
    real_path = wrf_dir / "real.exe"
    if not real_path.is_file():
        logger.error("[red]Could not find real.exe at[/red] {real_path}")
        return 1

    cmd = ["srun", real_path]  # TODO Make slurm configurable!
    res = utils.call_external_process(cmd, wrf_dir, logger)
    for log_file in wrf_dir.glob("rsl.*"):
        shutil.copy(log_file, log_dir / log_file.name)
    (log_dir / "real.log").write_text(res.stdout)

    if not res.success or "SUCCESS COMPLETE REAL_EM INIT" not in res.stdout:
        logger.error("real.exe could not complete, check logs.")
        return 1

    logger.info("real finished successfully")

    data_dir = preprocessing_dir / "initial_coundary_nc"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(
        wrf_dir / "wrfinput_d01",
        data_dir / "wrfinput_d01",
    )
    logger.info(f"Moved wrfinput_d01 to {data_dir}")
    shutil.move(
        wrf_dir / "wrfbdy_d01",
        data_dir / "wrfbdy_d01",
    )
    logger.info(f"Moved wrfbdy_d01 to {data_dir}")


@app.command()
def split_wrfbdy(
    experiment_path: Path,
):
    """
    Split the wrfbdy file into multiple files, one for each assimilation cycle.
    """

    logger, _ = get_logger(LoggerConfig(experiment_path, "preprocess-split-wrfbdy"))
    cfg = config.read_config(experiment_path / "config.toml")
    preprocessing_dir = experiment_path / cfg.directories.work_sub / "preprocessing"
    data_dir = preprocessing_dir / "initial_boundary_nc"

    wrfbdy_path = data_dir / "wrfbdy_d01"
    if not wrfbdy_path.is_file():
        logger.error("[red]Could not find wrfbdy_d01 at[/red] {wrfbdy_path}")
        return 1

    cycles = cycling.get_cycle_information(cfg)
    if len(cycles) == 0:
        logger.error("[time_control] configuration is invalid, no cycles found")
        return 1
    logger.info(f"Working for {len(cycles)} assimilation cycles")

    with netCDF4.Dataset(wrfbdy_path, "r") as ds_in:
        bdy_times = ds_in["Times"][:]
        bdy_times = [
            datetime.datetime.strptime(
                t.tobytes().decode(), "%Y-%m-%d_%H:%M:%S"
            ).replace(tzinfo=datetime.timezone.utc)
            for t in bdy_times.data
        ]

        for cycle in cycles:
            logger.info(
                f"Cycle {cycle.index}, start {cycle.start.isoformat()} end {cycle.end.isoformat()}"
            )

            time_mask = np.zeros(len(bdy_times), dtype=bool)
            for j, t in enumerate(bdy_times):
                time_mask[j] = cycle.start <= t <= cycle.end

            if not np.any(time_mask):
                logger.error(
                    f"No boundary data in wrfbdy found for cycle {cycle.index}!"
                )
                continue

            output_path = data_dir / f"{wrfbdy_path.name}_cycle_{cycle.index}"
            logger.info(f"Writing wrfbdy file for cycle {cycle.index} to {output_path}")
            with netCDF4.Dataset(output_path, "w") as ds_out:
                for name, dim in ds_in.dimensions.items():
                    size = dim.size
                    if name == "Time":
                        size = np.sum(time_mask)
                    ds_out.createDimension(name, size)

                for name, var in ds_in.variables.items():
                    ds_out.createVariable(name, var.dtype, var.dimensions)
                    ds_out[name][:] = var[time_mask]
                    for attr_name in var.ncattrs():
                        ds_out[name].setncattr(attr_name, var.getncattr(attr_name))
