import os
import shutil
from itertools import chain
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated
from wrf_ensembly import config, experiment, external, namelist, utils, wrf
from wrf_ensembly.console import logger

app = typer.Typer()


@app.command()
def wps_namelist(experiment_path: Path):
    """Generates the WPS namelist for the whole experiment period."""

    logger.setup("preprocess-wps-namelist", experiment_path)
    exp = experiment.Experiment(experiment_path)
    cfg = exp.cfg

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

    preprocess_dir = exp.paths.work_preprocessing
    namelist_path = preprocess_dir / "namelist.wps"
    namelist.write_namelist(wps_namelist, namelist_path)

    logger.info(f"Wrote WPS namelist to {namelist_path}")


@app.command()
def wrf_namelist(
    experiment_path: Path, cycle: Annotated[Optional[int], typer.Argument()] = None
):
    """
    Generates the WRF namelist, for use by real.exe for creating initial and boundary conditions.
    If cycle is specified, the namelist will be generated for that cycle only.
    """

    logger.setup("preprocess-wrf-namelist", experiment_path)
    exp = experiment.Experiment(experiment_path)
    cfg = config.read_config(experiment_path / "config.toml")

    if cycle is None:
        start = cfg.time_control.start
        end = cfg.time_control.end

        logger.info("Generating WRF namelist for whole experiment")
    else:
        cur_cycle = exp.cycles[cycle]
        start = cur_cycle.start
        end = cur_cycle.end

        logger.info(f"Generating WRF namelist for cycle {cycle}")

    logger.info(f"From {start.isoformat()} until {end.isoformat()}")

    wrf_namelist = {
        "time_control": {
            **wrf.timedelta_to_namelist_items(end - start),
            **wrf.datetime_to_namelist_items(start, "start"),
            **wrf.datetime_to_namelist_items(end, "end"),
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

    # chem_in_opt override for chemical initial conditions
    if cfg.data.manage_chem_ic:
        if "chem" in wrf_namelist:
            # If the user has already set chem_in_opt, warn about overriding
            if "chem_in_opt" in wrf_namelist["chem"]:
                logger.warning(
                    "chem_in_opt already set in WRF namelist, overriding with 0. Check `Data.manage_chem_ic` in config.toml to disable this."
                )

            wrf_namelist["chem"]["chem_in_opt"] = 0
        else:
            wrf_namelist["chem"] = {"chem_in_opt": 0}

    preprocess_dir = exp.paths.work_preprocessing
    namelist_path = preprocess_dir / "namelist.input"
    namelist.write_namelist(wrf_namelist, namelist_path)

    logger.info(f"Wrote WRF namelist to {namelist_path}")


@app.command()
def setup(experiment_path: Path):
    """
    Setups the preprocessing environment by copying WRF/WPS to the correct places and
    generating their namelists.
    """

    logger.setup("preprocess-setup", experiment_path)
    exp = experiment.Experiment(experiment_path)

    shutil.copytree(exp.paths.work_wrf, exp.paths.work_preprocessing_wrf)
    logger.info(f"Copied WRF to {exp.paths.work_preprocessing_wrf}")
    shutil.copytree(exp.paths.work_wps, exp.paths.work_preprocessing_wps)
    logger.info(f"Copied WPS to {exp.paths.work_preprocessing_wps}")

    wps_namelist(experiment_path)
    wrf_namelist(experiment_path)

    logger.info("Preprocessing ready to run")


@app.command()
def geogrid(experiment_path: Path):
    """
    Runs geogrid.exe for the experiment.
    """

    logger.setup("preprocess-geogrid", experiment_path)
    exp = experiment.Experiment(experiment_path)
    wps_dir = exp.paths.work_preprocessing_wps

    if (wps_dir / "geo_em.d01.nc").exists():
        logger.warning("geo_em.d01.nc already exists, skipping geogrid.exe")
        raise typer.Exit(1)

    # Link the correct table
    if exp.cfg.geogrid.table is None:
        logger.error("No GEOGRID.TBL specified in config.toml")
        raise typer.Exit(1)
    table_path = (wps_dir / "geogrid" / exp.cfg.geogrid.table).resolve()

    table_target = wps_dir / "geogrid" / "GEOGRID.TBL"
    table_target.unlink(missing_ok=True)
    table_target.symlink_to(table_path)
    print("table_target", table_target)
    logger.info(f"Linked {table_path} to {table_target}")

    utils.copy(exp.paths.work_preprocessing / "namelist.wps", wps_dir / "namelist.wps")

    geogrid_path = wps_dir / "geogrid.exe"
    if not geogrid_path.is_file():
        logger.error("Could not find geogrid.exe at {geogrid_path}")
        raise typer.Exit(1)

    res = external.runc([geogrid_path], wps_dir, "geogrid.log")
    if res.returncode != 0:
        logger.error("Error is fatal")
        raise typer.Exit(1)

    logger.info("Geogrid finished successfully!")
    logger.debug(f"stdout:\n{res.output}")

    return 0


@app.command()
def ungrib(experiment_path: Path):
    """
    Runs ungrib.exe for the experiment, after linking the grib files into the WPS directory
    """

    logger.setup("preprocess-ungrib", experiment_path)
    exp = experiment.Experiment(experiment_path)
    wps_dir = exp.paths.work_preprocessing_wps
    data_dir = exp.cfg.data.meteorology.resolve()

    for f in chain(
        wps_dir.glob("FILE:*"), wps_dir.glob("PFILE:*"), wps_dir.glob("GRIBFILE.*")
    ):
        logger.debug(f"Removing old WPS intermediate file/link {f}")
        f.unlink()

    # Add namelist
    utils.copy(exp.paths.work_preprocessing / "namelist.wps", wps_dir / "namelist.wps")

    # Link Vtable
    if exp.cfg.data.meteorology_vtable.is_absolute():
        vtable_path = exp.cfg.data.meteorology_vtable
    else:
        vtable_path = (
            wps_dir / "ungrib" / "Variable_Tables" / exp.cfg.data.meteorology_vtable
        ).resolve()
    if not vtable_path.is_file() or vtable_path.is_symlink():
        logger.error(f"Vtable {vtable_path} does not exist")
        raise typer.Exit(1)
    logger.info(f"[green]Linking Vtable[/green] {vtable_path}")
    (wps_dir / "Vtable").unlink(missing_ok=True)
    (wps_dir / "Vtable").symlink_to(vtable_path)

    # Make symlinks for grib files
    i = 0
    for i, grib_file in enumerate(data_dir.glob(exp.cfg.data.meteorology_glob)):
        link_path = wps_dir / f"GRIBFILE.{utils.int_to_letter_numeral(i + 1)}"
        link_path.symlink_to(grib_file)
        logger.debug(f"Created symlink for {grib_file} at {link_path}")
    if i == 0:
        logger.error("No GRIB files found")
        raise typer.Exit(1)
    logger.info(f"Linked {i+1} GRIB files to {wps_dir} from {data_dir}")

    # Run ungrib.exe
    ungrib_path = wps_dir / "ungrib.exe"
    if not ungrib_path.is_file():
        logger.error("Could not find ungrib.exe at {ungrib_path}")
        raise typer.Exit(1)

    res = external.runc([ungrib_path], wps_dir, "ungrib.log")
    if res.returncode != 0 or "Successful completion of ungrib" not in res.output:
        logger.error("Ungrib could not finish successfully")
        logger.error("Check the `ungrib.log` file for more info.")
        raise typer.Exit(1)

    logger.info("Ungrib finished successfully!")
    return 0


@app.command()
def metgrid(
    experiment_path: Path, force: Annotated[Optional[bool], typer.Option()] = False
):
    """
    Run metgrid.exe to produce the `met_em*.nc` files.
    """

    logger.setup("preprocess-metgrid", experiment_path)
    exp = experiment.Experiment(experiment_path)
    wps_dir = exp.paths.work_preprocessing_wps

    if len(list(wps_dir.glob("met_em*.nc"))) > 0:
        if not force:
            logger.warning("met_em files seem to exist, skipping metgrid.exe")
            raise typer.Exit(0)
        else:
            for f in wps_dir.glob("met_em*.nc"):
                logger.debug(f"Removing old met_em file {f}")
                f.unlink()

    utils.copy(
        exp.paths.work_preprocessing / "namelist.wps",
        wps_dir / "namelist.wps",
    )

    metgrid_path = wps_dir / "metgrid.exe"
    if not metgrid_path.is_file():
        logger.error(f"Could not find metgrid.exe at {metgrid_path}")
        raise typer.Exit(1)

    res = external.runc([metgrid_path], wps_dir, "metgrid.log")
    if res.returncode != 0 or "Successful completion of metgrid" not in res.output:
        logger.error("Metgrid could not finish successfully")
        logger.error("Check the `metgrid.log` file for more info.")
        raise typer.Exit(1)

    logger.info("Metgrid finished successfully!")


@app.command()
def real(experiment_path: Path, cycle: int, cores=None):
    """
    Run real.exe to produce the initial (wrfinput) and boundary (wrfbdy) conditions for
    one cycle. You should run this for all cycles to have initial/boundary conditions for
    your experiment.
    """

    logger.setup(f"preprocess-real-cycle_{cycle}", experiment_path)

    exp = experiment.Experiment(experiment_path)
    wps_dir = exp.paths.work_preprocessing_wps
    wrf_dir = exp.paths.work_preprocessing_wrf

    # Clean WRF dir from old met_em files
    for p in wrf_dir.glob("met_em*nc"):
        p.unlink()

    # Link met_em files to WRF directory
    count = 0
    for p in wps_dir.glob("met_em*nc"):
        count += 1
        target = wrf_dir / p.name
        target.symlink_to(p.resolve())
        logger.debug(f"Created symlink for {p} at {target}")

    if count == 0:
        logger.error("No met_em files found")
        raise typer.Exit(1)

    logger.info(f"Linked {count} met_em files to {wrf_dir}")

    # Generate namelist
    wrf_namelist(experiment_path, cycle)
    utils.copy(
        exp.paths.work_preprocessing / "namelist.input",
        wrf_dir / "namelist.input",
    )

    # Determine number of cores
    if cores is None:
        if "SLURM_NTASKS" in os.environ:
            cores = int(os.environ["SLURM_NTASKS"])
        else:
            cores = 1
    logger.info("Using {cores} cores for real.exe")

    # Run real
    real_path = wrf_dir / "real.exe"
    if not real_path.is_file():
        logger.error("[red]Could not find real.exe at[/red] {real_path}")
        raise typer.Exit(1)

    cmd = [
        exp.cfg.slurm.mpirun_command,
        "-n",
        str(cores),
        str(real_path.resolve()),
    ]
    external.runc(cmd, wrf_dir, "real.log")
    for log_file in wrf_dir.glob("rsl.*"):
        logger.add_log_file(log_file)

    rsl_path = wrf_dir / "rsl.out.0000"
    if not rsl_path.is_file():
        logger.error("Could not find rsl.out.0000, wrf did not execute probably.")
        raise typer.Exit(1)
    else:
        rsl = rsl_path.read_text()
        if "SUCCESS COMPLETE REAL_EM INIT" not in rsl:
            logger.error("real.exe could not complete, check logs.")
            raise typer.Exit(1)

    logger.info("real finished successfully")

    data_dir = exp.paths.data_icbc
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(
        wrf_dir / "wrfinput_d01",
        data_dir / f"wrfinput_d01_cycle_{cycle}",
    )
    logger.info(f"Moved wrfinput_d01 to {data_dir}")
    shutil.move(
        wrf_dir / "wrfbdy_d01",
        data_dir / f"wrfbdy_d01_cycle_{cycle}",
    )
    logger.info(f"Moved wrfbdy_d01 to {data_dir}")

    shutil.copyfile(
        wrf_dir / "namelist.input", data_dir / f"namelist.input_cycle_{cycle}"
    )


@app.command()
def clean(experiment_path: Path):
    """
    Deletes the preprocessing directory and all its contents. Specifically removes:
    - One copy of WPS and WRF
    - Intermediate files (FILE_* and GRIBFILE.*)
    - met_em files
    """

    logger.setup("preprocess-clean", experiment_path)
    exp = experiment.Experiment(experiment_path)

    logger.info(f"Removing {exp.paths.work_preprocessing}")
    shutil.rmtree(exp.paths.work_preprocessing)
    exp.paths.work_preprocessing.mkdir(parents=True, exist_ok=True)
