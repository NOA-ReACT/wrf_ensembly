from pathlib import Path
import shutil
from itertools import chain
from typing import Optional
from typing_extensions import Annotated

import typer

from wrf_ensembly.console import logger
from wrf_ensembly import config, cycling, namelist, wrf, utils, templates

app = typer.Typer()


@app.command()
def wps_namelist(experiment_path: Path):
    """Generates the WPS namelist for the whole experiment period."""

    logger.setup("preprocess-wps-namelist", experiment_path)
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
def wrf_namelist(
    experiment_path: Path, cycle: Annotated[Optional[int], typer.Argument()] = None
):
    """
    Generates the WRF namelist, for use by real.exe for creating initial and boundary conditions.
    If cycle is specified, the namelist will be generated for that cycle only.
    """

    logger.setup("preprocess-wrf-namelist", experiment_path)
    cfg = config.read_config(experiment_path / "config.toml")

    if cycle is None:
        start = cfg.time_control.start
        end = cfg.time_control.end

        logger.info("Generating WRF namelist for whole experiment")
    else:
        cycles = cycling.get_cycle_information(cfg)
        cur_cycle = cycles[cycle]
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

    logger.setup("preprocess-setup", experiment_path)
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
def geogrid(experiment_path: Path):
    """
    Runs geogrid.exe for the experiment.
    """

    logger.setup("preprocess-geogrid", experiment_path)
    cfg = config.read_config(experiment_path / "config.toml")
    preprocessing_dir = experiment_path / cfg.directories.work_sub / "preprocessing"
    wps_dir = preprocessing_dir / "WPS"

    if (wps_dir / "geo_em.d01.nc").exists():
        logger.warning("geo_em.d01.nc already exists, skipping geogrid.exe")
        return 0

    utils.copy(
        preprocessing_dir / "namelist.wps",
        wps_dir / "namelist.wps",
    )

    geogrid_path = wps_dir / "geogrid.exe"
    if not geogrid_path.is_file():
        logger.error("Could not find geogrid.exe at {geogrid_path}")
        return typer.Exit(1)

    res = utils.call_external_process([geogrid_path], wps_dir)
    if not res.success:
        logger.error("Error is fatal")
        return typer.Exit(1)

    logger.info("Geogrid finished successfully!")
    logger.debug(f"stdout:\n{res.stdout.strip()}")

    return 0


@app.command()
def ungrib(experiment_path: Path):
    """
    Runs ungrib.exe for the experiment, after linking the grib files into the WPS directory
    """

    logger.setup("preprocess-ungrib", experiment_path)
    cfg = config.read_config(experiment_path / "config.toml")
    preprocessing_dir = experiment_path / cfg.directories.work_sub / "preprocessing"
    wps_dir = preprocessing_dir / "WPS"
    data_dir = cfg.data.meteorology.resolve()

    for f in chain(
        wps_dir.glob("FILE:*"), wps_dir.glob("PFILE:*"), wps_dir.glob("GRIBFILE.*")
    ):
        logger.debug(f"Removing old WPS intermediate file/link {f}")
        f.unlink()

    # Add namelist
    utils.copy(
        preprocessing_dir / "namelist.wps",
        wps_dir / "namelist.wps",
    )

    # Link Vtable
    if cfg.data.meteorology_vtable.is_absolute():
        vtable_path = cfg.data.meteorology_vtable
    else:
        vtable_path = (
            wps_dir / "ungrib" / "Variable_Tables" / cfg.data.meteorology_vtable
        ).resolve()
    if not vtable_path.is_file() or vtable_path.is_symlink():
        logger.error(f"Vtable {vtable_path} does not exist")
        return typer.Exit(1)
    logger.info(f"[green]Linking Vtable[/green] {vtable_path}")
    (wps_dir / "Vtable").unlink(missing_ok=True)
    (wps_dir / "Vtable").symlink_to(vtable_path)

    # Make symlinks for grib files
    i = 0
    for i, grib_file in enumerate(data_dir.glob(cfg.data.meteorology_glob)):
        link_path = wps_dir / f"GRIBFILE.{utils.int_to_letter_numeral(i + 1)}"
        link_path.symlink_to(grib_file)
        logger.debug(f"Created symlink for {grib_file} at {link_path}")
    if i == 0:
        logger.error("No GRIB files found")
        return typer.Exit(1)
    logger.info(f"Linked {i+1} GRIB files to {wps_dir} from {data_dir}")

    # Run ungrib.exe
    ungrib_path = wps_dir / "ungrib.exe"
    if not ungrib_path.is_file():
        logger.error("Could not find ungrib.exe at {ungrib_path}")
        return typer.Exit(1)

    res = utils.call_external_process([ungrib_path], wps_dir)
    if not res.success or "Successful completion of ungrib" not in res.stdout:
        logger.error("Ungrib could not finish successfully")
        logger.error("Check the `ungrib.log` file for more info.")
        return typer.Exit(1)

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
    cfg = config.read_config(experiment_path / "config.toml")
    preprocessing_dir = experiment_path / cfg.directories.work_sub / "preprocessing"
    wps_dir = preprocessing_dir / "WPS"

    if len(list(wps_dir.glob("met_em*.nc"))) > 0:
        if not force:
            logger.warning("met_em files seem to exist, skipping metgrid.exe")
            return 0
        else:
            for f in wps_dir.glob("met_em*.nc"):
                logger.debug(f"Removing old met_em file {f}")
                f.unlink()

    utils.copy(
        preprocessing_dir / "namelist.wps",
        wps_dir / "namelist.wps",
    )

    metgrid_path = wps_dir / "metgrid.exe"
    if not metgrid_path.is_file():
        logger.error("Could not find metgrid.exe at {metgrid_path}")
        return typer.Exit(1)

    res = utils.call_external_process([metgrid_path], wps_dir)
    if not res.success or "Successful completion of metgrid" not in res.stdout:
        logger.error("Metgrid could not finish successfully")
        logger.error("Check the `metgrid.log` file for more info.")
        return typer.Exit(1)

    logger.info("Metgrid finished successfully!")

    return 0


@app.command()
def real(experiment_path: Path, cycle: int):
    """
    Run real.exe to produce the initial (wrfinput) and boundary (wrfbdy) conditions for
    one cycle. You should run this for all cycles to have initial/boundary conditions for
    your experiment.
    """

    logger.setup(f"preprocess-real-cycle_{cycle}", experiment_path)

    cfg = config.read_config(experiment_path / "config.toml")
    preprocessing_dir = experiment_path / cfg.directories.work_sub / "preprocessing"
    wps_dir = preprocessing_dir / "WPS"
    wrf_dir = preprocessing_dir / "WRF"

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
        return typer.Exit(1)

    logger.info(f"Linked {count} met_em files to {wrf_dir}")

    # Generate namelist
    wrf_namelist(experiment_path, cycle)
    utils.copy(
        preprocessing_dir / "namelist.input",
        wrf_dir / "namelist.input",
    )

    # Run real
    real_path = wrf_dir / "real.exe"
    if not real_path.is_file():
        logger.error("[red]Could not find real.exe at[/red] {real_path}")
        return typer.Exit(1)

    cmd = [
        "mpirun",
        "-n",
        "1",
        str(real_path.resolve()),
    ]  # TODO Make srun/mpirun configurable!
    utils.call_external_process(cmd, wrf_dir, "real.log")
    for log_file in wrf_dir.glob("rsl.*"):
        logger.add_log_file(log_file)

    rsl_path = wrf_dir / "rsl.out.0000"
    if not rsl_path.is_file():
        logger.error("Could not find rsl.out.0000, wrf did not execute probably.")
        return typer.Exit(1)
    else:
        rsl = rsl_path.read_text()
        if "SUCCESS COMPLETE REAL_EM INIT" not in rsl:
            logger.error("real.exe could not complete, check logs.")
            return typer.Exit(1)

    logger.info("real finished successfully")

    data_dir = experiment_path / cfg.directories.output_sub / "initial_boundary"
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
def jobfile(experiment_path: Path):
    """Creates a jobfile for running all preprocessing steps. Useful if you want to run WPS and real on your processing nodes."""

    logger.setup(f"preprocess-jobfile", experiment_path)
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")

    experiment_name = cfg.metadata.name
    cycles = cycling.get_cycle_information(cfg)

    # Write jobfile
    slurm_args = cfg.slurm
    env_modules = []
    if "env_modules" in slurm_args:
        env_modules = slurm_args["env_modules"]
        del slurm_args["env_modules"]
    slurm_args |= {"job-name": f"{experiment_name}__preprocess"}

    commands = [
        f"conda run -n wrf python -m wrf_ensembly preprocess setup {experiment_path}",
        f"conda run -n wrf python -m wrf_ensembly preprocess geogrid {experiment_path}",
        f"conda run -n wrf python -m wrf_ensembly preprocess ungrib {experiment_path}",
        f"conda run -n wrf python -m wrf_ensembly preprocess metgrid {experiment_path}",
    ] + [
        f"conda run -n wrf python -m wrf_ensembly preprocess real {experiment_path} {cycle}"
        for cycle in range(len(cycles))
    ]

    jobfile = experiment_path / "jobfiles" / "preprocess.sh"
    jobfile.parent.mkdir(parents=True, exist_ok=True)

    jobfile.write_text(
        templates.generate(
            "slurm_job.sh.j2",
            slurm_args=slurm_args,
            env_modules=env_modules,
            commands=commands,
        )
    )
    logger.info(f"Wrote jobfile to {jobfile}")
