import os
import shutil
import sys
from itertools import chain
from pathlib import Path
from typing import Optional

import click

from wrf_ensembly import external, utils, wrf
from wrf_ensembly.click_utils import pass_experiment_path
from wrf_ensembly.console import logger
from wrf_ensembly import experiment


@click.group(name="preprocess")
def preprocess_cli():
    pass


@preprocess_cli.command()
@click.option(
    "--only-namelist",
    is_flag=True,
    help="If set, do not copy WPS/WRF, only generate namelist",
    default=False,
)
@pass_experiment_path
def setup(experiment_path: Path, only_namelist: bool):
    """
    Setups the preprocessing environment by copying WRF/WPS to the correct places and
    generating their namelists.
    """

    logger.setup("preprocess-setup", experiment_path)
    exp = experiment.Experiment(experiment_path)

    if not only_namelist:
        shutil.copytree(exp.paths.work_wrf, exp.paths.work_preprocessing_wrf)
        logger.info(f"Copied WRF to {exp.paths.work_preprocessing_wrf}")
        shutil.copytree(exp.paths.work_wps, exp.paths.work_preprocessing_wps)
        logger.info(f"Copied WPS to {exp.paths.work_preprocessing_wps}")

    wrf.generate_wps_namelist(exp.cfg, exp.paths.work_preprocessing_wps)
    logger.info(f"Generated WPS namelist at {exp.paths.work_preprocessing_wps}")

    logger.info("Preprocessing ready to run")


@preprocess_cli.command()
@pass_experiment_path
def geogrid(experiment_path: Path):
    """
    Runs geogrid.exe for the experiment.
    """

    logger.setup("preprocess-geogrid", experiment_path)
    exp = experiment.Experiment(experiment_path)
    exp.set_wrf_environment()
    wps_dir = exp.paths.work_preprocessing_wps

    if (wps_dir / "geo_em.d01.nc").exists():
        logger.warning("geo_em.d01.nc already exists, skipping geogrid.exe")
        sys.exit(1)

    # Link the correct table
    if exp.cfg.geogrid.table is None:
        logger.error("No GEOGRID.TBL specified in config.toml")
        sys.exit(1)
    table_path = (wps_dir / "geogrid" / exp.cfg.geogrid.table).resolve()

    table_target = wps_dir / "geogrid" / "GEOGRID.TBL"
    table_target.unlink(missing_ok=True)
    table_target.symlink_to(table_path)
    logger.info(f"Linked {table_path} to {table_target}")

    geogrid_path = wps_dir / "geogrid.exe"
    if not geogrid_path.is_file():
        logger.error("Could not find geogrid.exe at {geogrid_path}")
        sys.exit(1)

    res = external.runc([geogrid_path], wps_dir, "geogrid.log")
    if res.returncode != 0:
        logger.error("Error is fatal")
        sys.exit(1)

    logger.info("Geogrid finished successfully!")
    logger.debug(f"stdout:\n{res.output}")

    return 0


@preprocess_cli.command()
@pass_experiment_path
def ungrib(experiment_path: Path):
    """
    Runs ungrib.exe for the experiment, after linking the grib files into the WPS directory
    """

    logger.setup("preprocess-ungrib", experiment_path)
    exp = experiment.Experiment(experiment_path)
    exp.set_wrf_environment()
    wps_dir = exp.paths.work_preprocessing_wps
    data_dir = exp.cfg.data.meteorology.resolve()

    for f in chain(
        wps_dir.glob("FILE:*"), wps_dir.glob("PFILE:*"), wps_dir.glob("GRIBFILE.*")
    ):
        logger.debug(f"Removing old WPS intermediate file/link {f}")
        f.unlink()

    # Link Vtable
    if exp.cfg.data.meteorology_vtable.is_absolute():
        vtable_path = exp.cfg.data.meteorology_vtable
    else:
        vtable_path = (
            wps_dir / "ungrib" / "Variable_Tables" / exp.cfg.data.meteorology_vtable
        ).resolve()
    if not vtable_path.is_file() or vtable_path.is_symlink():
        logger.error(f"Vtable {vtable_path} does not exist")
        sys.exit(1)
    logger.info(f"[green]Linking Vtable[/green] {vtable_path}")
    (wps_dir / "Vtable").unlink(missing_ok=True)
    (wps_dir / "Vtable").symlink_to(vtable_path)

    # Make symlinks for grib files
    i = -1
    for i, grib_file in enumerate(data_dir.glob(exp.cfg.data.meteorology_glob)):
        link_path = wps_dir / f"GRIBFILE.{utils.int_to_letter_numeral(i + 1)}"
        link_path.symlink_to(grib_file)
        logger.debug(f"Created symlink for {grib_file} at {link_path}")
    if i == -1:
        logger.error("No GRIB files found")
        sys.exit(1)
    logger.info(f"Linked {i+1} GRIB files to {wps_dir} from {data_dir}")

    # Run ungrib.exe
    ungrib_path = wps_dir / "ungrib.exe"
    if not ungrib_path.is_file():
        logger.error("Could not find ungrib.exe at {ungrib_path}")
        sys.exit(1)

    res = external.runc([ungrib_path], wps_dir, "ungrib.log")
    if res.returncode != 0 or "Successful completion of ungrib" not in res.output:
        logger.error("Ungrib could not finish successfully")
        logger.error("Check the `ungrib.log` file for more info.")
        sys.exit(1)

    logger.info("Ungrib finished successfully!")
    return 0


@preprocess_cli.command()
@click.option(
    "--force", is_flag=True, help="Force metgrid to run even if met_em files exist"
)
@pass_experiment_path
def metgrid(experiment_path: Path, force: bool):
    """
    Run metgrid.exe to produce the `met_em*.nc` files.
    """

    logger.setup("preprocess-metgrid", experiment_path)
    exp = experiment.Experiment(experiment_path)
    exp.set_wrf_environment()
    wps_dir = exp.paths.work_preprocessing_wps

    if len(list(wps_dir.glob("met_em*.nc"))) > 0:
        if not force:
            logger.warning("met_em files seem to exist, skipping metgrid.exe")
            sys.exit(0)
        else:
            for f in wps_dir.glob("met_em*.nc"):
                logger.debug(f"Removing old met_em file {f}")
                f.unlink()

    metgrid_path = wps_dir / "metgrid.exe"
    if not metgrid_path.is_file():
        logger.error(f"Could not find metgrid.exe at {metgrid_path}")
        sys.exit(1)

    res = external.runc([metgrid_path], wps_dir, "metgrid.log")
    if res.returncode != 0 or "Successful completion of metgrid" not in res.output:
        logger.error("Metgrid could not finish successfully")
        logger.error("Check the `metgrid.log` file for more info.")
        sys.exit(1)

    logger.info("Metgrid finished successfully!")


@preprocess_cli.command()
@click.argument("cycle", required=True, type=int)
@click.option(
    "--cores", type=int, help="Number of cores to use for real.exe", default=None
)
@pass_experiment_path
def real(experiment_path: Path, cycle: int, cores):
    """
    Run real.exe to produce the initial (wrfinput) and boundary (wrfbdy) conditions the
    given CYCLE. You should run this for all cycles to have initial/boundary conditions for
    your experiment.
    """

    logger.setup(f"preprocess-real-cycle_{cycle}", experiment_path)

    exp = experiment.Experiment(experiment_path)
    exp.set_wrf_environment()
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
        sys.exit(1)

    logger.info(f"Linked {count} met_em files to {wrf_dir}")

    # Generate namelist
    wrf.generate_wrf_namelist(
        exp.cfg, exp.cycles[cycle], False, wrf_dir / "namelist.input"
    )
    logger.info(f"Generated namelist at {wrf_dir / 'namelist.input'}")

    # Determine number of cores
    if cores is None:
        if "SLURM_NTASKS" in os.environ:
            cores = int(os.environ["SLURM_NTASKS"])
        else:
            cores = 1
    logger.info(f"Using {cores} cores for real.exe")

    # Clean up old log files
    for log_file in wrf_dir.glob("rsl.*"):
        logger.debug(f"Removing old log file {log_file}")
        log_file.unlink()

    # Run real
    real_path = wrf_dir / "real.exe"
    if not real_path.is_file():
        logger.error("[red]Could not find real.exe at[/red] {real_path}")
        sys.exit(1)

    cmd = [
        *exp.cfg.slurm.mpirun_command.split(" "),
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
        sys.exit(1)
    else:
        rsl = rsl_path.read_text()
        if "SUCCESS COMPLETE REAL_EM INIT" not in rsl:
            logger.error("real.exe could not complete, check logs.")
            logger.error(
                "Last 50 lines of rsl.out.0000:\n" + "\n".join(rsl.split("\n")[-50:])
            )

            rsl = (wrf_dir / "rsl.error.0000").read_text()
            logger.error(
                "Last 50 lines of rsl.error.0000:\n" + "\n".join(rsl.split("\n")[-50:])
            )
            sys.exit(1)

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


@preprocess_cli.command()
@click.option(
    "--jobs", type=int, help="Number of processes to use (also respects SLURM_NTASKS)"
)
@pass_experiment_path
def interpolate_chem(experiment_path: Path, jobs: Optional[int]):
    """
    Uses `interpolator-for-wrfchem` to interpolate the chemical initial conditions onto the WRF domain.

    Args:
        jobs: How many processes to use when interpolating the chemistry fields.
    """

    logger.setup("preprocess-interpolate-chem", experiment_path)
    exp = experiment.Experiment(experiment_path)

    # Check if a path is provided for chemistry fields
    if exp.cfg.data.chemistry is None:
        logger.error(
            "No chemistry data path provided in the configuration [data.chemistry]"
        )
        sys.exit(1)
    chem = exp.cfg.data.chemistry

    # Check if the mapping file exists
    mapping_path = experiment_path / "species_map.toml"
    if not mapping_path.is_file():
        logger.error(f"Species mapping file {mapping_path.resolve()} does not exist")
        sys.exit(1)

    # Disable HDF5 locking since we are taking care only to write from one process at each file
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # Good luck!

    # We need to run interpolator one time for the initial conditions and one time for each cycle's boundary conditions.
    # The necessary commands are gathered inside the array to run in parallel.
    commands = []

    met_em_path = exp.paths.work_preprocessing_wps
    wrfinput_path = exp.paths.data_icbc / "wrfinput_d01_cycle_0"
    commands.append(
        external.ExternalProcess(
            [
                "interpolator-for-wrfchem",
                chem.model_name,
                chem.path,
                met_em_path,
                mapping_path,
                wrfinput_path,
            ],
            log_filename="interpolator_wrfinput.log",
        )
    )
    for cycle in exp.cycles:
        wrfbdy_path = exp.paths.data_icbc / f"wrfbdy_d01_cycle_{cycle.index}"
        commands.append(
            external.ExternalProcess(
                [
                    "interpolator-for-wrfchem",
                    chem.model_name,
                    chem.path,
                    met_em_path,
                    mapping_path,
                    wrfinput_path,
                    f"--wrfbdy={wrfbdy_path}",
                    "--no-ic",
                ],
                log_filename=f"interpolator_wrfbdy_cycle_{cycle.index}.log",
            )
        )

    failure = False
    logger.info(f"Running interpolator-for-wrfchem with {jobs} jobs")
    if jobs is None:
        if "SLURM_NTASKS" in os.environ:
            jobs = int(os.environ["SLURM_NTASKS"])
            logger.info(f"Using {jobs} jobs from SLURM_NTASKS")
        else:
            jobs = 1
            logger.warning(
                "No job count specified (--jobs or SLURM_NTASKS), running with 1 job"
            )
    for res in external.run_in_parallel(commands, jobs):
        if res.returncode != 0:
            logger.error(
                f"interpolator-for-wrfchem failed with exit code {res.returncode}"
            )
            logger.error(res.output)
            failure = True

    if failure:
        logger.error("One or more interpolator-for-wrfchem commands failed, exiting")
        sys.exit(1)


@preprocess_cli.command()
@pass_experiment_path
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
