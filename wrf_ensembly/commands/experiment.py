from genericpath import exists
import shutil
from pathlib import Path
from typing import Optional
import typing_extensions

import typer
from rich.console import Console
from rich.table import Column, Table
from typing_extensions import Annotated

from wrf_ensembly import config, cycling, experiment, utils
from wrf_ensembly.console import logger

app = typer.Typer()


@app.command()
def create(
    experiment_path: Annotated[
        Path,
        typer.Argument(..., help="Where the experiment directory should be created"),
    ],
    config_path: Annotated[
        Optional[Path],
        typer.Option(
            ...,
            help="Path to the config file to use for the experiment. If not specified, the default config file will be used.",
        ),
    ] = None,
):
    """Create a new experiment directory."""

    logger.setup("experiment-create", experiment_path)

    # Create directory tree, add config file
    root = experiment_path
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        logger.error(f"Could not create directory `{root}`: {ex}")
        raise typer.Exit(1)

    if config_path is not None:
        utils.copy(config_path, experiment_path / "config.toml")
    else:
        # TODO Fix this, doesn't work like that
        # We gotta create a default config file
        raise NotImplementedError("Default config file not implemented yet")
        cfg = config.Config()
        config_path = experiment_path / "config.toml"
        config.write_config(config_path, cfg)

    exp = experiment.Experiment(experiment_path)

    # Create sub-directories
    exp.paths.obs.mkdir()
    exp.paths.work.mkdir()
    exp.paths.work_preprocessing.mkdir()
    exp.paths.jobfiles.mkdir()

    exp.paths.data.mkdir()
    exp.paths.data_analysis.mkdir()
    exp.paths.data_forecasts.mkdir()
    exp.paths.data_dart.mkdir()
    exp.paths.data_icbc.mkdir()
    exp.paths.data_diag.mkdir()

    exp.paths.logs.mkdir(exist_ok=True)
    exp.paths.logs_slurm.mkdir()


    logger.info("Experiment created successfully!")


@app.command()
def copy_model(
    experiment_path: Path,
    force: Annotated[
        bool,
        typer.Option(
            ..., help="If model directory already exists, remove and copy again"
        ),
    ] = False,
):
    """Setup the experiment (i.e., copy WRF/WPS, generate namelists, ...)"""

    logger.setup("experiment-copy-model", experiment_path)
    exp = experiment.Experiment(experiment_path)

    # Check if WRF/WPS directories already exist
    if exp.paths.work_wrf.exists():
        if force:
            logger.info(f"Removing existing WRF directory {exp.paths.work_wrf}")
            shutil.rmtree(exp.paths.work_wrf)
        else:
            logger.error(f"WRF directory {exp.paths.work_wrf} already exists")
            raise typer.Exit(1)
    if exp.paths.work_wps.exists():
        if force:
            logger.info(f"Removing existing WPS directory {exp.paths.work_wps}")
            shutil.rmtree(exp.paths.work_wps)
        else:
            logger.error(f"WPS directory {exp.paths.work_wps} already exists")
            raise typer.Exit(1)

    # Copy WRF/WPS in the work directory
    shutil.copytree(
        exp.cfg.directories.wrf_root / "run",
        exp.paths.work_wrf,
        symlinks=False,  # Maybe fix symlinks so that they are valid after getting copied?
    )
    logger.info(f"Copied WRF to {exp.paths.work_wrf}")

    shutil.copytree(exp.cfg.directories.wps_root, exp.paths.work_wps, symlinks=True)
    logger.info(f"Copied WPS to {exp.paths.work_wps}")

    for j in range(exp.cfg.assimilation.n_members):
        member_dir = exp.paths.member_path(j)

        if member_dir.exists():
            if force:
                logger.info(f"Removing existing member directory {member_dir}")
                shutil.rmtree(member_dir)
            else:
                logger.error(f"Member directory {member_dir} already exists")
                raise typer.Exit(1)

        shutil.copytree(exp.paths.work_wrf, member_dir, dirs_exist_ok=True)
        logger.info(f"Copied WRF to {member_dir}")


@app.command()
def cycle_info(experiment_path: Path):
    """Prints cycle information"""

    logger.setup("experiment-cycle-info", experiment_path)
    cfg = config.read_config(experiment_path / "config.toml")

    cycles = cycling.get_cycle_information(cfg)

    table = Table(
        "i",
        "Start",
        "End",
        Column(header="Duration", justify="right"),
        title="Cycle information",
    )

    for c in cycles:
        table.add_row(
            str(c.index),
            c.start.strftime("%Y-%m-%d %H:%M"),
            c.end.strftime("%Y-%m-%d %H:%M"),
            utils.seconds_to_pretty_hours((c.end - c.start).total_seconds()),
        )

    Console().print(table)
