import shutil
import sys
from pathlib import Path

import click
import pkg_resources
from rich.console import Console
from rich.table import Column, Table

from wrf_ensembly import config, cycling, experiment, utils
from wrf_ensembly.click_utils import pass_experiment_path
from wrf_ensembly.console import logger


@click.group(name="experiment")
def experiment_cli():
    pass


@experiment_cli.command()
@click.argument(
    "template",
    required=True,
    type=click.STRING,
)
@pass_experiment_path
def create(experiment_path: Path, template: str):
    """Create a new experiment directory."""
    logger.setup("experiment-create", experiment_path)

    # Create directory tree, add config file
    root = experiment_path
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        logger.error(f"Could not create directory `{root}`: {ex}")
        sys.exit(1)

    config_template_path = pkg_resources.resource_filename(
        "wrf_ensembly", f"config_templates/{template}.toml"
    )
    config_template_path = Path(config_template_path)
    utils.copy(config_template_path, root / "config.toml")

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


@experiment_cli.command()
@click.option(
    "--force",
    is_flag=True,
    help="If model directory already exists, remove and copy again",
)
@pass_experiment_path
def copy_model(experiment_path: Path, force: bool):
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
            sys.exit(1)
    if exp.paths.work_wps.exists():
        if force:
            logger.info(f"Removing existing WPS directory {exp.paths.work_wps}")
            shutil.rmtree(exp.paths.work_wps)
        else:
            logger.error(f"WPS directory {exp.paths.work_wps} already exists")
            sys.exit(1)

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
                sys.exit(1)

        shutil.copytree(exp.paths.work_wrf, member_dir, dirs_exist_ok=True)
        logger.info(f"Copied WRF to {member_dir}")


@experiment_cli.command()
@pass_experiment_path
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
