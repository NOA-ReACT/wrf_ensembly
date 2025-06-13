import csv
import shutil
import sys
from importlib import resources
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Column, Table

from wrf_ensembly import config, cycling, experiment, utils
from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.console import logger


@click.group(name="experiment", cls=GroupWithStartEndPrint)
def experiment_cli():
    pass


@experiment_cli.command()
@click.argument("template", required=True)
@pass_experiment_path
def create(experiment_path: Path, template: str):
    """
    Create a new experiment directory at EXPERIMENT_PATH. The TEMPLATE argument specifies
    which config file template to use when creating the experiment. Available templates are
    located in `wrf_ensembly/config_templates/`. The template name should not include the `.toml`
    extension, e.g., `default` for `default.toml`.

    If in doubt, use the `iridium_chem_4.6.0` template and modify it to your needs.
    """

    logger.setup("experiment-create", experiment_path)

    # Create directory tree, add config file
    root = experiment_path
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as ex:
        logger.error(f"Could not create directory `{root}`: {ex}")
        sys.exit(1)

    with resources.path(
        "wrf_ensembly.config_templates", f"{template}.toml"
    ) as config_template_path:
        if config_template_path is None or not config_template_path.exists():
            available_files = [f.stem for f in config_template_path.parent.iterdir()]
            available_files = ", ".join(available_files)

            logger.error(f"Template `{template}` not found.")
            logger.error(f"Available templates: {available_files}")
            sys.exit(1)

        utils.copy(config_template_path, root / "config.toml")

    exp = experiment.Experiment(experiment_path)
    exp.paths.create_directories()
    exp.save_status_to_db()

    logger.info("Experiment created successfully!")


@experiment_cli.command()
@click.option(
    "--force",
    is_flag=True,
    help="If model directory already exists, remove and copy again",
)
@pass_experiment_path
def copy_model(experiment_path: Path, force: bool):
    """
    Setup the experiment by doing the following
    """

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
@click.option(
    "--to-csv", type=click.Path(path_type=Path), help="Write cycle info to CSV file"
)
@pass_experiment_path
def cycle_info(experiment_path: Path, to_csv: Optional[Path]):
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

    if to_csv is not None:
        with open(to_csv, "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(["cycle_i", "start", "end", "duration_h"])
            for c in cycles:
                writer.writerow(
                    [
                        c.index,
                        c.start.strftime("%Y-%m-%d %H:%M"),
                        c.end.strftime("%Y-%m-%d %H:%M"),
                        int((c.end - c.start).total_seconds() // 60 // 60),
                    ]
                )
