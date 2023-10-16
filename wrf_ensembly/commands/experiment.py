from pathlib import Path
import shutil
from typing_extensions import Annotated
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table, Column

from wrf_ensembly.console import logger
from wrf_ensembly import config, utils, cycling

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
    root.mkdir(parents=True, exist_ok=True)

    if config_path is not None:
        utils.copy(config_path, experiment_path / "config.toml")
        cfg = config.read_config(experiment_path / "config.toml")
    else:
        # TODO Fix this, doesn't work like that
        # We gotta create a default config file
        cfg = config.Config()
        config_path = experiment_path / "config.toml"
        config.write_config(config_path, cfg)

    # Create sub-directories
    (root / cfg.directories.observations_sub).mkdir(parents=True, exist_ok=True)
    (root / cfg.directories.work_sub).mkdir(parents=True, exist_ok=True)
    (root / cfg.directories.work_sub / "preprocessing").mkdir()
    (root / "jobfiles").mkdir()

    output_dir = root / cfg.directories.output_sub
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis").mkdir()
    (output_dir / "forecasts").mkdir()
    (output_dir / "dart").mkdir()
    (output_dir / "initial_boundary").mkdir()
    (output_dir / "initial_boundary" / "cycled_inputs").mkdir()

    for j in range(cfg.assimilation.n_members):
        member_dir = cfg.get_member_dir(j)
        member_dir.mkdir(parents=True)

    logger.info("Experiment created successfully!")


@app.command()
def copy_model(experiment_path: Path):
    """Setup the experiment (i.e., copy WRF/WPS, generate namelists, ...)"""

    logger.setup("experiment-copy-model", experiment_path)
    cfg = config.read_config(experiment_path / "config.toml")
    work_dir = experiment_path / cfg.directories.work_sub

    # Copy WRF/WPS in the work directory
    shutil.copytree(
        cfg.directories.wrf_root / "run",
        work_dir / "WRF",
        symlinks=False,  # Maybe fix symlinks so that they are valid after getting copied?
    )
    logger.info(f"Copied WRF to {work_dir / 'WRF'}")

    shutil.copytree(cfg.directories.wps_root, work_dir / "WPS", symlinks=True)
    logger.info(f"Copied WPS to {work_dir / 'WPS'}")

    for j in range(cfg.assimilation.n_members):
        member_dir = cfg.get_member_dir(j)
        shutil.copytree(work_dir / "WRF", member_dir, dirs_exist_ok=True)
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
