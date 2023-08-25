from pathlib import Path
import shutil
from typing_extensions import Annotated
from typing import Optional

import typer

from wrf_ensembly.console import console, get_logger, LoggerConfig
from wrf_ensembly import config, cycling, utils

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

    logger, _ = get_logger(LoggerConfig(experiment_path, "experiment-create"))

    # Create directory tree, add config file
    root = experiment_path
    root.mkdir(parents=True, exist_ok=True)

    if config_path is not None:
        cfg = config.read_config(config_path)
        utils.copy(config_path, experiment_path / "config.toml")
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

    output_dir = root / cfg.directories.output_sub
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis").mkdir()
    (output_dir / "forecasts").mkdir()
    (output_dir / "prior").mkdir()
    (output_dir / "initial_boundary").mkdir()

    for j in range(cfg.assimilation.n_members):
        member_path = (root / cfg.directories.work_sub) / "ensemble" / f"member_{j}"
        member_path.mkdir(parents=True)

    logger.info("Experiment created successfully!")


@app.command()
def copy_model(experiment_path: Path):
    """Setup the experiment (i.e., copy WRF/WPS, generate namelists, ...)"""

    logger, _ = get_logger(LoggerConfig(experiment_path, "experiment-copy-model"))
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
        member_path = work_dir / "ensemble" / f"member_{j}"
        shutil.copytree(work_dir / "WRF", member_path, dirs_exist_ok=True)
        logger.info(f"Copied WRF to {member_path}")
