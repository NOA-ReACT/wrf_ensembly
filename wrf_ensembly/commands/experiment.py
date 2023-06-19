from pathlib import Path
import shutil
from typing_extensions import Annotated
from typing import Optional

import typer

from wrf_ensembly.console import console, get_logger, LoggerConfig
from wrf_ensembly import config, cycling

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
    exists_ok: Annotated[
        bool,
        typer.Option(
            default=False,
            help="If set, do not raise an error if the experiment path already exists.",
        ),
    ] = False,
):
    """Create a new experiment directory."""

    logger, _ = get_logger()

    if not exists_ok and experiment_path.exists():
        logger.error(f"Experiment path {experiment_path} already exists")
        raise typer.Exit(code=1)

    # Create directory tree, add config file
    root = experiment_path
    root.mkdir(parents=True, exist_ok=True)

    if config_path is not None:
        cfg = config.read_config(config_path)
        shutil.copyfile(config_path, experiment_path / "config.toml")
    else:
        # TODO Fix this, doesn't work like that
        # We gotta create a default config file
        cfg = config.Config()
        config_path = experiment_path / "config.toml"
        config.write_config(config_path, cfg)

    # Create sub-directories
    (root / cfg.directories.output_sub).mkdir(parents=True, exist_ok=True)
    (root / cfg.directories.observations_sub).mkdir(parents=True, exist_ok=True)
    (root / cfg.directories.work_sub).mkdir(parents=True, exist_ok=True)
    (root / cfg.directories.log_sub).mkdir(parents=True, exist_ok=True)

    (root / cfg.directories.work_sub / "preprocessing").mkdir()

    cycles = cycling.get_cycle_information(cfg)
    for i in range(len(cycles)):
        cycle_dir = root / cfg.directories.output_sub / f"cycle_T{i}"
        cycle_dir.mkdir()
        cycle_log_dir = root / cfg.directories.log_sub / f"cycle_T{i}"
        cycle_log_dir.mkdir()

        for j in range(cfg.assimilation.n_members):
            (cycle_dir / f"member_{j}").mkdir()
            (cycle_dir / "mean").mkdir()
            (cycle_log_dir / f"member_{j}").mkdir()

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
