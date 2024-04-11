import datetime as dt
import sys
from pathlib import Path
from typing import Optional

import click

from wrf_ensembly import config, cycling, experiment, observations
from wrf_ensembly.click_utils import pass_experiment_path
from wrf_ensembly.console import logger


@click.group(name="observations")
def observations_cli():
    pass


@observations_cli.command()
@click.argument("cycle", required=False, type=int)
@pass_experiment_path
def prepare(experiment_path: Path, cycle: Optional[int] = None):
    """Converts observation files to DART obs_seq format"""

    logger.setup("observations-prepare", experiment_path)
    exp = experiment.Experiment(experiment_path)

    # If a cycle is not given, we will convert for all cycles
    cycles = exp.cycles
    if cycle is not None:
        cycles = [c for c in cycles if c.index == cycle]

    # Prepare observation groups for all toml file
    obs_path = exp.paths.obs
    obs_groups = observations.read_observations(obs_path)
    names = list(obs_groups.keys())

    if len(names) == 0:
        logger.error("No observation groups found!")
        sys.exit(1)

    logger.info(f"Found observation groups: {', '.join(names)}")

    # Convert observation files for each cycle
    for c in cycles:
        logger.info(f"Converting observations for cycle {c.index}")
        logger.info(f"Cycle start: {c.start.isoformat()}")
        logger.info(f"Cycle end: {c.end.isoformat()}")

        # TODO Calculate window length!
        assimilation_window_start = c.end - dt.timedelta(minutes=30)
        assimilation_window_end = c.end + dt.timedelta(minutes=30)
        logger.info(
            f"Assimilation window start: {assimilation_window_start.isoformat()}"
        )
        logger.info(f"Assimilation window end: {assimilation_window_end.isoformat()}")

        cycle_files = []
        for key, obs_group in obs_groups.items():
            logger.info(f"Converting group {key}({obs_group.kind})")

            for i, file in enumerate(
                obs_group.get_files_in_window(
                    assimilation_window_start, assimilation_window_end
                )
            ):
                logger.info(f"Converting file {file.path} to obs_seq format")
                out = obs_path / f"cycle_{c.index}.{key}.{i}.obs_seq"
                obs_group.convert_file(file, out)
                cycle_files.append(out)

        if len(cycle_files) == 0:
            logger.warning("No observation files found for this cycle!")
            continue

        # Join files for this group
        logger.info(f"Joining files for cycle {c.index}")
        kinds = [v.kind for v in obs_groups.values()]
        observations.join_obs_seq(
            exp.cfg, cycle_files, obs_path / f"cycle_{c.index}.obs_seq", kinds
        )

        # Remove temporary files
        for f in cycle_files:
            f.unlink()


@observations_cli.command()
@click.argument("window_center", required=False, type=click.DateTime())
@click.argument(
    "output_path", required=True, type=click.Path(path_type=Path, writable=True)
)
@click.option("window_length", "-l", "--length", type=int, default=60)
@pass_experiment_path
def prepare_custom_window(
    experiment_path: Path,
    window_center: dt.datetime,
    output_path: Path,
    window_length: int,
):
    """Converts observation files to DART obs_seq format for a custom assimilation window"""

    logger.setup("observations-prepare-custom", experiment_path)
    exp = experiment.Experiment(experiment_path)

    # Prepare observation groups for all toml file
    obs_path = exp.paths.obs
    obs_groups = observations.read_observations(obs_path)
    names = list(obs_groups.keys())

    if len(names) == 0:
        logger.error("No observation groups found!")
        sys.exit(1)

    logger.info(f"Found observation groups: {', '.join(names)}")

    # Convert observation files for each cycle
    logger.info("Converting observations for custom assimilation window")
    window_center = window_center.replace(tzinfo=dt.timezone.utc)
    assimilation_window_start = window_center - dt.timedelta(minutes=window_length / 2)
    assimilation_window_end = window_center + dt.timedelta(minutes=window_length / 2)
    logger.info(f"Assimilation window start: {assimilation_window_start.isoformat()}")
    logger.info(f"Assimilation window end: {assimilation_window_end.isoformat()}")

    window_files = []
    for key, obs_group in obs_groups.items():
        logger.info(f"Converting group {key}({obs_group.kind})")

        for i, file in enumerate(
            obs_group.get_files_in_window(
                assimilation_window_start, assimilation_window_end
            )
        ):
            logger.info(f"Converting file {file.path} to obs_seq format")
            out = output_path.parent / f"{key}.{i}.obs_seq"
            obs_group.convert_file(file, out)
            window_files.append(out)

    if len(window_files) == 0:
        logger.warning("No observation files found for this window!")

    # Join files for this group
    logger.info(f"Joining files...")
    kinds = [v.kind for v in obs_groups.values()]
    observations.join_obs_seq(exp.cfg, window_files, output_path, kinds)

    # Remove temporary files
    for f in window_files:
        f.unlink()


@observations_cli.command()
@click.argument(
    "obs_seq_path", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.argument("nc_path", required=True, type=click.Path(path_type=Path))
@pass_experiment_path
def obs_seq_to_nc(experiment_path: Path, obs_seq_path: Path, nc_path: Path):
    """Convert the given obs_seq file to netCDF format"""

    exp = experiment.Experiment(experiment_path)
    logger.setup("observations-convert-obs-seq", experiment_path)

    observations.obs_seq_to_nc(exp, obs_seq_path, nc_path)
