from pathlib import Path
import datetime as dt
from typing import Optional

import typer

from wrf_ensembly.console import logger
from wrf_ensembly import (
    config,
    cycling,
    observations,
)


app = typer.Typer()


@app.command()
def prepare(experiment_path: Path, cycle: Optional[int] = None):
    """Converts observation files to DART obs_seq format"""

    logger.setup("observations-prepare", experiment_path)
    experiment_path = experiment_path.resolve()
    cfg = config.read_config(experiment_path / "config.toml")

    # If a cycle is not given, we will convert for all cycles
    cycles = cycling.get_cycle_information(cfg)
    if cycle is not None:
        cycles = [c for c in cycles if c.index == cycle]

    # Prepare observation groups for all toml file
    obs_path = experiment_path / "obs"
    obs_groups = observations.read_observations(obs_path)
    names = list(obs_groups.keys())

    if len(names) == 0:
        logger.error("No observation groups found!")
        raise typer.Exit(1)

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

        # Join files for this group
        logger.info(f"Joining files for cycle {c.index}")
        kinds = [v.kind for v in obs_groups.values()]
        observations.join_obs_seq(
            cfg, cycle_files, obs_path / f"cycle_{c.index}.obs_seq", kinds
        )

        # Remove temporary files
        for f in cycle_files:
            f.unlink()
