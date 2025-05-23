import datetime as dt
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Column, Table

from wrf_ensembly import experiment, external, observations, utils
from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.console import logger


@click.group(name="observations", cls=GroupWithStartEndPrint)
def observations_cli():
    pass


@observations_cli.command()
@click.argument("cycle", required=False, type=int)
@click.option(
    "--jobs",
    type=click.IntRange(min=0, max=None),
    help="How many files to process in parallel",
)
@pass_experiment_path
def prepare(experiment_path: Path, cycle: Optional[int], jobs: Optional[int]):
    """Converts observation files to DART obs_seq format"""

    logger.setup("observations-prepare", experiment_path)
    exp = experiment.Experiment(experiment_path)
    exp.set_dart_environment()

    jobs = utils.determine_jobs(jobs)
    logger.info(f"Using {jobs} jobs")

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

    # Gather all files for conversion
    cycle_files = {}
    for c in cycles:
        cycle_files[c.index] = {}

        # TODO Calculate window length!
        assimilation_window_start = c.end - dt.timedelta(minutes=30)
        assimilation_window_end = c.end + dt.timedelta(minutes=30)

        for key, obs_group in obs_groups.items():
            cycle_files[c.index][key] = []

            for i, file in enumerate(
                obs_group.get_files_in_window(
                    assimilation_window_start, assimilation_window_end
                )
            ):
                out = obs_path / f"cycle_{c.index}.{key}.{i}.obs_seq"
                cycle_files[c.index][key].append((file.path, out))

    # Print information about the files to convert
    table = Table(
        "Cycle",
        "Window start",
        "Window end",
        *(Column(header=k, justify="right") for k in names),
        title="Observation files to convert per cycle",
    )
    for c in cycles:
        table.add_row(
            str(c.index),
            c.start.strftime("%Y-%m-%d %H:%M"),
            c.end.strftime("%Y-%m-%d %H:%M"),
            *(str(len(cycle_files[c.index][k])) for k in names),
        )
    Console().print(table)

    # Convert observation files for each cycle in parallel
    commands = []
    for c in cycles:
        for key, obs_group in obs_groups.items():
            for file, out in cycle_files[c.index][key]:
                commands.append(
                    external.ExternalProcess(
                        [*obs_group.converter.split(" "), file, out], cwd=obs_group.cwd
                    )
                )
    for res in external.run_in_parallel(commands, jobs):
        if res.returncode != 0:
            logger.error(f"Failed to convert file: {res.command}")
            logger.error(res.output)
            sys.exit(1)

    # Join files per cycle
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = []
        for c in cycles:
            cycle_files_list = [
                out for key in names for _, out in cycle_files[c.index][key]
            ]
            if not cycle_files_list:
                continue

            # Filter only for files that exist because a converter will not create an empty
            # obs_seq if the input file doesn't have any valid data.
            cycle_files_list = [
                f for f in cycle_files_list if f.exists() and f.stat().st_size > 0
            ]

            futures.append(
                executor.submit(
                    observations.join_obs_seq,
                    exp.cfg,
                    cycle_files_list,
                    obs_path / f"cycle_{c.index}.obs_seq",
                    [v.kind for v in obs_groups.values()],
                )
            )

        for future in futures:
            future.result()

    # Remove temporary files
    for c in cycles:
        for key in names:
            for _, out in cycle_files[c.index][key]:
                out.unlink()


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
    exp.set_dart_environment()

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
    logger.info("Joining files...")
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
    exp.set_dart_environment()
    logger.setup("observations-convert-obs-seq", experiment_path)

    observations.obs_seq_to_nc(exp.cfg.directories.dart_root, obs_seq_path, nc_path)


@observations_cli.command()
@click.option(
    "--backup/--no-backup",
    is_flag=True,
    default=True,
    help="Whether to backup the original `obs` directory to `obs.tar.gz`. Existing backup is removed.",
)
@pass_experiment_path
def preprocess_for_wrf(experiment_path: Path, backup: bool):
    """
    Runs all obs_seq files through the WRF preprocessing utility to remove observations
    outside the domain and optionally increase obs. errors near the boundary.
    If and how much is this error increase can be configured inside the `observations` config group.
    """

    logger.setup("observations-preprocess-for-wrf", experiment_path)
    exp = experiment.Experiment(experiment_path)
    exp.set_dart_environment()

    # Find a wrfinput file to get the domain
    if exp.cfg.data.per_member_meteorology:
        wrfinput = exp.paths.data_icbc / "member_00" / "wrfinput_d01_member_00_cycle_0"
    else:
        wrfinput = exp.paths.data_icbc / "wrfinput_d01_cycle_0"
    if not wrfinput.exists():
        logger.error(f"wrfinput file not found at {wrfinput}")
        sys.exit(1)

    # Backup the original obs directory by compressing the directory
    if backup:
        target = exp.paths.data / "obs"
        target.unlink(missing_ok=True)

        logger.info(f"Backing up original `obs` directory to {target}")
        shutil.make_archive(str(target), "gztar", str(exp.paths.obs))

    obs_path = exp.paths.obs
    for cycle in exp.cycles:
        obs_seq = obs_path / f"cycle_{cycle.index}.obs_seq"
        if not obs_seq.exists():
            logger.warning(
                f"Skipping cycle {cycle.index} ({obs_seq}) as it does not exist"
            )
            continue

        assimilation_dt = cycle.end
        base_time = dt.datetime(1601, 1, 1, tzinfo=dt.timezone.utc)

        dart_days = (assimilation_dt - base_time).days
        dart_seconds = int(
            (
                assimilation_dt - assimilation_dt.replace(hour=0, minute=0, second=0)
            ).total_seconds()
        )
        logger.debug(f"DART time: {dart_days}, {dart_seconds}")

        # Date must be format in gregorian day, second format
        assimilation_dt = assimilation_dt.replace(tzinfo=dt.timezone.utc)

        logger.info(f"Preprocessing {obs_seq}")
        observations.preprocess_for_wrf(
            exp.cfg.directories.dart_root,
            wrfinput,
            obs_seq,
            (dart_days, dart_seconds),
            exp.cfg.observations.boundary_width,
            exp.cfg.observations.boundary_error_factor,
            exp.cfg.observations.boundary_error_width,
        )
