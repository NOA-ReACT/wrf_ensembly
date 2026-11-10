"""
Initial and lateral boundary conditions for ICON from IFS GRIB files.

The GRIB files come from MARS on a regular lat/lon grid and model levels:
`ifs_<YYYYMMDDHH>.grb` (analysis, with soil and surface fields) for the initial
conditions and `ifs_<YYYYMMDDHH>_<step>_lbc.grb` (forecast steps) for the boundaries.
For each of them we convert to netCDF, rename the variables to what ICON expects and
interpolate horizontally to the ICON grid (first order conservative), all with CDO.
ICON does the vertical interpolation itself when it reads the files.

The interpolation weights only depend on the two grids, so they are computed once
(`cdo gencon`) and applied to every file (`cdo remap`). That is what makes this much
faster than a `cdo remapcon` per file.
"""

import datetime as dt
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from wrf_ensembly.console import logger

# IFS short name -> name ICON expects. With DWD's ecCodes definitions, CDO calls the
# model level geopotential FI and the vertical velocity OMEGA.
IC_RENAME = {
    "FI": "GEOP_ML", "z": "GEOP_SFC", "OMEGA": "W", "lnsp": "LNPS",
    "clwc": "QC", "ciwc": "QI", "crwc": "QR", "cswc": "QS",
    "tsn": "T_SNOW", "sd": "W_SNOW", "rsn": "RHO_SNOW", "asn": "ALB_SNOW",
    "skt": "SKT", "sst": "SST",
    "stl1": "STL1", "stl2": "STL2", "stl3": "STL3", "stl4": "STL4",
    "ci": "CI", "src": "W_I", "sr": "Z0", "lsm": "LSM",
    # swvl1-4 hold the soil moisture index computed by MARS, not soil water
    "swvl1": "SMIL1", "swvl2": "SMIL2", "swvl3": "SMIL3", "swvl4": "SMIL4",
}  # fmt: skip
LBC_RENAME = {
    "OMEGA": "W", "clwc": "QC", "ciwc": "QI", "crwc": "QR", "cswc": "QS",
    "sp": "PS", "sst": "SST", "src": "W_I", "z": "GEOP_SFC",
}  # fmt: skip

IC_PATTERN = re.compile(r"ifs_\d{10}\.grb")
LBC_PATTERN = re.compile(r"ifs_.*_lbc\.grb")

REQUIRED_TOOLS = ("cdo", "grib_get", "codes_info")


class PreprocessError(Exception):
    pass


@dataclass
class Task:
    source: Path
    target: Path
    renames: dict[str, str]
    kind: str  # "IC" or "LBC"


def tool_environment(dwd_definitions: Path | None) -> dict[str, str]:
    """
    Environment for CDO and the ecCodes tools, with DWD's GRIB definitions in front of
    ECMWF's. Raises PreprocessError if a tool or the definitions are missing.
    """

    for tool in REQUIRED_TOOLS:
        if shutil.which(tool) is None:
            raise PreprocessError(
                f"{tool} not found. Load the modules that provide CDO and ecCodes "
                "(slurm.env_modules does this in batch jobs)."
            )

    if dwd_definitions is None and "DWD_DEF" in os.environ:
        dwd_definitions = Path(os.environ["DWD_DEF"])
    if dwd_definitions is None:
        raise PreprocessError(
            "Set icon.dwd_grib_definitions (or the DWD_DEF environment variable) to "
            "DWD's ecCodes definitions, see https://opendata.dwd.de/weather/lib/grib/"
        )
    if not dwd_definitions.is_dir():
        raise PreprocessError(f"{dwd_definitions} is not a directory")

    env = dict(os.environ)
    ecmwf_definitions = subprocess.run(
        ["codes_info", "-d"], capture_output=True, text=True, check=True
    ).stdout.strip()
    env["ECCODES_DEFINITION_PATH"] = f"{dwd_definitions}:{ecmwf_definitions}"
    return env


def _run(cmd: list[str], env: dict[str, str]) -> str:
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        raise PreprocessError(f"{' '.join(cmd)} failed:\n{res.stderr.strip()}")
    return res.stdout


def valid_time(grib: Path, env: dict[str, str]) -> dt.datetime:
    """Valid time of the first message of a GRIB file"""

    out = _run(
        ["grib_get", "-w", "count=1", "-p", "validityDate,validityTime", str(grib)], env
    )
    date, hhmm = out.split()
    return dt.datetime.strptime(f"{date}{int(hhmm):04d}", "%Y%m%d%H%M").replace(
        tzinfo=dt.timezone.utc
    )


def boundary_file_name(pattern: str, time: dt.datetime) -> str:
    """Fills in the date keywords of ICON's `latbc_filename`"""

    for keyword, value in (
        ("<y>", f"{time:%Y}"),
        ("<m>", f"{time:%m}"),
        ("<d>", f"{time:%d}"),
        ("<h>", f"{time:%H}"),
        ("<min>", f"{time:%M}"),
        ("<sec>", f"{time:%S}"),
    ):
        pattern = pattern.replace(keyword, value)
    if "<" in pattern:
        raise PreprocessError(
            f"icon.boundary_filename has a keyword we cannot fill in: {pattern}"
        )
    return pattern


def find_tasks(
    grib_dir: Path,
    initial_conditions: Path,
    boundary_dir: Path,
    boundary_filename: str,
    start: dt.datetime,
    end: dt.datetime,
    env: dict[str, str],
) -> list[Task]:
    """
    What needs converting: the IC file valid at `start` and the LBC files valid
    between `start` and `end`.
    """

    tasks = []
    for path in sorted(grib_dir.iterdir()):
        if IC_PATTERN.fullmatch(path.name):
            if valid_time(path, env) == start:
                tasks.append(Task(path, initial_conditions, IC_RENAME, "IC"))
        elif LBC_PATTERN.fullmatch(path.name):
            time = valid_time(path, env)
            if start <= time <= end:
                target = boundary_dir / boundary_file_name(boundary_filename, time)
                tasks.append(Task(path, target, LBC_RENAME, "LBC"))

    if not any(task.kind == "IC" for task in tasks):
        raise PreprocessError(
            f"No initial conditions (ifs_<YYYYMMDDHH>.grb) valid at {start} in {grib_dir}"
        )
    return tasks


def make_weights(source: Path, grid_file: Path, weights: Path, threads: int, env: dict):
    """Conservative interpolation weights from the grid of `source` to the ICON grid"""

    tmp = weights.with_name(weights.name + ".part")
    # One field is enough, all of them are on the same grid
    _run(
        ["cdo", "-s", "-P", str(threads), f"gencon,{grid_file}:1", "-seltimestep,1",
         "-selcode,129", str(source), str(tmp)],
        env,
    )  # fmt: skip
    tmp.rename(weights)


def convert(
    task: Task, grid_file: Path, weights: Path, threads: int, env: dict
) -> Path:
    """GRIB to netCDF, rename and interpolate one file to the ICON grid"""

    chname = "chname," + ",".join(f"{a},{b}" for a, b in task.renames.items())
    task.target.parent.mkdir(parents=True, exist_ok=True)
    # Under a temporary name first, so that an interrupted run never leaves a file
    # that looks complete
    tmp = task.target.with_name(task.target.name + ".part")
    _run(
        ["cdo", "-s", "-P", str(threads), "-f", "nc4",
         f"remap,{grid_file}:1,{weights}", f"-{chname}", str(task.source), str(tmp)],
        env,
    )  # fmt: skip

    # CDO only warns when a name to rename is missing, e.g. OMEGA when it did not find
    # DWD's definitions (the field is then called w). ICON would fail much later.
    names = set(_run(["cdo", "-s", "showname", str(tmp)], env).split())
    missing = sorted(set(task.renames.values()) - names)
    if missing:
        tmp.unlink()
        raise PreprocessError(
            f"{task.source.name}: no {', '.join(missing)} in the output, are "
            "icon.dwd_grib_definitions DWD's ecCodes definitions?"
        )
    tmp.rename(task.target)
    return task.target


def prepare(
    grib_dir: Path,
    grid_file: Path,
    initial_conditions: Path,
    boundary_dir: Path,
    boundary_filename: str,
    start: dt.datetime,
    end: dt.datetime,
    dwd_definitions: Path | None,
    jobs: int = 4,
    cores: int | None = None,
    overwrite: bool = False,
) -> int:
    """
    Makes the ICON initial and boundary condition files for an experiment from the
    IFS GRIB files in `grib_dir`.

    Args:
        jobs: Number of files converted at the same time.
        cores: Number of cores to share between them. Defaults to all of this job's.
        overwrite: Also convert files whose output exists already.

    Returns:
        The number of files written
    """

    env = tool_environment(dwd_definitions)
    for path in (grib_dir, grid_file):
        if not path.exists():
            raise PreprocessError(f"{path} does not exist")

    tasks = find_tasks(
        grib_dir, initial_conditions, boundary_dir, boundary_filename, start, end, env
    )
    n_all = len(tasks)
    if not overwrite:
        tasks = [task for task in tasks if not task.target.exists()]
    logger.info(
        f"{n_all} IFS file(s) for {start} to {end} in {grib_dir}, {len(tasks)} to convert"
    )
    if not tasks:
        return 0

    if cores is None:
        cores = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    jobs = max(1, min(jobs, len(tasks), cores))
    threads = max(1, cores // jobs)

    boundary_dir.mkdir(parents=True, exist_ok=True)
    weights = boundary_dir / f"ifs_to_{grid_file.stem}_weights.nc"
    if overwrite or not weights.exists():
        logger.info(f"Computing interpolation weights, {weights}")
        make_weights(tasks[0].source, grid_file, weights, cores, env)

    failed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(convert, task, grid_file, weights, threads, env): task
            for task in tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                logger.info(f"{task.kind} {task.source.name} -> {future.result()}")
            except PreprocessError as err:
                failed.append(task)
                logger.error(str(err))

    if failed:
        raise PreprocessError(f"{len(failed)} of {len(tasks)} file(s) failed")
    return len(tasks)
