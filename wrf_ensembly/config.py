from pathlib import Path
from datetime import datetime
from typing import Any
import os

import tomli
import tomli_w
from pydantic import BaseModel
import rich

from wrf_ensembly.console import console


class MetadataConfig(BaseModel):
    """Metadata about the experiment (name, ...)"""

    name: str
    """Name of the experiment"""

    description: str
    """Description of the experiment"""


class DirectoriesConfig(BaseModel):
    """Info about where the experiment will run (in, out, models,...)"""

    wrf_root: Path
    """Root directory of the WRF model. Should contain the `run` directory w/ real compiled."""

    wps_root: Path
    """Root directory to the WPS. Should contain the `geogrid.exe`, `metgrid.exe` and `ungrib.exe` executables."""

    dart_root: Path
    """Root directory of the DART. Should contain a `models/wrf` directory, compiled."""

    output_sub: Path = Path("./data")
    """Where to store the output analysis files, relative to the experiment root."""

    observations_sub: Path = Path("./obs")
    """Where to store the observations files, relative to the experiment root."""

    work_sub: Path = Path("./work")
    """Work directory for pre-processing and ensemble members, relative to the experiment root."""


class DomainControlConfig(BaseModel):
    xy_resolution: tuple[int, int]
    """Space between two grid points in the x and y directions, kilometers. Corresponds to dx and dy."""

    xy_size: tuple[int, int]
    """Number of grid points in the x and y directions. Corresponds to e_we and e_sn."""

    projection: str = "lambert"
    """Projection used for the grid"""

    ref_lat: float
    """Reference latitude for the projection"""

    ref_lon: float
    """Reference longitude for the projection"""

    truelat1: float
    """True latitude 1 for the projection"""

    truelat2: float
    """True latitude 2 for the projection"""

    stand_lon: float
    """Standard longitude for the projection"""


class TimeControlConfig(BaseModel):
    """Configuration related to the experiment time period."""

    start: datetime
    """Start timestamp of the experiment"""

    end: datetime
    """End timestamp of the experiment"""

    boundary_update_interval: int = 60 * 3
    """Time between incoming real data (lateral boundary conditions) in WRF, minutes"""

    output_interval: int = 60
    """Time between output (history) files in WRF, minutes"""

    analysis_interval: int = 60 * 6
    """Time between analysis/assimilation cycles, minutes"""


class DataConfig(BaseModel):
    """Configuration related to the data used in the experiment."""

    wps_geog: Path
    """Where the WPS_GEOG data is stored, should point to a directory"""

    meteorology: Path
    """Where the meteorological fields GRIB data is stored, should point to a directory"""

    meteorology_glob: str = "*.grib"
    """Glob pattern to use to find the meteorological fields GRIB files"""

    meteorology_vtable: Path = Path("Vtable.ERA-interim.pl")
    """Vtable to use for the meteorological fields GRIB files"""


class AssimilationConfig(BaseModel):
    """Configuration related assimilation"""

    n_members: int
    """Number of ensemble members."""

    cycled_variables: list[str]
    """Which variables to carry forward from the previous cycle"""

    extract_variables: list[str]
    """Which variables to extract from the WRF output into the statistics"""


class PertubationVariableConfig(BaseModel):
    mean: float = 1.0
    """Mean of the pertubation field"""

    sd: float = 1.0
    """Standard deviation of the pertubation field"""

    rounds: int = 10
    """Number of rounds of smoothing to apply to the pertubation field"""


class Config(BaseModel):
    metadata: MetadataConfig
    """Metadata about the experiment (name, ...)"""

    environment: dict[str, str] = {}
    """Environment variables to set when running the experiment"""

    directories: DirectoriesConfig
    """Info about where the experiment will run (in, out, models,...)"""

    domain_control: DomainControlConfig
    """Info about the experiment domain (grid, ...)"""

    time_control: TimeControlConfig
    """Configuration related to the experiment time period."""

    data: DataConfig
    """Configuration related to the data used in the experiment."""

    assimilation: AssimilationConfig
    """Configuration related to assimilation."""

    pertubations: dict[str, PertubationVariableConfig] = {}
    """Configuration related to pertubation of the initial conditions."""

    slurm: dict[str, Any] = {}
    """
    Arguments passed to slurm jobfiles. A special variable "env_modules" can be used
    for loading environment modules at the start of the job.
    """

    wrf_namelist: dict[str, dict[str, Any]]
    """Overrides for the WRF namelist"""


def read_config(path: Path, inject_environment=True) -> Config:
    """
    Reads a TOML configuration file and returns a Config object.

    Args:
        path: Path to the TOML configuration file
        inject_environment: Whether to inject variables from the [environment] group into the environment, defaults to True
    """
    with open(path, "rb") as f:
        cfg = tomli.load(f)

    cfg = Config(**cfg)

    if inject_environment:
        for k, v in cfg.environment.items():
            os.environ[k] = str(v)

    return cfg


def write_config(path: Path, cfg: Config):
    """
    Writes a Config object to a TOML file.

    Args:
        path: Path to the TOML configuration file
        cfg: Config object to write
    """
    with open(path, "wb") as f:
        tomli_w.dump(cfg.dict(), f)


def inspect(cfg: Config):
    rich.inspect(cfg, console=console)
