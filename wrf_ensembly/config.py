import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import rich
from mashumaro.mixins.toml import DataClassTOMLMixin

from wrf_ensembly.console import console


@dataclass
class MetadataConfig:
    """Metadata about the experiment (name, ...)"""

    name: str
    """Name of the experiment"""

    description: str
    """Description of the experiment"""


@dataclass
class DirectoriesConfig:
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


@dataclass
class DomainControlConfig:
    xy_resolution: tuple[int, int]
    """Space between two grid points in the x and y directions, kilometers. Corresponds to dx and dy."""

    xy_size: tuple[int, int]
    """Number of grid points in the x and y directions. Corresponds to e_we and e_sn."""

    projection: str
    """Projection used for the grid"""

    ref_lat: float
    """Reference latitude for the projection"""

    ref_lon: float
    """Reference longitude for the projection"""

    truelat1: float
    """True latitude 1 for the projection"""

    truelat2: Optional[float] = None
    """True latitude 2 for the projection"""

    stand_lon: Optional[float] = None
    """Standard longitude for the projection"""

    pole_lat: Optional[float] = None
    """Pole latitude for the projection"""

    pole_lon: Optional[float] = None
    """Pole longitude for the projection"""


@dataclass
class CycleConfig:
    """Configuration overrides for a specific cycle"""

    duration: Optional[int] = None
    """Duration of the cycle in minutes"""

    output_interval: Optional[int] = None
    """Override the output interval for this cycle"""


@dataclass
class TimeControlConfig:
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

    cycles: Dict[int, CycleConfig] = field(default_factory=dict)
    """Configuration overrides for specific cycles"""


@dataclass
class DataConfig:
    """Configuration related to the data used in the experiment."""

    wps_geog: Path
    """Where the WPS_GEOG data is stored, should point to a directory"""

    meteorology: Path
    """Where the meteorological fields GRIB data is stored, should point to a directory"""

    meteorology_glob: str = "*.grib"
    """Glob pattern to use to find the meteorological fields GRIB files"""

    meteorology_vtable: Path = Path("Vtable.ERA-interim.pl")
    """Vtable to use for the meteorological fields GRIB files"""

    manage_chem_ic: bool = False
    """If true, use the chemical initial conditions. In practice, this makes sure that the `chem_in_opt` namelist variable is set to 0 when running `real.exe` and to 1 when running `wrf.exe`"""


@dataclass
class AssimilationConfig:
    """Configuration related assimilation"""

    n_members: int
    """Number of ensemble members."""

    cycled_variables: list[str]
    """Which variables to carry forward from the previous cycle"""

    state_variables: list[str]
    """Which variables to use in the state vector"""

    filter_mpi_tasks: int = 1
    """If != 1, then filter will be executed w/ MPI and this many tasks (mpirun -n <filter_mpi_tasks>). Also check `slurm.mpirun_command`."""


@dataclass
class GeogridConfig:
    """Configuration related to geogrid (geographical data preprocessing)."""

    table: Optional[str] = "GEOGRID.TBL"


@dataclass
class PertubationVariableConfig:
    mean: float = 1.0
    """Mean of the pertubation field"""

    sd: float = 1.0
    """Standard deviation of the pertubation field"""

    rounds: int = 10
    """Number of rounds of smoothing to apply to the pertubation field"""

    def __str__(self) -> str:
        return f"mean={self.mean:.2f}, sd={self.sd:.2f}, rounds={self.rounds}"


@dataclass
class PertubationsConfig:
    """Configuration about pertubation fields"""

    variables: dict[str, PertubationVariableConfig] = field(default_factory=dict)
    """Configuration for each variable"""

    seed: Optional[int] = None
    """RNG seed to use when generating pertubation fields. If none, it will be randomly generated."""


@dataclass
class SlurmConfig:
    sbatch_command: str = "sbatch --parsable"
    """Command for sbatch (should probably include `--parsable`)"""

    command_prefix: str = ""  # e.g., "conda run -n wrf-ensembly"
    """Used to prefix all calls to `wrf-ensembly`, useful for using `conda run` or similar"""

    mpirun_command: str = "mpirun"
    """Command to run an MPI binary (might be srun in some clusters)"""

    env_modules: list[str] = field(default_factory=list)
    """List of environment modules to load in each job"""

    directives_large: dict[str, str | int] = field(default_factory=dict)
    """SLURM directives to add to the jobfile for big jobs (i.e., ensemble member advance)"""

    directives_small: dict[str, str | int] = field(default_factory=dict)
    """SLURM directives to add to small jobs (i.e., wrf-ensembly python steps)"""

    directives_statistics: dict[str, str | int] = field(default_factory=dict)
    """SLURM directives to add to statistics jobs"""


@dataclass
class Config(DataClassTOMLMixin):
    metadata: MetadataConfig
    """Metadata about the experiment (name, ...)"""

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

    geogrid: GeogridConfig
    """Configuration related to geogrid (geographical data preprocessing)."""

    pertubations: PertubationsConfig
    """Configuration related to pertubation of the initial conditions."""

    slurm: SlurmConfig
    """Configuration related to SLURM jobfiles."""

    wrf_namelist: dict[str, dict[str, Any]]
    """Overrides for the WRF namelist"""

    environment: dict[str, str] = field(default_factory=dict)
    """Environment variables to set when running the experiment"""


def read_config(path: Path, inject_environment=True) -> Config:
    """
    Reads a TOML configuration file and returns a Config object.

    Args:
        path: Path to the TOML configuration file
        inject_environment: Whether to inject variables from the [environment] group into the environment, defaults to True
    """

    cfg = Config.from_toml(path.read_text())

    if inject_environment:
        for k, v in cfg.environment.items():
            os.environ[k] = str(v)

    return cfg


def inspect(cfg: Config):
    rich.inspect(cfg, console=console)
