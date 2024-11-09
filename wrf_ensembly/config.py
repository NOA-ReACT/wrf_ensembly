import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional

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
class EnvironmentConfig:
    """Configuration related to the environment variables"""

    universal: dict[str, str] = field(default_factory=dict)
    """Applied to all commands"""

    wrf: dict[str, str] = field(default_factory=dict)
    """Applied to WRF/WPS commands"""

    dart: dict[str, str] = field(default_factory=dict)
    """Applied to DART commands"""


@dataclass
class DirectoriesConfig:
    """Info about where the experiment will run (in, out, models,...)"""

    wrf_root: Path
    """Root directory of the WRF model. Should contain the `run` directory w/ real compiled."""

    wps_root: Path
    """Root directory to the WPS. Should contain the `geogrid.exe`, `metgrid.exe` and `ungrib.exe` executables."""

    dart_root: Path
    """Root directory of the DART. Should contain a `models/wrf` directory, compiled."""

    scratch_root: Path = Path("./scratch")
    """
    Scratch directory used for temporarily storing model output files before post-processing them.
    If relative, will be inside the experiment directory
    """


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
class ChemistryDataConfig:
    """
    Configuration related to the chemistry global model fields used in the experiment
    These are used with [interpolator_for_wrfchem](https://github.com/NOA-ReACT/interpolator_for_wrfchem), so check that page for more info.
    """

    path: Path
    """
    Where the chemistry fields netCDF data is stored. Should be a directory of YYYY-MM-DD subdirectories, which include netCDF files.
    """

    model_name: str
    """
    Name of the chemistry model used to generate the chemistry fields.
    """


@dataclass
class DataConfig:
    """Configuration related to the data used in the experiment."""

    wps_geog: Path
    """Where the WPS_GEOG data is stored, should point to a directory"""

    meteorology: Path
    """Where the meteorological fields GRIB data is stored, should point to a directory"""

    chemistry: Optional[ChemistryDataConfig] = None
    """Configuration about the chemistry data used in the experiment"""

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
class ObservationsConfig:
    """Configuration related to observation preprocessing (mainly for the `observations preprocess-for-wrf` command)"""

    boundary_width: float = 0
    """By how many grid points to reduce the domain by when removing obs. from outside the domain"""

    boundary_error_factor: float = 2.5
    """If more than 0, inflate the error of observations near the boundary by this factor."""

    boundary_error_width: float = 1.0
    """If more than 0, the error this many grid points near the boundary are inflated by `boundary_error_factor`. Set to 0 to disable."""


@dataclass
class GeogridConfig:
    """Configuration related to geogrid (geographical data preprocessing)."""

    table: Optional[str] = "GEOGRID.TBL"


@dataclass
class PerturbationVariableConfig:
    operation: Literal["add", "multiply"]
    """Whether to add or multiply the perturbation field"""

    mean: float = 1.0
    """Mean of the perturbation field"""

    sd: float = 1.0
    """Standard deviation of the perturbation field"""

    rounds: int = 10
    """Number of rounds of smoothing to apply to the perturbation field"""

    boundary: int = 0
    """Size of the perturbation boundary, in grid points. If > 0, the given amount of rows/columns at the edges will not be pertubated (with a smoothing filter)."""

    def __str__(self) -> str:
        return f"operation={self.operation}, mean={self.mean:.2f}, sd={self.sd:.2f}, rounds={self.rounds}, boundary={self.boundary}"


@dataclass
class PerturbationsConfig:
    """Configuration about perturbation fields"""

    variables: dict[str, PerturbationVariableConfig] = field(default_factory=dict)
    """Configuration for each variable"""

    seed: Optional[int] = None
    """RNG seed to use when generating perturbation fields. If none, it will be randomly generated."""


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

    pre_commands: list[str] = field(default_factory=list)
    """Commands to run at the start of a job"""

    directives_large: dict[str, str | int] = field(default_factory=dict)
    """SLURM directives to add to the jobfile for big jobs (i.e., ensemble member advance)"""

    directives_small: dict[str, str | int] = field(default_factory=dict)
    """SLURM directives to add to small jobs (i.e., wrf-ensembly python steps)"""

    directives_postprocess: dict[str, str | int] = field(default_factory=dict)
    """SLURM directives to add to statistics jobs"""


@dataclass
class PostprocessConfig:
    variables_to_keep: Optional[list[str]] = None
    """
    Optionally, filter the variables in a file by a list of regular expressions. If None, all variables are kept.
    This filtering is applied during the `postprocess wrf-post` step.
    """

    compression_filters: str = "shf|zst,3"
    """
    Which compression filter to apply when producing the final cycle files
    (during `postprocess concatenate`). Consult the NCO manual for exact options available.
    This refers to lossless compression and is always a good idea but the default ZST
    algorithm might not be available on your system. Set to empty string to disable compression.
    """

    ppc_filter: str = "default=3#Z.*=6#X.*=6"
    """
    Controls lossy quantization, which is applied during `postprocess concatenate`. This
    affects the precision of the output files. Consult the NCO manual for exact options available. The default value applies the granular BR algorithm with 3 significant digits
    to all variables, except for those starting with Z or X, which get 6 significant digits.
    A small investigation has yielded that these values are a good compromise between file size and precision, at least for dust and wind related fields. Set to empty string to disable quantization.
    """

    scripts: list[str] = field(default_factory=list)
    # """
    # List to postprocessing scripts to run on each output analysis and forecast file.
    # The string can be any command. The script should write to the given path.
    # The following placeholders will be replaced:
    # - {in}: path to the input file (analysis or forecast, netCDF4)
    # - {out}: path to the output file (analysis or forecast, netCDF4)
    # - {d} member number
    # - {c} cycle number
    # """

    wrf_post_cores: int = 1
    """How many cores to use for the `wrf-post` step"""

    apply_scripts_cores: int = 1
    """How many cores to use for the `apply-scripts` step"""

    statistics_cores: int = 1
    """How many cores to use for the `statistics` step"""

    concatenate_cores: int = 1
    """How many cores to use for the `concatenate` step"""

    def __post_init__(self):
        """Validates the `script` field, making sure that all commands at least take the `{in}` and `{out}` placeholders."""
        for script in self.scripts:
            if "{in}" not in script or "{out}" not in script:
                raise ValueError(
                    f"Postprocessing script '{script}' does not contain the required placeholders {{in}} and {{out}}"
                )


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

    observations: ObservationsConfig
    """Configuration related to observations."""

    geogrid: GeogridConfig
    """Configuration related to geogrid (geographical data preprocessing)."""

    perturbations: PerturbationsConfig
    """Configuration related to perturbation of the initial conditions."""

    slurm: SlurmConfig
    """Configuration related to SLURM jobfiles."""

    postprocess: PostprocessConfig
    """Configuration related to wrfout postprocessing"""

    wrf_namelist: dict[str, dict[str, Any]]
    """Overrides for the WRF namelist"""

    wrf_namelist_per_member: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )
    """Overrides for the WRF namelist per ensemble member"""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
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
        for k, v in cfg.environment.universal.items():
            os.environ[k] = str(v)

    return cfg


def inspect(cfg: Config):
    rich.inspect(cfg, console=console)
