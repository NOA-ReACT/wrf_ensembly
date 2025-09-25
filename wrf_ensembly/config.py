import os
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import rich
from mashumaro.mixins.toml import DataClassTOMLMixin
from mashumaro.config import BaseConfig
from mashumaro.types import SerializationStrategy

from wrf_ensembly.console import console


class UTCDatetimeStrategy(SerializationStrategy):
    """
    Ensure datetimes are always serialized/deserialized with timezone info
    If the datetime has no timezone info, assume UTC.
    """

    def serialize(self, value: datetime) -> str:
        return value.isoformat()

    def deserialize(self, value: str) -> datetime:
        dt = datetime.fromisoformat(value)
        # If no timezone info, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt


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

    def is_equal(self, other) -> bool | str:
        """
        Compares with another instance of DomainControlConfig

        If they are exactly the same, returns True. If any field is different, returns
        the field name. If `other` is of different type, returns False.
        """

        if type(self) is not type(other):
            return False

        for f in fields(self):
            if getattr(self, f.name) != getattr(other, f.name):
                return f.name
        return True


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

    runtime_io: Optional[list[str]] = field(default_factory=list)
    """
    Optionally, add runtime I/O options to WRF. If set, it will create a text file in
    each member directory and set it's name in the `iofields_filename` namelist variable.
    One line per list item.
    More info: https://github.com/wrf-model/WRF/blob/master/doc/README.io_config
    """

    def is_equal(self, other) -> bool | str:
        """
        Compares with another instance of TimeControlConfig

        If they are exactly the same, returns True. If any field is different, returns
        the field name. If `other` is of different type, returns False.

        The `runtime_io` and `cycles` fields is not included in the comparison.
        """

        ignored_fields = ["runtime_io", "cycles"]

        if type(self) is not type(other):
            return False

        for f in fields(self):
            if f in ignored_fields:
                continue
            if getattr(self, f.name) != getattr(other, f.name):
                return f.name
        return True


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
    """
    Where the meteorological fields GRIB data is stored, should point to a directory
    See also `per_member_meteorology`.
    """

    per_member_meteorology: bool = False
    """
    Whether to have a separate meteorology directory for each ensemble member. If true, then
    the `meteorology` field should contain the %MEMBER% placeholder for the member number.
    For example, `/path/to/data/meteorology/member_%MEMBER%`.
    """

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

    half_window_length_minutes: int = 30
    """Half-length of the window in which observations will be considered, in minutes. For example, if set to 30, then observations from 30 minutes before and after the cycle end time will be used. Default is 30 minutes."""


@dataclass
class SuperorbingConfig:
    """Configuration of how to superorb a specific instrument's observations"""

    spatial_radius_x_meters: float
    """Spatial radius in the x direction, meters"""

    spatial_radius_y_meters: float
    """Spatial radius in the y direction, meters"""

    spatial_radius_z: Optional[float] = None
    """
    Spatial radius in the z direction, in whatever units the observation's z_type is in
    (e.g., meters for height, hPa for pressure).

    If missing, no superobing in the z direction is done.
    """

    temporal_radius_seconds: int = 60
    """Temporal radius in seconds"""


@dataclass
class ObservationsConfig:
    """Configuration related to observation preprocessing (mainly for the `observations preprocess-for-wrf` command)"""

    instruments_to_assimilate: Optional[list[str]] = None
    """Which instruments to assimilate. If None, all available instruments are used."""

    superorbing: Dict[str, SuperorbingConfig] = field(default_factory=dict)
    """Configuration of how to superorb specific instruments. Key is the instrument name."""

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

    min_value: Optional[float] = None
    """Minimum value of the perturbation field. If None, no minimum is applied."""

    max_value: Optional[float] = None
    """Maximum value of the perturbation field. If None, no maximum is applied."""

    def __str__(self) -> str:
        return f"operation={self.operation}, mean={self.mean:.2f}, sd={self.sd:.2f}, rounds={self.rounds}, boundary={self.boundary}"


@dataclass
class PerturbationsConfig:
    """Configuration about perturbation fields"""

    variables: dict[str, PerturbationVariableConfig] = field(default_factory=dict)
    """Configuration for each variable"""

    seed: Optional[int] = None
    """RNG seed to use when generating perturbation fields. If none, it will be randomly generated."""

    apply_perturbations_every_cycle: bool = False
    """Whether to apply the perturbations at the start of every cycle when using `slurm run-experiment`"""


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
class ProcessorConfig:
    """
    Configuration for a data processor, which are used to apply transformations to the wrfout files
    """

    processor: str
    """Name of the processor to use. Can be a built-in processor or an external one."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters for the processor. Depends on the processor type."""


@dataclass
class PostprocessConfig:
    variables_to_keep: Optional[list[str]] = None
    """
    Optionally, filter the variables in a file by a list of regular expressions. If None, all variables are kept.
    This filtering is applied during the `postprocess process-pipeline` step.
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

    processors: list[ProcessorConfig] = field(default_factory=lambda: [])
    """
    List of data processors to apply to each output analysis and forecast file.
    Each processor is specified as a dictionary with a 'processor' key indicating
    the processor type, and additional keys for processor-specific configuration.

    By default, the built-in XWRFProcessor will always be used as the first step.

    Other built-in processors:
    - "script": Run external scripts (for backward compatibility with 0.9.0 where we had the `apply-scripts` command).

    External processors can be specified as "module.path:ClassName" or as a path to a Python file.

    Examples:
    processors = [
        {"processor" = "script", "parameters" = {"script" = "python enhance_data.py {in} {out}"}},
        {"processor" = "my_package.processors:CustomProcessor", "parameters" = {"param" =  "value"}}
        {"processor" = "/path/to/file.py:MyProcessor", "parameters" = {"param2": "value2"}},
    ]
    """

    compute_ensemble_statistics_in_job: bool = True
    """
    Set this to false to disable the computation of mean/spread for each cycle when
    using slurm jobs.

    Useful when running 1-member experiments or sensitivity studies with different parameters
    per member.
    """

    keep_per_member: bool = False
    """
    Set to true to also concatenate per_member files when running the `concatenate` command.
    If enabled, you will get a `forecast_mean`, `forecast_sd` and `forecast_member_{d}` file for each cycle.
    """

    processor_cores: int = 1
    """How many cores to use for the processor pipeline step"""

    statistics_cores: int = 1
    """How many cores to use for the `statistics` step"""

    concatenate_cores: int = 1
    """How many cores to use for the `concatenate` step"""

    cdo_path = "cdo"
    """
    Path to the CDO executable or command needed to run cdo (e.g. micromamba run cdo).
    If not set, will use the one in the PATH environment variable.
    """

    ncrcat_cmd = "ncrcat"
    """
    Path to the NCO executable or command needed to run ncrcat (e.g. micromamba run ncrcat).
    If not set, will use the one in the PATH environment variable.
    """


@dataclass
class Config(DataClassTOMLMixin):
    class Config(BaseConfig):
        serialization_strategy = {datetime: UTCDatetimeStrategy()}

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
