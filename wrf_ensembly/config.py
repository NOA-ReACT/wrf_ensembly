import os
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, override

import rich
import tomli
from mashumaro.config import BaseConfig
from mashumaro.mixins.toml import DataClassTOMLMixin
from mashumaro.types import SerializationStrategy

from wrf_ensembly.console import console


class UTCDatetimeStrategy(SerializationStrategy):
    """
    Ensure datetimes are always serialized/deserialized with timezone info
    If the datetime has no timezone info, assume UTC.
    """

    @override
    def serialize(self, value: datetime) -> str:
        return value.isoformat()

    @override
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

    truelat2: float | None = None
    """True latitude 2 for the projection"""

    stand_lon: float | None = None
    """Standard longitude for the projection"""

    pole_lat: float | None = None
    """Pole latitude for the projection"""

    pole_lon: float | None = None
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

    duration: int | None = None
    """Duration of the cycle in minutes"""

    output_interval: int | None = None
    """Override the output interval for this cycle"""

    forecast_extension: int | None = None
    """Override the forecast extension (minutes past the cycle end) for this cycle"""


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

    forecast_extension: int = 0
    """
    Number of minutes to integrate each cycle's forward (member) run past its end.
    The extra timesteps fall inside the next cycle's window but are independent of the
    analysis produced at the cycle end, providing longer-lead forecasts for verification.
    Does not affect the assimilation boundary: the analysis, DART prior and the next
    cycle's start are still keyed off the (unextended) cycle end. Defaults to 0 (disabled).
    The final cycle is clamped to the experiment end, since no boundary data exists beyond it.
    """

    cycles: dict[int, CycleConfig] = field(default_factory=dict)
    """Configuration overrides for specific cycles"""

    runtime_io: list[str] | None = field(default_factory=list)
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

        The `runtime_io`, `cycles` and `forecast_extension` fields are not included in
        the comparison. `forecast_extension` only changes forward-run length and the
        forecasts produced, never the assimilation state, so it is safe to change on restart.
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

    chemistry: ChemistryDataConfig | None = None
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

    use_inflation: bool = False
    """
    Set to true if using DART inflation to manage the inflation files (moving output files to input for next cycle) and correctly set `inf_initial_from_restart`/`inf_sd_initial_from_restart` in DART's namelist.
    You still need to set the inflation settings correctly in the DART namelist.
    """


@dataclass
class TemporalBinConfig:
    """Configuration for temporal binning of one instrument/quantity pair."""

    bin_minutes: int
    """Width of each time window in minutes."""

    offset_minutes: int = 0
    """
    Shift bin boundaries by this many minutes relative to UTC midnight alignment.
    For example, with bin_minutes=60 and offset_minutes=-30, bins run from :30 to :30,
    centering each window on a full hour — useful when comparing against hourly model output.
    """

    reduce_instrument_error: bool = True
    """
    Whether averaging n observations into a bin reduces the instrument error by sqrt(n).
    Set to false when the errors within a bin are correlated (e.g. smoothed retrievals),
    in which case the binned instrument error stays at rms(individual errors).
    """


@dataclass
class SuperObsConfig:
    """Configuration related to how we generate superobservations (or superobs) for one instrument/quantity pair"""

    hoz_bin_sizes: dict[str, int]
    """
    A dictionary where keys are dimension names (i.e. along_track) and values are how long is each bin.
    All valid-QC observations inside the bin are combined into one superob.
    """

    vert_bin_sizes: dict[str, int]
    """
    A dictionary where keys are vertical dimension names (e.g. height_bin) and values are how long is each bin.
    All valid-QC observations inside the bin are combined into one superob.
    """


@dataclass
class ThinningConfig:
    """Configuration for stride thinning of one instrument/quantity pair."""

    keep_every_n: int
    """Keep every N-th good-QC observation for DA; the rest become validation hold-outs (qc_flag = -1)."""


@dataclass
class ObservationsConfig:
    """Configuration related to observation preprocessing (mainly for the `observations preprocess-for-wrf` command)"""

    instruments_to_assimilate: list[str] | None = None
    """Which instruments to assimilate. If None, all available instruments are used."""

    error_inflation_factor: dict[str, float] = field(default_factory=dict)
    """
    Error inflation factor per instrument and quantity.
    The key is the `instrument.quantity` string, e.g. `sonde.U`.
    The value is the factor by which to multiply the observation error.
    The factor is applied when determining which observations are used in each cycle
    (`observations prepare-cycles` command).
    """

    boundary_width: float = 0
    """By how many grid points to reduce the domain by when removing obs. from outside the domain"""

    boundary_error_factor: float = 2.5
    """If more than 0, inflate the error of observations near the boundary by this factor."""

    boundary_error_width: float = 1.0
    """If more than 0, the error this many grid points near the boundary are inflated by `boundary_error_factor`. Set to 0 to disable."""

    superobs: dict[str, SuperObsConfig] = field(default_factory=dict)
    """
    Superobservation configuration per instrument and quantity.
    The key is the `instrument.quantity` string, e.g. `sonde.U`.
    The value is a `SuperObsConfig` instance.
    """

    thinning: dict[str, ThinningConfig] = field(default_factory=dict)
    """
    Stride thinning configuration per instrument and quantity.
    The key is the `instrument.quantity` string, e.g. `AEOLUS_L2B_MIE.HLOS_WIND`.
    The value is a `ThinningConfig` instance.
    Thinning is applied after superobbing. Good-QC observations not selected for DA
    are marked with qc_flag = -1 (validation hold-out) and remain in the database.
    """

    temporal_binning: dict[str, TemporalBinConfig] = field(default_factory=dict)
    """
    Temporal binning configuration per instrument and quantity.
    The key is the `instrument.quantity` string, e.g. `AERONET.AOD_550nm`.
    The value is a TemporalBinConfig instance.
    Incompatible with superobs (spatial grid binning) for the same instrument-quantity pair.
    """


@dataclass
class FirstDeparturesRegimeConfig:
    """Configuration for regime-based first departures analysis."""

    instrument: str
    """The instrument name (e.g., 'MODIS', 'VIIRS', 'AERONET')."""

    quantity: str
    """The observation quantity to analyze (e.g., 'AOD_550nm', 'PM2_5_DRY')."""

    bins: list[float]
    """Bin edges for regime classification. Use float('inf') for unbounded upper limit."""

    labels: list[str]
    """Labels for each regime (length should be len(bins) - 1)."""

    spatial_resolution: float = 1.0
    """Resolution in degrees for spatial binning (default: 1.0)."""


@dataclass
class FirstDeparturesConfig:
    """Configuration for first departures analysis."""

    instrument_quantity_pairs: list[str] = field(default_factory=list)
    """List of instrument.quantity pairs to analyze (e.g., ['MODIS.AOD_550nm', 'VIIRS.AOD_550nm']). If empty, will analyze all available pairs."""

    bias_map_colorbar_ranges: dict[str, tuple[float, float]] = field(
        default_factory=dict
    )
    """Colorbar ranges for the bias maps, the keys should be the instrument.quantity pairs with the value being a 2 float tuple. Use this to override the auto colorbar if a bad region is ruining your map."""

    excluded_bboxes: list[tuple[float, float, float, float]] = field(
        default_factory=list
    )
    """List of bounding boxes to exclude from the analysis (list of 4 floats: min_lat, min_lon, max_lat, max_lon). Contradicts the included_bboxes setting."""

    analysis_bbox: tuple[float, float, float, float] | None = None
    """If defined, only do analysis inside this box. Applied before `excluded_bboxes`."""

    regimes: list[FirstDeparturesRegimeConfig] = field(default_factory=list)
    """Regime configurations for different instrument-quantity pairs."""


@dataclass
class ValidationConfig:
    """Configuration related to validation of the experiment"""

    instruments: list[str] = field(default_factory=lambda: [])
    """List of instruments to use for validation, if missing it will use all available instruments."""

    prefer_extended_forecast: bool = True
    """
    When forecasts overlap in time (because `time_control.forecast_extension` is set),
    decide which forecast frame to use at each timestamp during interpolation.

    If true (default), prefer the longer-lead, independent forecast from the earlier cycle
    (i.e. the extended forecast that has not seen the analysis at the cycle end) — the
    correct background for O-B verification. If false, prefer the shorter-lead,
    analysis-driven forecast from the later cycle (the legacy behaviour).

    Applied experiment-wide so every row in the validation output shares one semantics.
    A no-op when `forecast_extension` is 0 (no overlap exists).
    """

    first_departures: FirstDeparturesConfig = field(
        default_factory=FirstDeparturesConfig
    )
    """Configuration for first departures (O-B) analysis."""


@dataclass
class GeogridConfig:
    """Configuration related to geogrid (geographical data preprocessing)."""

    table: str | None = "GEOGRID.TBL"


@dataclass
class PerturbationVariableConfig:
    """Configuration about how to perturb a specific variable"""

    operation: Literal["add", "multiply", "assign"]
    """Whether to add or multiply the perturbation field to the variable, or to directly assign it (overwritting the variable)"""

    perturb_every_cycle: bool = False
    """Whether to generate a perturbation field for this variable at every cycle. If false, it will be perturbed only at the first cycle."""

    different_field_every_cycle: bool = True
    """If `perturb_every_cycle` is true, whether to generate a different perturbation field at every cycle, or to use the same field."""

    midcycle_taper_width: int = 0
    """
    Width of the tapering region applied to the perturbation field during midcycles.
    This only applies if `perturb_every_cycle` is true. If > 0, an area of size
    `midcycle_taper_width` grid points at the edges of the domain will be perturbed with
    weight equal to 1, and the perturbation will be tapered to 0 in the next
    `midcycle_taper_width` grid points. This is useful to avoid discontinuities at
    the boundaries when perturbing every cycle. Doesn't work if `operation` is `assign`.
    """

    mean: float = 1.0
    """Mean of the perturbation field"""

    sd: float = 1.0
    """Standard deviation of the perturbation field"""

    gaussian_sigma: float = 2.5
    """Standard deviation for the Gaussian filter applied to the perturbation field"""

    boundary: int = 0
    """Size of the perturbation boundary, in grid points. If > 0, the given amount of rows/columns at the edges will not be pertubated (with a smoothing filter)."""

    min_value: float | None = None
    """Minimum value of the perturbation field. If None, no minimum is applied."""

    max_value: float | None = None
    """Maximum value of the perturbation field. If None, no maximum is applied."""

    @override
    def __str__(self) -> str:
        return f"operation={self.operation}, mean={self.mean:.2f}, sd={self.sd:.2f}, gaussian_sigma={self.gaussian_sigma}, boundary={self.boundary}"


@dataclass
class PerturbationsConfig:
    """Configuration about perturbation fields"""

    variables: dict[str, PerturbationVariableConfig] = field(default_factory=dict)
    """Configuration for each variable"""

    seed: int | None = None
    """RNG seed to use when generating perturbation fields. If none, it will be randomly generated."""


@dataclass
class SlurmDirectivesConfig:
    default: dict[str, str | int] = field(default_factory=dict)
    """SLURM directives applied to all jobs as a baseline"""

    advance_model: dict[str, str | int] = field(default_factory=dict)
    """SLURM directives for ensemble member advance jobs (overrides default)"""

    preprocess: dict[str, str | int] = field(default_factory=dict)
    """SLURM directives for preprocessing jobs (WPS, real) (overrides default)"""

    make_analysis: dict[str, str | int] = field(default_factory=dict)
    """SLURM directives for filter/analysis/cycle jobs (overrides default)"""

    postprocess: dict[str, str | int] = field(default_factory=dict)
    """SLURM directives for postprocessing jobs (overrides default)"""


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

    directives: SlurmDirectivesConfig = field(default_factory=SlurmDirectivesConfig)
    """Per-job-type SLURM directives; job-specific ones override the default"""


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
    variables_to_keep: list[str] | None = None
    """
    Optionally, filter the variables in a file by a list of regular expressions. If None, all variables are kept.
    This filtering is applied during the `postprocess process-pipeline` step.
    """

    variables_to_keep_ensemble: list[str] | None = None
    """
    Optionally, filter the variables stored in the per-member ensemble file by a list of regular
    expressions. If None, all variables (after applying `variables_to_keep`) are kept. Use this
    to limit the size of the ensemble file, which can become large with many members.
    Only used when `keep_per_member = true`.
    """

    compression: str = "zlib"
    """
    Compression algorithm to use: 'zlib', 'zstd', 'bzip2', 'szip', or 'none'.
    Note: 'zstd' and 'bzip2' require HDF5 plugins to be installed.
    Use 'none' to disable compression entirely.
    """

    compression_level: int = 4
    """
    Compression level (0-9). Higher values give better compression but are slower.
    A value of 0 disables compression even if a compression algorithm is specified.
    """

    shuffle: bool = True
    """
    Apply the shuffle filter before compression. This reorders bytes to improve
    compression ratios, especially for floating-point data. Recommended to keep enabled.
    """

    significant_digits: int = 3
    """
    Number of significant digits to preserve during quantization (lossy compression).
    Set to zero to disable quantization entirely. Requires netcdf-c >= 4.9.0.
    """

    significant_digits_overrides: dict[str, int] = field(
        default_factory=lambda: {"Z.*": 6, "X.*": 6}
    )
    """
    Per-variable overrides for significant_digits. Keys are regex patterns that match
    variable names, values are the number of significant digits to use.
    Variables matching a pattern use that value instead of the default.
    Example: {"Z.*": 6, "X.*": 6} gives 6 digits to variables starting with Z or X.
    """

    quantize_mode: str = "GranularBitRound"
    """
    Quantization algorithm: 'BitGroom', 'BitRound', or 'GranularBitRound'.
    'GranularBitRound' typically provides best compression for geophysical data.
    Only used when significant_digits is not None.
    """

    processors: list[ProcessorConfig] = field(default_factory=lambda: [])
    """
    List of data processors to apply to each output analysis and forecast file.
    Each processor is specified as a dictionary with a 'processor' key indicating
    the processor type, and additional keys for processor-specific configuration.

    By default, the built-in XWRFProcessor will always be used as the first step.

    External processors can be specified as "module.path:ClassName" or as a path to a Python file.

    Examples:
    processors = [
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


@dataclass
class PlotVariableConfig:
    """Configuration for a single variable to plot"""

    name: str
    """Variable name in the netCDF file"""

    level: int | None = None
    """Vertical level index to select. None means the variable is 2D."""

    pressure_level: float | None = None
    """Pressure level in hPa to interpolate to. Requires an `air_pressure` variable in the dataset.
    If set, takes precedence over `level`."""

    extent: tuple[float, float, float, float] | None = None
    """Geographical extent of the plot (in degrees, min_lon, max_lon, min_lat, max_lat)"""

    vmin: float | None = None
    """Colorbar minimum for forecast/analysis panels. None means auto."""

    vmax: float | None = None
    """Colorbar maximum for forecast/analysis panels. None means auto."""

    diff_vmin: float | None = None
    """Colorbar minimum for the difference panel. None means auto."""

    diff_vmax: float | None = None
    """Colorbar maximum for the difference panel. None means auto."""

    spread_vmin: float | None = None
    """Colorbar minimum for forecast/analysis spread panels. None means auto."""

    spread_vmax: float | None = None
    """Colorbar maximum for forecast/analysis spread panels. None means auto."""

    cmap: str = "viridis"
    """Colormap for forecast/analysis panels"""

    diff_cmap: str = "RdBu_r"
    """Colormap for the difference panel"""

    spread_cmap: str = "viridis"
    """Colormap for spread panels"""


@dataclass
class ForecastVsAnalysisPlotsConfig:
    """Configuration for forecast vs analysis comparison plots"""

    variables: list[PlotVariableConfig] = field(default_factory=list)
    """List of variables to plot"""

    arrangement: str = "horizontal"
    """Panel arrangement: 'horizontal' (1x3) or 'vertical' (3x1)"""

    dpi: int = 150
    """DPI for saved plot images"""

    include_spread: bool = False
    """Whether to also generate spread comparison plots"""


@dataclass
class ForecastPlotsConfig:
    """Configuration for forecast-only plots"""

    variables: list[PlotVariableConfig] = field(default_factory=list)
    """List of variables to plot"""

    dpi: int = 150
    """DPI for saved plot images"""

    include_spread: bool = False
    """Whether to also generate spread plots"""


@dataclass
class PlotsConfig:
    """Configuration for diagnostic plots"""

    forecast_vs_analysis: ForecastVsAnalysisPlotsConfig = field(
        default_factory=ForecastVsAnalysisPlotsConfig
    )
    """Configuration for forecast vs analysis comparison plots"""

    forecasts: ForecastPlotsConfig = field(default_factory=ForecastPlotsConfig)
    """Configuration for forecast-only plots"""


@dataclass
class CopyFileConfig:
    """
    Configuration about a file to copy into the DART directory before running assimilation
    """

    source: Path
    """Path to the source file to copy"""

    destination_name: str | None = None
    """Path to the destination file inside the DART directory. If None, will use the same name as the source file."""


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

    validation: ValidationConfig
    """Configuration related to validation of the experiment"""

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

    dart_namelist: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Overrides for the DART namelist (input.nml)"""

    extra_dart_files: list[CopyFileConfig] = field(default_factory=list)
    """List of extra files to copy into the DART directory before running assimilation"""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    """Environment variables to set when running the experiment"""

    plots: PlotsConfig = field(default_factory=PlotsConfig)
    """Configuration for diagnostic plots"""


def _convert_datetimes_to_iso(obj: Any) -> Any:
    """
    Recursively convert datetime objects to ISO format strings.
    This is needed because tomli.loads() returns datetime objects, but mashumaro's
    from_dict() expects strings that it can deserialize using the SerializationStrategy.
    """

    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: _convert_datetimes_to_iso(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_datetimes_to_iso(item) for item in obj]
    else:
        return obj


def _deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries, with values from override taking precedence.

    For nested dictionaries, merges them recursively.
    For other types (lists, primitives), override completely replaces base.

    Args:
        base: Base dictionary
        override: Dictionary with overriding values

    Returns:
        Merged dictionary
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = _deep_merge_dicts(result[key], value)
        else:
            # For everything else (lists, primitives, or new keys), override wins
            result[key] = value

    return result


def read_config(path: Path, inject_environment=True) -> Config:
    """
    Reads a TOML configuration file and returns a Config object.

    If an `env_config.toml` file exists alongside the main config file, it will be
    loaded and merged with the main config. The env_config values take precedence,
    allowing you to override environment-specific settings (like SLURM directives,
    paths, etc.) while keeping the main config focused on experiment settings.

    This is useful for managing multiple HPC environments using symlinks:
    ```
    config.toml          # Main experiment config
    env_config.toml -> env_aris.toml      # Symlink to environment-specific config
    env_aris.toml        # ARIS cluster settings
    env_iridium.toml     # Iridium cluster settings
    ```

    Args:
        path: Path to the TOML configuration file
        inject_environment: Whether to inject variables from the [environment] group into the environment, defaults to True
    """
    # Read base config
    base_text = path.read_text()

    # Check for env_config.toml in the same directory
    env_config_path = path.parent / "env_config.toml"
    if env_config_path.exists():
        env_text = env_config_path.read_text()
        # Parse both as raw dicts (tomli handles datetime parsing)
        base_dict = tomli.loads(base_text)
        env_dict = tomli.loads(env_text)
        # Merge with env_config taking precedence
        merged_dict = _deep_merge_dicts(base_dict, env_dict)
        # Convert datetime objects to ISO strings for mashumaro deserialization
        merged_dict = _convert_datetimes_to_iso(merged_dict)
        # Deserialize the merged dict
        cfg = Config.from_dict(merged_dict)
    else:
        cfg = Config.from_toml(base_text)

    if inject_environment:
        for k, v in cfg.environment.universal.items():
            os.environ[k] = str(v)

    return cfg


def inspect(cfg: Config):
    rich.inspect(cfg, console=console)
