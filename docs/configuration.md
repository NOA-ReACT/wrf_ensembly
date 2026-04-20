# Configuration

WRF Ensembly is configured through a single TOML configuration file, typically named `config.toml` in the experiment directory. This file contains all the settings needed to run an ensemble assimilation experiment, from model directories to SLURM job parameters.

The configuration is structured into several sections, each controlling different aspects of the experiment. Below is a comprehensive reference of all available configuration options.

## Template System

When creating a new experiment, you can start from a built-in template instead of writing a config from scratch:

```bash
wrf-ensembly $EXP_PATH experiment create <template>
```

Available templates:

| Template | Description |
|----------|-------------|
| `default` | Minimal starting point with all required fields and sensible defaults. Use this as a blank slate. |
| `aris_chem_4.5.2` | WRF-CHEM v4.5.2 configuration for the ARIS HPC cluster. Covers the North Atlantic domain at 30 km resolution with dust/sea salt aerosols. |
| `iridium_chem_4.6.0` | WRF-CHEM v4.6.0 configuration for the Iridium cluster. Smaller domain suitable for development runs. |

Templates are TOML files stored in `wrf_ensembly/config_templates/`. You can inspect them directly for reference.

## Environment-Specific Overrides (`env_config.toml`)

To separate experiment settings from cluster-specific settings, you can place an `env_config.toml` file alongside your `config.toml`. When present, it is deep-merged with the main config, with `env_config.toml` values taking precedence.

This is useful when managing multiple HPC environments:

```
config.toml              # Main experiment config (committed to version control)
env_config.toml -> env_aris.toml   # Symlink to environment-specific config
env_aris.toml            # ARIS cluster settings (paths, SLURM directives)
env_iridium.toml         # Iridium cluster settings
```

Switch environments by changing the symlink target.

## Configuration Structure

The configuration file is organized into the following main sections:

- **[metadata](#metadata)** - Basic experiment information
- **[directories](#directories)** - Paths to model installations and data
- **[domain_control](#domain-control)** - Grid and projection settings
- **[time_control](#time-control)** - Experiment timing and cycle configuration
- **[data](#data)** - Input data locations and settings
- **[assimilation](#assimilation)** - Ensemble and DART configuration
- **[observations](#observations)** - Observation processing settings
- **[validation](#validation)** - Validation and first-departure analysis
- **[geogrid](#geogrid)** - Geographical data preprocessing
- **[perturbations](#perturbations)** - Initial condition perturbation settings
- **[slurm](#slurm)** - SLURM job configuration
- **[postprocess](#postprocess)** - Post-processing and output settings
- **[plots](#plots)** - Diagnostic plot configuration
- **[environment](#environment)** - Environment variables
- **[wrf_namelist](#wrf-namelist)** - WRF namelist overrides
- **[dart_namelist](#dart-namelist)** - DART namelist overrides
- **[extra_dart_files](#extra-dart-files)** - Extra files to copy into the DART directory

## Metadata

Basic information about the experiment.

```toml
[metadata]
name = "my_experiment"
description = "A description of what this experiment does"
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | **Required.** Name of the experiment |
| `description` | string | **Required.** Description of the experiment |

## Directories

Paths to model installations and data directories.

```toml
[directories]
wrf_root = "/path/to/WRF"
wps_root = "/path/to/WPS"
dart_root = "/path/to/DART"
scratch_root = "./scratch"
```

| Field | Type | Description |
|-------|------|-------------|
| `wrf_root` | Path | **Required.** Root directory of the WRF model. Should contain the `run` directory with `real.exe` compiled |
| `wps_root` | Path | **Required.** Root directory of WPS. Should contain the `geogrid.exe`, `metgrid.exe` and `ungrib.exe` executables |
| `dart_root` | Path | **Required.** Root directory of DART. Should contain a `models/wrf` directory, compiled |
| `scratch_root` | Path | Scratch directory for temporarily storing model output files before post-processing. If relative, will be inside the experiment directory. *Default: `./scratch`* |

## Domain Control

Grid configuration and map projection settings.

```toml
[domain_control]
xy_resolution = [30, 30]  # km
xy_size = [340, 130]      # grid points
projection = "lambert"
ref_lat = 20.0
ref_lon = -17.0
truelat1 = 20.0
truelat2 = 18.0
stand_lon = -11.0
```

| Field | Type | Description |
|-------|------|-------------|
| `xy_resolution` | [int, int] | **Required.** Space between grid points in x and y directions (kilometers). Corresponds to WRF's `dx` and `dy` |
| `xy_size` | [int, int] | **Required.** Number of grid points in x and y directions. Corresponds to WRF's `e_we` and `e_sn` |
| `projection` | string | **Required.** Map projection for the grid |
| `ref_lat` | float | **Required.** Reference latitude for the projection |
| `ref_lon` | float | **Required.** Reference longitude for the projection |
| `truelat1` | float | **Required.** First true latitude for the projection |
| `truelat2` | float | Second true latitude for the projection |
| `stand_lon` | float | Standard longitude for the projection |
| `pole_lat` | float | Pole latitude for the projection |
| `pole_lon` | float | Pole longitude for the projection |

## Time Control

Experiment timing, cycle configuration, and I/O settings.

```toml
[time_control]
start = 2025-03-01T00:00:00Z
end = 2025-04-30T00:00:00Z
boundary_update_interval = 180  # minutes
output_interval = 60           # minutes
analysis_interval = 180        # minutes
runtime_io = ["+:h:0:EDUST1,EDUST2,EDUST3,EDUST4,EDUST5"]

# Per-cycle overrides
[time_control.cycles.0]
duration = 120
output_interval = 30
```

| Field | Type | Description |
|-------|------|-------------|
| `start` | datetime | **Required.** Start timestamp of the experiment |
| `end` | datetime | **Required.** End timestamp of the experiment |
| `boundary_update_interval` | int | Time between incoming real data (lateral boundary conditions) in minutes. *Default: 180* |
| `output_interval` | int | Time between output (history) files in minutes. *Default: 60* |
| `analysis_interval` | int | Time between analysis/assimilation cycles in minutes. *Default: 360* |
| `runtime_io` | [string] | Runtime I/O options for WRF. Creates a text file in each member directory for `iofields_filename`. See [WRF I/O documentation](https://github.com/wrf-model/WRF/blob/master/doc/README.io_config) |
| `cycles` | dict | Per-cycle configuration overrides. Keys are cycle numbers (0-indexed) |

### Per-Cycle Configuration

You can override certain settings for specific cycles:

| Field | Type | Description |
|-------|------|-------------|
| `duration` | int | Override the cycle duration in minutes |
| `output_interval` | int | Override the output interval for this cycle |

## Data

Input data locations and processing settings.

```toml
[data]
wps_geog = "/path/to/WPS_GEOG"
meteorology = "/path/to/meteorology"
meteorology_glob = "*.grib"
meteorology_vtable = "Vtable.ERA-interim.pl"
per_member_meteorology = false
manage_chem_ic = false

# Chemistry data (optional)
[data.chemistry]
path = "/path/to/chemistry/data"
model_name = "cams_global_forecasts"
```

| Field | Type | Description |
|-------|------|-------------|
| `wps_geog` | Path | **Required.** Directory containing WPS geographical data |
| `meteorology` | Path | **Required.** Directory containing meteorological GRIB files |
| `meteorology_glob` | string | Glob pattern for finding meteorological files. *Default: `"*.grib"`* |
| `meteorology_vtable` | Path | Vtable file for meteorological data. *Default: `"Vtable.ERA-interim.pl"`* |
| `per_member_meteorology` | bool | Whether to use separate meteorology for each member. If true, `meteorology` should contain `%MEMBER%` placeholder. *Default: false* |
| `manage_chem_ic` | bool | Whether to manage chemical initial conditions. Sets `chem_in_opt` to 0 for `real.exe` and 1 for `wrf.exe`. *Default: false* |

### Chemistry Data

Optional configuration for chemistry model data (used with WRF-CHEM):

| Field | Type | Description |
|-------|------|-------------|
| `path` | Path | **Required.** Directory containing chemistry data in YYYY-MM-DD subdirectories |
| `model_name` | string | **Required.** Name of the chemistry model (e.g., "cams_global_forecasts") |

## Assimilation

Ensemble configuration and DART settings.

```toml
[assimilation]
n_members = 30
cycled_variables = ["U", "V", "P", "PH", "THM", "MU", "QVAPOR"]
state_variables = ["U", "V", "W", "PH", "THM", "MU", "QVAPOR", "PSFC"]
filter_mpi_tasks = 24
half_window_length_minutes = 30
use_inflation = false
```

| Field | Type | Description |
|-------|------|-------------|
| `n_members` | int | **Required.** Number of ensemble members |
| `cycled_variables` | [string] | **Required.** Variables to carry forward from the previous cycle |
| `state_variables` | [string] | **Required.** Variables to include in the state vector for assimilation |
| `filter_mpi_tasks` | int | Number of MPI tasks for DART filter. If != 1, filter runs with MPI. *Default: 1* |
| `half_window_length_minutes` | int | Half-length of the observation window in minutes. Observations within this window around the analysis time are used. *Default: 30* |
| `use_inflation` | bool | Whether to manage DART inflation files between cycles (sets `inf_initial_from_restart`/`inf_sd_initial_from_restart` appropriately). You still need to configure inflation in the DART namelist. *Default: false* |

## Observations

Observation processing and quality control settings.

```toml
[observations]
boundary_width = 2.0
boundary_error_factor = 2.5
boundary_error_width = 1.0
instruments_to_assimilate = ["MODIS", "AERONET"]
error_inflation_factor = { "sonde.U" = 1.5 }

[observations.superobs.AEOLUS_L2B_MIE.HLOS_WIND]
hoz_bin_sizes = { along_track = 5 }
vert_bin_sizes = { height_bin = 2 }

[observations.thinning.MODIS.AOD_550nm]
keep_every_n = 3

[observations.temporal_binning.AERONET.AOD_550nm]
bin_minutes = 60
offset_minutes = -30
```

| Field | Type | Description |
|-------|------|-------------|
| `instruments_to_assimilate` | [string] | Which instruments to assimilate. If not set, all available instruments are used |
| `error_inflation_factor` | dict | Per `instrument.quantity` error inflation factors applied during cycle preparation |
| `boundary_width` | float | How many grid points to reduce the domain by when removing observations outside the domain. *Default: 0* |
| `boundary_error_factor` | float | Factor to inflate observation errors near the boundary. *Default: 2.5* |
| `boundary_error_width` | float | Width in grid points where boundary error inflation is applied. Set to 0 to disable. *Default: 1.0* |
| `superobs` | dict | Superobservation configuration per `instrument.quantity` pair |
| `thinning` | dict | Stride thinning configuration per `instrument.quantity` pair |
| `temporal_binning` | dict | Temporal binning configuration per `instrument.quantity` pair |

### Superobservations

Superobbing merges nearby observations into one. Configure per `instrument.quantity`:

| Field | Type | Description |
|-------|------|-------------|
| `hoz_bin_sizes` | dict | Horizontal dimension bin sizes (dimension name → bin length) |
| `vert_bin_sizes` | dict | Vertical dimension bin sizes (dimension name → bin length) |

### Stride Thinning

Keeps every N-th good-QC observation; the rest become validation hold-outs (`qc_flag = -1`). Applied after superobbing.

| Field | Type | Description |
|-------|------|-------------|
| `keep_every_n` | int | Keep every N-th good-QC observation |

### Temporal Binning

Bins observations into time windows. Incompatible with superobbing for the same pair.

| Field | Type | Description |
|-------|------|-------------|
| `bin_minutes` | int | Width of each time window in minutes |
| `offset_minutes` | int | Shift bin boundaries by this many minutes relative to UTC midnight. *Default: 0* |

## Validation

Settings for validating the experiment against observations.

```toml
[validation]
instruments = ["MODIS", "AERONET"]

[validation.first_departures]
instrument_quantity_pairs = ["MODIS.AOD_550nm", "AERONET.AOD_550nm"]
bias_map_colorbar_ranges = { "MODIS.AOD_550nm" = [-0.5, 0.5] }
excluded_bboxes = [[-10.0, -20.0, 10.0, 20.0]]

[[validation.first_departures.regimes]]
instrument = "MODIS"
quantity = "AOD_550nm"
bins = [0.0, 0.2, 0.5, 1.0]
labels = ["low", "medium", "high"]
spatial_resolution = 1.0
```

| Field | Type | Description |
|-------|------|-------------|
| `instruments` | [string] | Instruments to use for validation. *Default: all available* |

### First Departures (`validation.first_departures`)

Configuration for O-B (observation minus background) departure analysis:

| Field | Type | Description |
|-------|------|-------------|
| `instrument_quantity_pairs` | [string] | `instrument.quantity` pairs to analyze. *Default: all available* |
| `bias_map_colorbar_ranges` | dict | Colorbar ranges for bias maps per `instrument.quantity` pair (overrides auto-scaling) |
| `excluded_bboxes` | [[float]] | Bounding boxes to exclude from analysis. Each entry is `[min_lat, min_lon, max_lat, max_lon]` |
| `regimes` | [RegimeConfig] | Regime-based analysis configurations |

Each regime entry (`[[validation.first_departures.regimes]]`):

| Field | Type | Description |
|-------|------|-------------|
| `instrument` | string | Instrument name |
| `quantity` | string | Observation quantity |
| `bins` | [float] | Bin edges for regime classification |
| `labels` | [string] | Label for each regime (length = `len(bins) - 1`) |
| `spatial_resolution` | float | Spatial binning resolution in degrees. *Default: 1.0* |

## Geogrid

Geographical data preprocessing settings.

```toml
[geogrid]
table = "GEOGRID.TBL.ARW_CHEM"
```

| Field | Type | Description |
|-------|------|-------------|
| `table` | string | Name of the GEOGRID table file to use. *Default: `"GEOGRID.TBL"`* |

## Perturbations

Initial condition perturbation settings for ensemble generation.

```toml
[perturbations]
seed = 42

# Per-variable perturbation settings
[perturbations.variables.DUST_EMIS_WEIGHT]
operation = "multiply"
mean = 1.0
sd = 0.5
gaussian_sigma = 2.5
boundary = 0
min_value = 0.1
max_value = 3.0
perturb_every_cycle = false
different_field_every_cycle = true
midcycle_taper_width = 0
```

| Field | Type | Description |
|-------|------|-------------|
| `seed` | int | Random seed for perturbation generation. If not set, randomly generated |
| `variables` | dict | Per-variable perturbation configuration |

### Per-Variable Perturbation Settings

| Field | Type | Description |
|-------|------|-------------|
| `operation` | "add", "multiply", or "assign" | **Required.** How to apply the perturbation to the variable |
| `mean` | float | Mean of the perturbation field. *Default: 1.0* |
| `sd` | float | Standard deviation of the perturbation field. *Default: 1.0* |
| `gaussian_sigma` | float | Standard deviation for the Gaussian smoothing filter applied to the perturbation field. *Default: 2.5* |
| `boundary` | int | Size of perturbation boundary in grid points. If > 0, edges won't be perturbed (with a smoothing filter). *Default: 0* |
| `min_value` | float | Minimum value for the perturbation field |
| `max_value` | float | Maximum value for the perturbation field |
| `perturb_every_cycle` | bool | Whether to generate a new perturbation at every cycle. If false, only perturbed at the first cycle. *Default: false* |
| `different_field_every_cycle` | bool | If `perturb_every_cycle` is true, whether to use a different random field each cycle. *Default: true* |
| `midcycle_taper_width` | int | Width of the tapering region at domain edges during mid-cycles. Only applies when `perturb_every_cycle` is true. Tapers the perturbation from full weight at the edge to 0 over this many grid points. Does not work with `operation = "assign"`. *Default: 0* |

## SLURM

SLURM job configuration and resource allocation.

```toml
[slurm]
sbatch_command = "sbatch --parsable"
command_prefix = "micromamba run -n wrf"
mpirun_command = "mpirun"
env_modules = ["intel/2021.4"]
pre_commands = ["export OMP_NUM_THREADS=1"]

# Baseline directives applied to all jobs
[slurm.directives.default]
partition = "compute"
nodes = 1
cpus-per-task = 1

# Job-type-specific overrides
[slurm.directives.advance_model]
ntasks-per-node = 24

[slurm.directives.preprocess]
ntasks-per-node = 24

[slurm.directives.make_analysis]
ntasks-per-node = 1

[slurm.directives.postprocess]
ntasks-per-node = 4
```

| Field | Type | Description |
|-------|------|-------------|
| `sbatch_command` | string | Command for submitting SLURM jobs. *Default: `"sbatch --parsable"`* |
| `command_prefix` | string | Prefix for all `wrf-ensembly` commands (e.g., for conda/micromamba environment activation) |
| `mpirun_command` | string | Command for running MPI jobs. *Default: `"mpirun"`* |
| `env_modules` | [string] | Environment modules to load in each job |
| `pre_commands` | [string] | Shell commands to run at the start of each job |

### SLURM Directives (`slurm.directives`)

Directives are organized by job type. Job-type-specific sections override the `default` section. Any `#SBATCH` directive can be specified as a key.

| Section | Description |
|---------|-------------|
| `slurm.directives.default` | Baseline directives applied to all jobs |
| `slurm.directives.advance_model` | Overrides for ensemble member advance jobs (WRF forward runs) |
| `slurm.directives.preprocess` | Overrides for preprocessing jobs (WPS, real.exe) |
| `slurm.directives.make_analysis` | Overrides for filter/analysis/cycle jobs |
| `slurm.directives.postprocess` | Overrides for post-processing jobs |

## Postprocess

Post-processing settings for model output.

```toml
[postprocess]
variables_to_keep = ["DUST_\\d", "U", "V", "wind_.*"]
variables_to_keep_ensemble = ["DUST_\\d"]
compression = "zlib"
compression_level = 4
shuffle = true
significant_digits = 3
significant_digits_overrides = { "Z.*" = 6, "X.*" = 6 }
quantize_mode = "GranularBitRound"
keep_per_member = false
compute_ensemble_statistics_in_job = true

[[postprocess.processors]]
processor = "my_package.processors:CustomProcessor"
params = { param = "value" }

[[postprocess.processors]]
processor = "/path/to/custom_processor.py:MyProcessor"
params = { custom_param = "value" }
```

| Field | Type | Description |
|-------|------|-------------|
| `variables_to_keep` | [string] | Regular expressions for variables to keep in output. If not set, all variables are kept |
| `variables_to_keep_ensemble` | [string] | Subset of variables to keep in per-member ensemble files. If not set, uses `variables_to_keep`. Only applies when `keep_per_member = true` |
| `compression` | string | Compression algorithm: `"zlib"`, `"zstd"`, `"bzip2"`, `"szip"`, or `"none"`. Note: `zstd` and `bzip2` require HDF5 plugins. *Default: `"zlib"`* |
| `compression_level` | int | Compression level (0–9). 0 disables compression. *Default: 4* |
| `shuffle` | bool | Apply the shuffle filter before compression (improves compression ratios). *Default: true* |
| `significant_digits` | int | Number of significant digits for quantization (lossy compression). 0 disables quantization. Requires netcdf-c >= 4.9.0. *Default: 3* |
| `significant_digits_overrides` | dict | Per-variable regex overrides for `significant_digits`. *Default: `{"Z.*": 6, "X.*": 6}`* |
| `quantize_mode` | string | Quantization algorithm: `"BitGroom"`, `"BitRound"`, or `"GranularBitRound"`. *Default: `"GranularBitRound"`* |
| `keep_per_member` | bool | Whether to also produce concatenated per-member files. *Default: false* |
| `compute_ensemble_statistics_in_job` | bool | Whether to compute ensemble mean/spread in SLURM jobs. Disable for single-member experiments. *Default: true* |
| `processors` | [ProcessorConfig] | List of custom data processors to apply |

### Data Processors

WRF Ensembly supports custom data processors for post-processing model output. The built-in `XWRFProcessor` (which applies CF-compliance diagnostics via xwrf) is always applied first.

#### Custom Processors

Specify custom processors using:
- Module path: `"my_package.processors:CustomProcessor"`
- File path: `"/path/to/file.py:MyProcessor"`

Each processor entry has:

| Field | Type | Description |
|-------|------|-------------|
| `processor` | string | Processor identifier (module path or file path) |
| `params` | dict | Processor-specific parameters |

## Plots

Configuration for diagnostic plots generated during post-processing.

```toml
[plots.forecast_vs_analysis]
arrangement = "horizontal"
dpi = 150
include_spread = false

[[plots.forecast_vs_analysis.variables]]
name = "DUST_1"
level = 0
vmin = 0.0
vmax = 100.0
cmap = "YlOrBr"

[plots.forecasts]
dpi = 150
include_spread = false

[[plots.forecasts.variables]]
name = "U"
pressure_level = 850.0
```

### Forecast vs Analysis Plots (`plots.forecast_vs_analysis`)

Side-by-side comparison of forecast and analysis fields:

| Field | Type | Description |
|-------|------|-------------|
| `arrangement` | string | Panel arrangement: `"horizontal"` (1×3) or `"vertical"` (3×1). *Default: `"horizontal"`* |
| `dpi` | int | Image resolution. *Default: 150* |
| `include_spread` | bool | Whether to also generate spread comparison plots. *Default: false* |
| `variables` | [PlotVariableConfig] | Variables to plot |

### Forecast-Only Plots (`plots.forecasts`)

| Field | Type | Description |
|-------|------|-------------|
| `dpi` | int | Image resolution. *Default: 150* |
| `include_spread` | bool | Whether to also generate spread plots. *Default: false* |
| `variables` | [PlotVariableConfig] | Variables to plot |

### Plot Variable Configuration

Each `[[plots.*.variables]]` entry supports:

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | **Required.** Variable name in the netCDF file |
| `level` | int | Vertical level index to select. Omit for 2D variables |
| `pressure_level` | float | Pressure level in hPa to interpolate to (requires `air_pressure` in dataset). Takes precedence over `level` |
| `extent` | [float, float, float, float] | Geographic extent `[min_lon, max_lon, min_lat, max_lat]` |
| `vmin` / `vmax` | float | Colorbar range for forecast/analysis panels |
| `diff_vmin` / `diff_vmax` | float | Colorbar range for the difference panel |
| `spread_vmin` / `spread_vmax` | float | Colorbar range for spread panels |
| `cmap` | string | Colormap for forecast/analysis panels. *Default: `"viridis"`* |
| `diff_cmap` | string | Colormap for the difference panel. *Default: `"RdBu_r"`* |
| `spread_cmap` | string | Colormap for spread panels. *Default: `"viridis"`* |

## Environment

Environment variables to set when running the experiment.

```toml
[environment]
# Applied to all commands
universal = { OMP_NUM_THREADS = "1", MALLOC_TRIM_THRESHOLD = "536870912" }

# Applied only to WRF/WPS commands
wrf = { WRF_EM_CORE = "1" }

# Applied only to DART commands
dart = { DART_DEBUG = "1" }
```

| Field | Type | Description |
|-------|------|-------------|
| `universal` | dict | Environment variables applied to all commands |
| `wrf` | dict | Environment variables applied only to WRF/WPS commands |
| `dart` | dict | Environment variables applied only to DART commands |

## WRF Namelist

WRF namelist overrides and per-member customizations.

```toml
[wrf_namelist]
# Global namelist overrides
[wrf_namelist.time_control]
history_interval = 60
restart_interval = 3600

[wrf_namelist.domains]
time_step = 180
max_dom = 1

[wrf_namelist.physics]
mp_physics = 10
ra_lw_physics = 4

# Per-member namelist overrides
[wrf_namelist_per_member.member_001.physics]
mp_physics = 8

[wrf_namelist_per_member.member_002.physics]
mp_physics = 6
```

You can override any WRF namelist variable by specifying it in the appropriate section. The structure follows the WRF namelist format with sections like `time_control`, `domains`, `physics`, etc.

For per-member customizations, use the `wrf_namelist_per_member` section with the member name (e.g., `member_001`) as the key.

## DART Namelist

Overrides for the DART `input.nml` namelist. The structure mirrors the DART namelist sections.

```toml
[dart_namelist]

[dart_namelist.filter_nml]
num_output_state_members = 0
inf_flavor = [2, 0]
inf_initial = [1.0, 1.0]
```

Any DART namelist variable can be overridden here. The structure follows DART's `input.nml` format.

## Extra DART Files

Extra files to copy into the DART working directory before running assimilation. Useful for observation error tables, inflation files, or other DART input files.

```toml
[[extra_dart_files]]
source = "/path/to/obs_error_table.txt"
destination_name = "obs_error_table"  # Optional; defaults to source filename

[[extra_dart_files]]
source = "/path/to/inflation_mean.nc"
```

| Field | Type | Description |
|-------|------|-------------|
| `source` | Path | **Required.** Path to the source file |
| `destination_name` | string | Filename inside the DART directory. If not set, uses the source filename |

## Example Configuration

Here's a complete example configuration file:

```toml
[metadata]
name = "dust_experiment"
description = "North African dust assimilation experiment"

[directories]
wrf_root = "/opt/WRF-4.5"
wps_root = "/opt/WPS-4.5"
dart_root = "/opt/DART"

[domain_control]
xy_resolution = [30, 30]
xy_size = [340, 130]
projection = "lambert"
ref_lat = 20.0
ref_lon = -17.0
truelat1 = 20.0
truelat2 = 18.0
stand_lon = -11.0

[time_control]
start = 2025-03-01T00:00:00Z
end = 2025-03-31T00:00:00Z
boundary_update_interval = 180
output_interval = 60
analysis_interval = 360

[data]
wps_geog = "/data/WPS_GEOG"
meteorology = "/data/ERA5"
meteorology_vtable = "Vtable.ERA-interim.pl"
manage_chem_ic = true

[data.chemistry]
path = "/data/CAMS"
model_name = "cams_global_forecasts"

[assimilation]
n_members = 20
cycled_variables = ["U", "V", "P", "PH", "THM", "MU", "QVAPOR", "DUST_1", "DUST_2", "DUST_3", "DUST_4", "DUST_5"]
state_variables = ["U", "V", "W", "PH", "THM", "MU", "QVAPOR", "PSFC", "DUST_1", "DUST_2", "DUST_3", "DUST_4", "DUST_5"]
filter_mpi_tasks = 20
half_window_length_minutes = 30
use_inflation = false

[perturbations]
[perturbations.variables.DUST_EMIS_WEIGHT]
operation = "multiply"
mean = 1.0
sd = 0.5
gaussian_sigma = 2.5
min_value = 0.1
max_value = 3.0

[slurm]
mpirun_command = "mpirun"
env_modules = ["intel/2021.4"]

[slurm.directives.default]
partition = "compute"
nodes = 1
cpus-per-task = 1

[slurm.directives.advance_model]
ntasks-per-node = 20

[slurm.directives.preprocess]
ntasks-per-node = 20

[slurm.directives.make_analysis]
ntasks-per-node = 1

[slurm.directives.postprocess]
ntasks-per-node = 4

[postprocess]
variables_to_keep = ["DUST_\\d", "U", "V", "wind_.*"]
compression = "zlib"
compression_level = 4
significant_digits = 3
compute_ensemble_statistics_in_job = true

[wrf_namelist.domains]
time_step = 180
max_dom = 1

[wrf_namelist.physics]
mp_physics = 10
ra_lw_physics = 4
```

This configuration sets up a dust assimilation experiment with 20 ensemble members, running on a Lambert conformal conic projection grid over North Africa, with 6-hour assimilation cycles.
