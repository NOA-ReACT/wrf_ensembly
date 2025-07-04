# Configuration

WRF Ensembly is configured through a single TOML configuration file, typically named `config.toml` in the experiment directory. This file contains all the settings needed to run an ensemble assimilation experiment, from model directories to SLURM job parameters.

The configuration is structured into several sections, each controlling different aspects of the experiment. Below is a comprehensive reference of all available configuration options.

## Configuration Structure

The configuration file is organized into the following main sections:

- **[metadata](#metadata)** - Basic experiment information
- **[directories](#directories)** - Paths to model installations and data
- **[domain_control](#domain-control)** - Grid and projection settings
- **[time_control](#time-control)** - Experiment timing and cycle configuration
- **[data](#data)** - Input data locations and settings
- **[assimilation](#assimilation)** - Ensemble and DART configuration
- **[observations](#observations)** - Observation processing settings
- **[geogrid](#geogrid)** - Geographical data preprocessing
- **[perturbations](#perturbations)** - Initial condition perturbation settings
- **[slurm](#slurm)** - SLURM job configuration
- **[postprocess](#postprocess)** - Post-processing and output settings
- **[environment](#environment)** - Environment variables
- **[wrf_namelist](#wrf-namelist)** - WRF namelist overrides

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
```

| Field | Type | Description |
|-------|------|-------------|
| `n_members` | int | **Required.** Number of ensemble members |
| `cycled_variables` | [string] | **Required.** Variables to carry forward from the previous cycle |
| `state_variables` | [string] | **Required.** Variables to include in the state vector for assimilation |
| `filter_mpi_tasks` | int | Number of MPI tasks for DART filter. If != 1, filter runs with MPI. *Default: 1* |

## Observations

Observation processing and quality control settings.

```toml
[observations]
boundary_width = 2.0
boundary_error_factor = 2.5
boundary_error_width = 1.0
```

| Field | Type | Description |
|-------|------|-------------|
| `boundary_width` | float | How many grid points to reduce the domain by when removing observations outside the domain. *Default: 0* |
| `boundary_error_factor` | float | Factor to inflate observation errors near the boundary. *Default: 2.5* |
| `boundary_error_width` | float | Width in grid points where boundary error inflation is applied. Set to 0 to disable. *Default: 1.0* |

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
apply_perturbations_every_cycle = false

# Per-variable perturbation settings
[perturbations.variables.DUST_EMIS_WEIGHT]
operation = "multiply"
mean = 2.6
sd = 0.8
rounds = 20
boundary = 0
min_value = 0.1
max_value = 10.0
```

| Field | Type | Description |
|-------|------|-------------|
| `seed` | int | Random seed for perturbation generation. If not set, randomly generated |
| `apply_perturbations_every_cycle` | bool | Whether to apply perturbations at the start of every cycle. *Default: false* |
| `variables` | dict | Per-variable perturbation configuration |

### Per-Variable Perturbation Settings

| Field | Type | Description |
|-------|------|-------------|
| `operation` | "add" or "multiply" | **Required.** Whether to add or multiply the perturbation |
| `mean` | float | Mean of the perturbation field. *Default: 1.0* |
| `sd` | float | Standard deviation of the perturbation field. *Default: 1.0* |
| `rounds` | int | Number of smoothing rounds to apply. *Default: 10* |
| `boundary` | int | Size of perturbation boundary in grid points. If > 0, edges won't be perturbed. *Default: 0* |
| `min_value` | float | Minimum value for the perturbation field |
| `max_value` | float | Maximum value for the perturbation field |

## SLURM

SLURM job configuration and resource allocation.

```toml
[slurm]
sbatch_command = "sbatch --parsable"
command_prefix = "micromamba run -n wrf"
mpirun_command = "mpirun"
env_modules = ["intel/2021.4"]
pre_commands = ["export OMP_NUM_THREADS=1"]

# Job resource configurations
[slurm.directives_large]
partition = "compute"
nodes = 2
ntasks-per-node = 24
cpus-per-task = 1
mem = "64G"

[slurm.directives_small]
partition = "compute"
nodes = 1
ntasks-per-node = 8
cpus-per-task = 1
mem = "16G"

[slurm.directives_postprocess]
partition = "compute"
nodes = 1
ntasks = 24
cpus-per-task = 1
mem = "32G"
```

| Field | Type | Description |
|-------|------|-------------|
| `sbatch_command` | string | Command for submitting SLURM jobs. *Default: `"sbatch --parsable"`* |
| `command_prefix` | string | Prefix for all `wrf-ensembly` commands (e.g., conda environment activation) |
| `mpirun_command` | string | Command for running MPI jobs. *Default: `"mpirun"`* |
| `env_modules` | [string] | Environment modules to load in each job |
| `pre_commands` | [string] | Commands to run at the start of each job |
| `directives_large` | dict | SLURM directives for large jobs (ensemble member advance) |
| `directives_small` | dict | SLURM directives for small jobs (Python steps) |
| `directives_postprocess` | dict | SLURM directives for post-processing jobs |

## Postprocess

Post-processing settings for model output.

```toml
[postprocess]
variables_to_keep = ["DUST_\\d", "U", "V", "wind_.*"]
compression_filters = "shf|dfl"
ppc_filter = "default=3#Z.*=6#X.*=6"
keep_per_member = false
compute_ensemble_statistics_in_job = true
processor_cores = 1
statistics_cores = 24
concatenate_cores = 24
cdo_path = "cdo"
ncrcat_cmd = "ncrcat"

# Custom processors
[[postprocess.processors]]
processor = "script"
params = { script = "python enhance_data.py {in} {out}" }

[[postprocess.processors]]
processor = "/path/to/custom_processor.py:MyProcessor"
params = { custom_param = "value" }
```

| Field | Type | Description |
|-------|------|-------------|
| `variables_to_keep` | [string] | Regular expressions for variables to keep in output. If not set, all variables are kept |
| `compression_filters` | string | NCO compression filters to apply. *Default: `"shf|zst,3"`* |
| `ppc_filter` | string | Lossy quantization settings for precision control. *Default: `"default=3#Z.*=6#X.*=6"`* |
| `keep_per_member` | bool | Whether to keep per-member files in addition to ensemble statistics. *Default: false* |
| `compute_ensemble_statistics_in_job` | bool | Whether to compute ensemble statistics in SLURM jobs. *Default: true* |
| `processor_cores` | int | Number of cores for processor pipeline. *Default: 1* |
| `statistics_cores` | int | Number of cores for statistics computation. *Default: 1* |
| `concatenate_cores` | int | Number of cores for concatenation step. *Default: 1* |
| `cdo_path` | string | Path to CDO executable. *Default: `"cdo"`* |
| `ncrcat_cmd` | string | Path to ncrcat executable. *Default: `"ncrcat"`* |
| `processors` | [ProcessorConfig] | List of custom data processors to apply |

### Data Processors

WRF Ensembly supports custom data processors for post-processing model output. The built-in `XWRFProcessor` is always applied first.

#### Built-in Processors

- **script**: Execute external scripts (for backward compatibility)

#### Custom Processors

You can specify custom processors using:
- Module path: `"my_package.processors:CustomProcessor"`
- File path: `"/path/to/file.py:MyProcessor"`

Each processor can have custom parameters passed via the `params` dictionary.

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

[perturbations]
apply_perturbations_every_cycle = false

[perturbations.variables.DUST_EMIS_WEIGHT]
operation = "multiply"
mean = 1.0
sd = 0.5
rounds = 15
min_value = 0.1
max_value = 3.0

[slurm.directives_large]
partition = "compute"
nodes = 1
ntasks-per-node = 20
mem = "64G"

[postprocess]
variables_to_keep = ["DUST_\\d", "U", "V", "wind_.*"]
compression_filters = "shf|dfl"
compute_ensemble_statistics_in_job = true
```

This configuration sets up a dust assimilation experiment with 20 ensemble members, running on a Lambert conformal conic projection grid over North Africa, with 6-hour assimilation cycles.
