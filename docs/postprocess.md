# Postprocessing

WRF Ensembly provides a comprehensive postprocessing pipeline for transforming raw WRF output files into analysis-ready datasets. The postprocessing system consists of several stages that can be executed individually or through SLURM jobs, allowing for flexible and efficient data processing workflows.

## Overview

The postprocessing pipeline transforms raw `wrfout` files through several stages:

1. **Data Processing Pipeline** - Applies custom transformations and standardizes the data format
2. **Ensemble Statistics** - Computes mean and standard deviation across ensemble members
3. **Concatenation** - Combines temporal files into single files per cycle with compression
4. **Cleanup** - Removes temporary files to save disk space

All postprocessing commands follow the pattern:

```bash
wrf-ensembly EXPERIMENT_PATH postprocess COMMAND [OPTIONS]
```

## File Transformations and Output Paths

The postprocessing pipeline transforms raw WRF output files in several key ways, making them more suitable for analysis and visualization while reducing file sizes.

### Key Transformations

**Coordinate Standardization:**
- Time dimension renamed from `XTIME` to `t` with proper CF-compliant attributes
- Latitude/longitude coordinates renamed from `XLAT`/`XLONG` to `latitude`/`longitude`
- Standardized coordinate attributes (`standard_name`, `axis`, `units`)

**Data Processing:**
- **Destaggering**: WRF's staggered grids are destaggered to regular grids
- **Diagnostic variables**: New variables computed (e.g., `air_density`, `air_pressure`, `geopotential_height`)
- **Wind transformation**: Grid-relative winds (`U`, `V`) converted to earth-relative (`wind_east`, `wind_north`)
- **Variable filtering**: Only specified variables retained based on `variables_to_keep` configuration

**File Structure:**
- Ensemble statistics computed: `*_mean` and `*_sd` files for each time step
- Temporal concatenation: Individual time files combined into single files per cycle
- Compression applied: Lossless compression and optional quantization reduce file sizes
- Custom variables: User-defined processors can add derived variables (e.g., extinction coefficients, AOD)

### Output File Paths

The postprocessing system creates files in several locations within your experiment directory:

**Intermediate Files (scratch directory):**
```
EXPERIMENT_PATH/scratch/
├── analysis/cycle_XXX/member_YY/
│   ├── wrfout_*               # Raw WRF analysis files
│   ├── wrfout_*_post          # Processed individual files
│   ├── wrfout_*_mean          # Ensemble mean files
│   └── wrfout_*_sd            # Ensemble std deviation files
└── forecasts/cycle_XXX/member_YY/
    ├── wrfout_*               # Raw WRF forecast files
    ├── wrfout_*_post          # Processed individual files
    ├── wrfout_*_mean          # Ensemble mean files
    └── wrfout_*_sd            # Ensemble std deviation files
```

**Final Output Files (data directory):**
```
EXPERIMENT_PATH/data/
├── analysis/cycle_XXX/
│   ├── analysis_mean_cycle_XXX.nc   # Concatenated analysis ensemble mean
│   └── analysis_sd_cycle_XXX.nc     # Concatenated analysis ensemble std dev
└── forecasts/cycle_XXX/
    ├── forecast_mean_cycle_XXX.nc   # Concatenated forecast ensemble mean
    ├── forecast_sd_cycle_XXX.nc     # Concatenated forecast ensemble std dev
    └── forecast_member_YY_cycle_XXX.nc  # Individual members (if keep_per_member=true)
```

The final concatenated files in the `data/` directory are the primary analysis products, containing all time steps for a cycle in CF-compliant NetCDF format with optimized compression.

### Customization Options

Key configuration options for controlling the transformations:

- **`variables_to_keep`**: Regular expressions to filter which variables are retained
- **`compression_filters`**: Control lossless compression algorithms (e.g., `"shf|zst,3"`)
- **`ppc_filter`**: Precision control for lossy quantization (e.g., `"default=3#Z.*=6"`)
- **`keep_per_member`**: Whether to save individual member files alongside ensemble statistics
- **`processors`**: Custom data processors for specialized transformations

For complete configuration details, see the [Configuration documentation](configuration.md#postprocess).

## Commands

### print-variables-to-keep

```bash
wrf-ensembly EXPERIMENT_PATH postprocess print-variables-to-keep
```

Prints which variables will be kept in the wrfout files after postprocessing. This is useful to check if the variables you want to keep are actually kept or check how the regex filters are applied.

The variables are defined in the `PostprocessConfig.variables_to_keep` configuration option. This command requires that cycle 0 output for member 0 exists.

**Example output:**
```
DUST_\\d -> DUST_1: (('Time', 'bottom_top', 'south_north', 'west_east',)) (float32)
DUST_\\d -> DUST_2: (('Time', 'bottom_top', 'south_north', 'west_east',)) (float32)
U -> U: (('Time', 'bottom_top', 'south_north', 'west_east',)) (float32)
wind_.* -> wind_east: (('Time', 'bottom_top', 'south_north', 'west_east',)) (float32)
wind_.* -> wind_north: (('Time', 'bottom_top', 'south_north', 'west_east',)) (float32)
```

### process-pipeline

```bash
wrf-ensembly EXPERIMENT_PATH postprocess process-pipeline [--cycle CYCLE] [--jobs JOBS]
```

Applies the configured data processor pipeline to output files. This command replaces the legacy `apply-scripts` approach with a more efficient plugin-based system that processes data in memory using a configurable pipeline of `DataProcessor` instances.

**Options:**
- `--cycle CYCLE` - Cycle to process (uses current cycle if not specified)
- `--jobs JOBS` - Number of files to process in parallel (default: determined automatically)

The pipeline processes all `wrfout` files in both the analysis and forecast directories for the specified cycle, creating `*_post` files that contain the processed data.

### statistics

```bash
wrf-ensembly EXPERIMENT_PATH postprocess statistics [--cycle CYCLE] [--jobs JOBS]
```

Calculates ensemble mean and standard deviation from the forecast/analysis files for the given cycle. This function reads the `*_post` files created by the `process-pipeline` command and computes ensemble statistics across all members.

**Options:**
- `--cycle CYCLE` - Cycle to compute statistics for (uses current cycle if not specified)
- `--jobs JOBS` - Number of files to process in parallel (default: determined automatically)

**Output files:**
- `*_mean` - Ensemble mean files
- `*_sd` - Ensemble standard deviation files

For single-member experiments, the command simply copies the files without computing statistics.

### concatenate

```bash
wrf-ensembly EXPERIMENT_PATH postprocess concatenate [--cycle CYCLE] [--jobs JOBS]
```

Concatenates all output files (mean and standard deviation) into consolidated files for each cycle. Uses the `*_mean` and `*_sd` files created by the `statistics` command and combines them temporally into single NetCDF files.

**Options:**
- `--cycle CYCLE` - Cycle to concatenate (uses current cycle if not specified)
- `--jobs JOBS` - Number of NCO commands to execute in parallel (default: 4)

**Output files:**
- `forecast_mean_cycle_XXX.nc` - Concatenated forecast ensemble mean
- `forecast_sd_cycle_XXX.nc` - Concatenated forecast ensemble standard deviation
- `analysis_mean_cycle_XXX.nc` - Concatenated analysis ensemble mean
- `analysis_sd_cycle_XXX.nc` - Concatenated analysis ensemble standard deviation

If `keep_per_member` is enabled, also creates:
- `forecast_member_XX_cycle_XXX.nc` - Individual member forecasts

The concatenation process applies compression and quantization filters as specified in the configuration.

### clean

```bash
wrf-ensembly EXPERIMENT_PATH postprocess clean [--cycle CYCLE] [--remove-wrfout]
```

Cleans up the scratch directory for the given cycle to save disk space. This command should be run after completing the other postprocessing steps.

**Options:**
- `--cycle CYCLE` - Cycle to clean (uses current cycle if not specified)
- `--remove-wrfout` - Remove raw wrfout files (default: true)

## Data Processor Pipeline

The heart of the postprocessing system is the data processor pipeline, which applies a series of transformations to each WRF output file. The pipeline is highly configurable and extensible through a plugin system.

### Built-in Processors

#### XWRFPostProcessor

The `XWRFPostProcessor` is automatically applied as the first step in every pipeline. It performs essential transformations including:

- **Air density computation** - Calculates air density from model variables
- **xWRF postprocessing** - Applies destaggering and diagnostic computations
- **Coordinate standardization** - Renames dimensions and coordinates to standard names
- **Variable filtering** - Removes unused variables and filters based on `variables_to_keep`
- **Metadata cleanup** - Fixes coordinate attributes and removes serialization artifacts

#### ScriptProcessor

The `ScriptProcessor` allows integration of external scripts for backward compatibility. Scripts must accept `{in}` and `{out}` placeholders for input and output files.

**Configuration example:**
```toml
[[postprocess.processors]]
processor = "script"
params = { script = "python enhance_data.py {in} {out}" }
```

The script can also use additional placeholders:
- `{d}` - Member number (0-based)
- `{c}` - Cycle number

### Custom Processors

You can create custom processors by implementing the `DataProcessor` abstract base class:

```python
from wrf_ensembly.processors import DataProcessor, ProcessingContext
import xarray as xr

class MyCustomProcessor(DataProcessor):
    def __init__(self, custom_param="default", **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param

    def process(self, ds: xr.Dataset, context: ProcessingContext) -> xr.Dataset:
        """Apply your custom processing logic here"""
        # Example: Add a new variable
        ds['custom_variable'] = ds['U'] * 2.0
        ds['custom_variable'].attrs['description'] = 'Custom variable'

        # Access context information
        if context.member == 0:
            # Special processing for first member
            pass

        return ds

    @property
    def name(self) -> str:
        return "My Custom Processor"
```

### Processor Configuration

Processors are configured in the `postprocess.processors` section of the configuration file:

```toml
# Built-in script processor
[[postprocess.processors]]
processor = "script"
params = { script = "python add_variable.py {in} {out}" }

# External processor from module
[[postprocess.processors]]
processor = "my_package.processors:CustomProcessor"
params = { custom_param = "value" }

# External processor from file
[[postprocess.processors]]
processor = "/path/to/custom_processor.py:MyProcessor"
params = { param1 = "value1", param2 = 42 }
```

### Processor Loading

Processors can be loaded from several sources:

1. **Built-in processors** - Use names like `"script"`
2. **Module paths** - Format: `"module.path:ClassName"`
3. **File paths** - Format: `"/path/to/file.py:ClassName"`
4. **File with auto-discovery** - Format: `"/path/to/file.py"` (uses filename as class name)

### Processing Context

The `ProcessingContext` object provides processors with metadata about the current processing operation:

```python
@dataclass
class ProcessingContext:
    member: int          # Ensemble member number (0-based)
    cycle: int           # Cycle number
    input_file: Path     # Path to input file
    output_file: Path    # Path to output file
    config: Config       # Full experiment configuration
```

This context allows processors to adapt their behavior based on the member, cycle, or other experiment parameters.

## Best Practices

### Variable Filtering

Use regular expressions in `variables_to_keep` to efficiently filter variables:

```toml
variables_to_keep = [
    "DUST_\\d",           # All dust species (DUST_1, DUST_2, etc.)
    "U|V|W",              # Wind components
    "wind_.*",            # Derived wind variables
    "T|P|PH|MU",          # Thermodynamic variables
    "QVAPOR",             # Water vapor
    "longitude|latitude", # Coordinates
    "t"                   # Time
]
```

Keep only the variables you need for your science to reduce file sizes and processing time. Avoid using overly broad patterns that may match unintended variables.
You can check which variables will be kept using the `print-variables-to-keep` command, which prints the variables that match the regex patterns defined in `variables_to_keep`.

### Compression and Quantization

The default compression settings provide a good balance between file size and processing time:

- **Compression filters**: `"shf|zst,3"` applies shuffle and Zstandard compression
- **Quantization**: `"default=3#Z.*=6#X.*=6"` keeps 3 significant digits for most variables, 6 for height/horizontal coordinates

These filters are passed to nco's commands when doing the concatenation step. Read NCO's documentation for more details on the available compression and quantization options.
If your netCDF libraries are not built to handle Zstandard compression, you will not see any errors but the files will not be compressed. Always double check your netCDF output file sizes and check them with `ncdump -hs` to see which compression is applied. If you can't seem to get Zstandard to work, just use `deflate` instead.

### Resource Allocation

For large ensembles, tune the core allocation:

```toml
processor_cores = 24    # Parallel file processing
statistics_cores = 8  # Parallel statistics computation
concatenate_cores = 8  # Parallel NCO operations
```

Your main bottleneck here will likely be RAM usage, so ensure that you have enough RAM to run all these things in parallel. The `processor_cores` setting controls how many files are processed in parallel, while `statistics_cores` and `concatenate_cores` control parallelism for statistics computation and NCO concatenation, respectively.

### Error Handling

The pipeline stops on any processor error. To debug issues:

1. Check log files in the experiment's `logs/` directory
2. Test processors individually with small datasets
3. Use the `print-variables-to-keep` command to verify variable filtering
4. Examine intermediate `*_post` files for processing artifacts

## Integration with SLURM

The postprocessing commands integrate seamlessly with SLURM through the `slurm postprocess` command, which runs all postprocessing steps as a single job. The SLURM configuration uses the `directives_postprocess` settings for resource allocation.

For more information on SLURM integration, see the [SLURM documentation](usage.md#slurm).
