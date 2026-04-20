# Observations

WRF-Ensembly provides an observation system for handling observational data in ensemble assimilation experiments. The system converts observations from various instruments into a standardized format, processes them spatially and temporally for your experiment domain, and integrates with DART for data assimilation.

For configuration options, see the [Configuration](configuration.md#observations) reference.

## Overview

The observation system consists of several key components:

- **Standardized Format** - A unified parquet-based observation format; all raw data is converted to this target.
- **Instrument & Quantity Registry** - Definitions for supported instruments, physical quantities, and observation operators (`definitions.py`).
- **Converters** - Instrument-specific modules that transform raw observation data to the standard format.
- **Database Management** - DuckDB-based storage and querying of observations within experiments.
- **Spatial/Temporal Processing** - Tools for trimming observations to experiment domains and time windows.
- **Density Reduction** - Superobbing, temporal binning, and stride thinning to reduce observation density.
- **DART Integration** - Conversion to DART's obs_seq format for data assimilation.
- **Validation** - Interpolation routines to compare model output against observations (O-B statistics).
- **Plotting** - Observation-only and observation-vs-model plotting routines.

While DART already includes an observation management system, WRF-Ensembly includes its own to facilitate extra tools, such as validation and plotting. This also makes it easier to write converters in Python, where even obscure observation formats usually have library support.

Because WRF-Ensembly supports plotting and model comparisons, the aim is to support not only the observations that will be assimilated but all observations you will use in an experiment. For example, you might assimilate Instrument A while using Instrument B for validation — both should be added to the experiment.

## Observation Data Format

The WRF-Ensembly observation format is a standardized schema stored as parquet files. Each observation record contains:

| Field | Type | Description |
|-------|------|-------------|
| `instrument` | string | Name of the observing instrument (e.g., `AERONET`, `MODIS`) |
| `quantity` | string | Physical quantity being observed (e.g., `AOD_550nm`) |
| `time` | timestamp | UTC time of the observation |
| `longitude` | float | Longitude in degrees (-180 to 180) |
| `latitude` | float | Latitude in degrees (-90 to 90) |
| `z` | float | Vertical coordinate value |
| `z_type` | string | Type of vertical coordinate (see below) |
| `value` | float | The observed value (can be NaN for missing data) |
| `value_uncertainty` | float | Observation uncertainty/error estimate |
| `qc_flag` | int | Quality control flag (see [Quality Control](#quality-control)) |
| `orig_coords` | struct | Original coordinate information (indices, shape, names) |
| `orig_filename` | string | Name of the original data file |
| `metadata` | JSON string | Additional metadata (e.g., Aeolus azimuth angle) |

### Vertical Coordinate Types

| Value | Description | Units |
|-------|-------------|-------|
| `surface` | Surface observations | N/A |
| `pressure` | Pressure levels | hPa |
| `height` | Height above surface | meters |
| `model_level` | Model vertical levels | level number |
| `columnar` | Column-integrated values | N/A |

For `instrument` and `quantity`, the convention is: everything instrument- and retrieval-specific is encoded in `instrument`, while `quantity` is instrument-agnostic. For example, `AEOLUS_L2A_MLE` and `AEOLUS_L2A_SCA` are two "instruments" representing different retrieval algorithms, but both use `LIDAR_EXTINCTION_355nm` as the quantity.

### Array Reconstruction

While DA requires tabular data with one observation per row, it is often useful to reconstruct the original array structure (e.g., a vertical profile over time, or a satellite image). To facilitate this, the `orig_coords` field stores:

- `indices`: The observation's position in the original array.
- `shape`: The shape of the original array.
- `names`: The names of the original dimensions.

The function `wrf_ensembly.observations.definitions.reshape_to_native()` reconstructs these arrays from a filtered DataFrame.

### Internal Database Fields

After observations are added to the experiment database, a few additional fields are used in the DuckDB file:

| Field | Type | Description |
|-------|------|-------------|
| `x`, `y` | float | Projected coordinates in the WRF grid coordinate system |
| `model_forecast` | double | Model forecast equivalent value at this observation |
| `model_analysis` | double | Model analysis equivalent value at this observation |
| `model_forecast_spread` | double | Ensemble spread of the model forecast at this observation |
| `model_analysis_spread` | double | Ensemble spread of the model analysis at this observation |

The schema is kept stable so you can query `observations.duckdb` directly if needed.

## Instruments, Quantities, and Observation Operators

The `wrf_ensembly/observations/definitions.py` module is the central registry for instruments and quantities. It defines:

- **`INSTRUMENT_REGISTRY`** — maps instrument name strings to `InstrumentSpec` (label, geometry type, axis information for plotting).
- **`QUANTITY_REGISTRY`** — maps quantity name strings to `QuantitySpec` (label, units, colormap, model mapping or operator, DART quantity name).

### Supported Instruments

| Instrument | Description | Geometry |
|------------|-------------|----------|
| `AEOLUS_L2A_MLE` | AEOLUS L2A MLE extinction retrieval | Profile curtain |
| `AEOLUS_L2A_SCA` | AEOLUS L2A SCA extinction retrieval | Profile curtain |
| `AEOLUS_L2A_AEL_PRO` | AEOLUS L2A AEL-PRO extinction retrieval | Profile curtain |
| `AEOLUS_L2B_RAYLEIGH` | AEOLUS L2B HLOS Wind (Rayleigh channel) | Wind results curtain |
| `AEOLUS_L2B_MIE` | AEOLUS L2B HLOS Wind (Mie channel) | Wind results curtain |
| `MSG_SEVIRI` | Meteosat SEVIRI brightness temperatures | 2D map swath |

Additional instruments used by converters but not yet in `INSTRUMENT_REGISTRY` (plotting/operator support is limited): `AERONET`, `MODIS`, `VIIRS`, `EarthCARE_ATLID_EBD`, `REMOTAP_SPEXONE`.

### Supported Quantities

| Quantity | Label | Units | WRF Equivalent | DART Quantity |
|----------|-------|-------|----------------|---------------|
| `LIDAR_EXTINCTION_355nm` | Lidar Extinction @ 355nm | 1/m | `EXT355` | `LIDAR_EXTINCTION_355nm` |
| `HLOS_WIND` | HLOS Wind | m/s | operator (see below) | `SAT_HLOS_WIND` |
| `BT_WV62` | Brightness Temp WV 6.2 µm | K | `WV62` | — |
| `BT_WV73` | Brightness Temp WV 7.3 µm | K | `WV73` | — |
| `BT_IR87` | Brightness Temp IR 8.7 µm | K | `IR87` | — |
| `BT_IR108` | Brightness Temp IR 10.8 µm | K | `IR108` | — |
| `BT_IR120` | Brightness Temp IR 12.0 µm | K | `IR120` | — |
| `AOD_355nm` | AOD @ 355nm | — | `AOD_355` | — |
| `AOD_500nm` | AOD @ 500nm | — | `AOD_500` | `AIRSENSE_AOD` |
| `AOD_550nm` | AOD @ 550nm | — | `AOD_550` | `AIRSENSE_AOD` |
| `AOD_1064nm` | AOD @ 1064nm | — | `AOD_1064` | — |

### Observation Operators

`QuantitySpec` supports two ways to map observations to model fields:

- **`model_equivalent`** — A direct WRF variable name. The system interpolates that variable to the observation location/time and uses it as-is. Example: `AOD_550nm` maps to WRF's `AOD_550`.

- **`operator`** — An `OperatorSpec` for non-trivial transformations. The operator specifies:
  - `func`: A Python callable `(model_fields: dict, metadata: dict) -> ndarray` that returns model-equivalent values.
  - `required_model_fields`: `ModelFieldSpec` list specifying which WRF variables to interpolate (with `dims=2` for 2D or `dims=3` for 3D vertical interpolation).
  - `required_metadata`: Keys to extract from the observation `metadata` JSON column.

  The only current complex operator is `HLOS_WIND`, which projects the model's `wind_east` and `wind_north` (3D) onto the satellite's HLOS direction using the `azimuth` metadata field from Aeolus observations.

## Quality Control

QC flags control which observations are used for assimilation and validation:

| QC Flag | Meaning | Behavior |
|---------|---------|---------|
| `0` | Good observation | Used in assimilation and validation |
| `> 0` | Instrument or data quality issue | Excluded from assimilation; present in database |
| `-1` | Validation hold-out (stride thinning) | Not assimilated; used in validation |
| `99` | Zero or negative variance | Set automatically; excluded from assimilation |

Positive QC flag values come from instrument-native QA flags and are set by the converters. Negative values are WRF-Ensembly internal flags.

## Workflow Overview

1. **Convert Raw Data** — Transform instrument-specific files to the standardized parquet format using `wrf-ensembly-obs convert`.
2. **Add to Experiment** — Import observations into the experiment DuckDB with spatial/temporal trimming. Density reduction (superobs, temporal binning, thinning) is applied here.
3. **Prepare Cycles** — Extract observations for each assimilation cycle's time window and convert to DART obs_seq format.
4. **Assimilate** — DART reads the obs_seq files during `ensemble filter`.
5. **Validate (optional)** — Interpolate model output to observation locations, then analyze O-B statistics.

## Converting Observations

Converters are accessed through the `wrf-ensembly-obs convert` subcommand. This CLI is independent of `wrf-ensembly` and works without an experiment context.

### Available Converters

| Converter | Instrument | Quantity | Input Format |
|-----------|------------|----------|--------------|
| `aeronet` | `AERONET` | `AOD_{wavelength}nm` | `.lev20` tabular files |
| `remotap-spexone` | `REMOTAP_SPEXONE` | AOD | NetCDF files |
| `aeolus-l2a` | `AEOLUS_L2A_MLE`, `AEOLUS_L2A_SCA`, `AEOLUS_L2A_AEL_PRO` | `LIDAR_EXTINCTION_355nm` | AEOLUS DBL L2A files |
| `aeolus-l2b` | `AEOLUS_L2B_MIE`, `AEOLUS_L2B_RAYLEIGH` | `HLOS_WIND` | AEOLUS DBL L2B files |
| `earthcare-atl-ebd` | `EarthCARE_ATLID_EBD` | `LIDAR_EXTINCTION_355nm` | EarthCARE ATLID EBD files |
| `modis` | `MODIS` | `AOD_550nm` | MODIS AOD HDF4 files |
| `viirs` | `VIIRS` | `AOD_550nm` | VIIRS AOD NetCDF files |
| `msg-seviri` | `MSG_SEVIRI` | `BT_WV62`, `BT_WV73`, `BT_IR87`, `BT_IR108`, `BT_IR120` | SEVIRI native format (via satpy) |

### Using Converters

```bash
# Convert AERONET data (specify quantities with --quantity)
wrf-ensembly-obs convert aeronet input_files/*.lev20 output.parquet \
  --quantity AOD_500nm --quantity AOD_550nm

# Convert AEOLUS L2A data
wrf-ensembly-obs convert aeolus-l2a input.DBL output.parquet

# Convert AEOLUS L2B HLOS wind data
wrf-ensembly-obs convert aeolus-l2b input.DBL output.parquet

# Convert MSG SEVIRI (requires a WRF grid file for reprojection)
wrf-ensembly-obs convert msg-seviri input.nat wrfinput_d01 output.parquet

# Get help for a specific converter
wrf-ensembly-obs convert aeronet --help
```

### Creating New Converters

To add support for a new instrument:

1. Create the converter module in `wrf_ensembly/observations/converters/your_instrument.py`.
2. Implement the conversion function that returns a pandas DataFrame with the required schema.
3. Add a Click command and register it in `converters/__init__.py` and `observations/cli.py`.
4. Optionally add `InstrumentSpec` and `QuantitySpec` entries to `definitions.py` for plotting and validation support.

Use `template.py` as a starting point. Key requirements:
- All required columns must be present and correctly typed.
- Use `obs_io.validate_schema()` to verify compliance.
- Handle `orig_coords` to preserve array structure information.
- Set appropriate `qc_flag` values based on instrument QA flags.

## Experiment Integration

### Adding Observations to Experiments

Once observations are converted, add them to the experiment:

```bash
# Add individual files or directories
wrf-ensembly $EXP_PATH obs add /path/to/observations/*.parquet

# Use parallel processing for large datasets
wrf-ensembly $EXP_PATH obs add /path/to/observations/*.parquet --jobs 4
```

The `add` command performs:
1. **Spatial Trimming** — Removes observations outside the WRF domain. Observations whose entire profile/row is outside are dropped; observations at the boundary of an array group have their value set to NaN (to preserve array reconstructability).
2. **Temporal Trimming** — Filters to the experiment's cycling time range.
3. **Density Reduction** — Applies superobs, temporal binning, or stride thinning if configured.
4. **Database Storage** — Inserts trimmed observations into the experiment's DuckDB database at `obs/observations.duckdb`.

### Observation Database

Each experiment maintains a DuckDB database. You can interact with it via CLI commands or directly from Python:

```bash
# Show summary of available observations and files
wrf-ensembly $EXP_PATH obs show

# Remove all observations from a specific file
wrf-ensembly $EXP_PATH obs delete 'original_file.nc'
```

## Density Reduction

Observations are often denser than the model grid, so WRF-Ensembly supports three mutually exclusive (per instrument-quantity pair) methods to reduce density. These are configured in `config.toml` and applied during `obs add`.

### Spatial Superobbing (`superobs`)

Groups observations into spatial bins defined by their original array dimensions, then averages each bin into a single superobservation. Uncertainty is reduced by sqrt(n) for the instrument error component and augmented by the within-bin standard deviation as representativeness error.

```toml
[observations.superobs."AEOLUS_L2A_MLE.LIDAR_EXTINCTION_355nm"]
hoz_bin_sizes = {profile = 5}       # Bin every 5 profiles together
vert_bin_sizes = {height_bin = 2}   # Bin every 2 height levels together
```

### Temporal Binning (`temporal_binning`)

Groups observations into fixed-width UTC time windows, producing one superob per window. Cannot be used together with `superobs` for the same instrument-quantity pair.

```toml
[observations.temporal_binning."AERONET.AOD_550nm"]
bin_minutes = 60        # 60-minute windows
offset_minutes = -30    # Shift bins to be centered on full hours (:30 to :30)
```

### Stride Thinning (`thinning`)

Keeps every N-th good-QC observation for DA; the others are marked `qc_flag = -1` (validation hold-out) and remain in the database. Applied after superobbing.

```toml
[observations.thinning."AEOLUS_L2B_MIE.HLOS_WIND"]
keep_every_n = 3    # Keep 1 in 3 observations for assimilation
```

## Cycle Preparation

Before data assimilation, observations must be prepared for each cycle:

```bash
# Prepare observations for all cycles
wrf-ensembly $EXP_PATH obs prepare-cycles

# Prepare for a specific cycle
wrf-ensembly $EXP_PATH obs prepare-cycles --cycle 0

# Use parallel jobs for DART conversion
wrf-ensembly $EXP_PATH obs prepare-cycles --jobs 4

# Skip DART conversion (write parquet files only, for inspection)
wrf-ensembly $EXP_PATH obs prepare-cycles --skip-dart
```

This process:
1. Queries the DuckDB for observations within the cycle's assimilation window (controlled by `assimilation.half_window_length_minutes`).
2. Filters to instruments in `observations.instruments_to_assimilate` (or all instruments if unset).
3. Applies the `error_inflation_factor` if configured.
4. Writes `obs/cycle_NNN.parquet` for inspection.
5. Converts observations with `qc_flag = 0` to `obs/cycle_NNN.obs_seq` using the DART converter.

### Assimilation Window

```toml
[assimilation]
half_window_length_minutes = 30  # ±30 minutes around cycle end time
```

## DART Integration

### Observation Type Mapping

Most WRF-Ensembly quantities map directly to DART by their quantity name (e.g., `LIDAR_EXTINCTION_355nm`, `SAT_HLOS_WIND`). A small lookup table handles cases where the names differ:

```python
OBS_TYPE_TABLE = {
    "AOD_500nm": "AIRSENSE_AOD",
    "AOD_550nm": "AIRSENSE_AOD",
}
```

### Building the DART Converter

The system requires a custom DART converter located at `$DART_ROOT/observations/obs_converters/wrf_ensembly/`. Build it before use:

```bash
cd $DART_ROOT/observations/obs_converters/wrf_ensembly/work
./quickbuild.sh
```

The converter reads observations from stdin as CSV and writes a DART obs_seq file. Longitudes are converted to [0, 360) as required by DART.

## Configuration

```toml
[observations]
instruments_to_assimilate = ["AERONET", "MODIS"]  # null = use all instruments
boundary_width = 0          # Grid points to exclude near domain boundaries
boundary_error_factor = 2.5 # Error inflation factor near boundaries
boundary_error_width = 1.0  # Width (grid points) of boundary inflation zone

[observations.error_inflation_factor]
"AERONET.AOD_550nm" = 2.0   # Inflate errors for this pair by 2x

[observations.superobs."AEOLUS_L2A_MLE.LIDAR_EXTINCTION_355nm"]
hoz_bin_sizes = {profile = 5}
vert_bin_sizes = {height_bin = 2}

[observations.temporal_binning."AERONET.AOD_550nm"]
bin_minutes = 60
offset_minutes = 0

[observations.thinning."AEOLUS_L2B_MIE.HLOS_WIND"]
keep_every_n = 3
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `instruments_to_assimilate` | `[string]` or null | null | Instruments to include in assimilation. Null uses all. |
| `boundary_width` | float | 0 | Grid points to exclude near domain boundaries |
| `boundary_error_factor` | float | 2.5 | Error inflation multiplier near boundaries |
| `boundary_error_width` | float | 1.0 | Width (grid points) of boundary error inflation zone |
| `error_inflation_factor` | `{str: float}` | `{}` | Per-instrument.quantity error multiplier |
| `superobs` | `{str: SuperObsConfig}` | `{}` | Spatial superobbing per instrument.quantity |
| `temporal_binning` | `{str: TemporalBinConfig}` | `{}` | Temporal binning per instrument.quantity |
| `thinning` | `{str: ThinningConfig}` | `{}` | Stride thinning per instrument.quantity |

## Validation

WRF-Ensembly includes tools to compare model output against observations (first departure / O-B analysis). These commands are under the `validation` group.

### Interpolating the Model

```bash
wrf-ensembly $EXP_PATH validation interpolate-model
```

This reads the processed wrfout files (forecast mean and analysis mean), interpolates them to each observation's location, time, and vertical level, and stores the results in the `model_forecast` and `model_analysis` columns in DuckDB. Requires completed `postprocess` runs.

### First Departures Analysis

```bash
# Analyze all instrument-quantity pairs with model values
wrf-ensembly $EXP_PATH validation analyze-first-departures

# Analyze specific pairs
wrf-ensembly $EXP_PATH validation analyze-first-departures \
  --instrument-quantity MODIS.AOD_550nm \
  --instrument-quantity AERONET.AOD_550nm

# Restrict to a time window
wrf-ensembly $EXP_PATH validation analyze-first-departures \
  --start-time "2023-08-01T00:00:00" \
  --end-time "2023-08-31T23:59:59"
```

For each instrument-quantity pair, the analysis generates (saved to `data/validation/first_departures/{instrument}/{quantity}/`):
- **Statistics file** — Bias, standard deviation, RMSE.
- **Histogram** — Distribution of O-B departures.
- **Time series** — Mean and std over time.
- **Spatial maps** — Gridded bias and std.
- **Regime plots** — If regime configurations are defined in `config.toml`.

Pairs to analyze can be configured in `config.toml` or specified via `--instrument-quantity`. If neither is set, all pairs with model values in the database are analyzed.

```toml
[validation.first_departures]
instrument_quantity_pairs = ["MODIS.AOD_550nm", "AERONET.AOD_500nm"]

[[validation.first_departures.regimes]]
instrument = "MODIS"
quantity = "AOD_550nm"
bins = [0.0, 0.2, 0.5, 1.0, .inf]
labels = ["very_low", "low", "moderate", "high"]
spatial_resolution = 1.0
```

## Command Reference

### Experiment-Level Commands (`wrf-ensembly $EXP_PATH obs`)

| Command | Description |
|---------|-------------|
| `add FILES [--jobs N]` | Add parquet observation files to the experiment database |
| `show` | Print tables of available quantities and files |
| `delete FILENAME` | Remove all observations from a specific original file |
| `prepare-cycles [--cycle N] [--jobs N] [--skip-dart]` | Prepare obs_seq files for each cycle |
| `cycle-summary` | Print observation counts per cycle |
| `cycle-info CYCLE [--as-json]` | Detailed per-file stats for a specific cycle |
| `plot-cycle-locations CYCLE` | Plot observation locations for a cycle on a map |
| `plot-compare-obs-to-grid CYCLE [--center-lat] [--center-lon] [--window-size] [--instrument] [--quantity] [--keep-only-good-qc]` | Plot observations overlaid on the WRF grid (useful for tuning superobbing) |
| `plot FILENAME [--dpi] [--vmin] [--vmax] [--ylim] [--qc] [--no-robust] [--with-model]` | Plot observations from a specific file; `--with-model` produces O-B panels |

### Validation Commands (`wrf-ensembly $EXP_PATH validation`)

| Command | Description |
|---------|-------------|
| `interpolate-model` | Interpolate model forecast/analysis to observation locations |
| `analyze-first-departures [--instrument-quantity IQ] [--start-time] [--end-time]` | Compute and plot O-B statistics |

### Standalone File Operations (`wrf-ensembly-obs convert`)

| Command | Description |
|---------|-------------|
| `aeronet INPUT OUTPUT [--quantity Q]` | Convert AERONET `.lev20` files |
| `remotap-spexone INPUT OUTPUT` | Convert REMOTAP SPEXONE NetCDF files |
| `aeolus-l2a INPUT OUTPUT [--mle/--no-mle] [--sca/--no-sca] [--ael-pro/--no-ael-pro]` | Convert AEOLUS L2A files |
| `aeolus-l2b INPUT OUTPUT` | Convert AEOLUS L2B HLOS wind files |
| `earthcare-atl-ebd INPUT OUTPUT` | Convert EarthCARE ATLID EBD files |
| `modis INPUT OUTPUT` | Convert MODIS AOD HDF4 files |
| `viirs INPUT OUTPUT` | Convert VIIRS AOD NetCDF files |
| `msg-seviri INPUT WRFINPUT OUTPUT` | Convert MSG SEVIRI native files (requires WRF input for grid) |

## Adding a New Observation Kind

This section is a step-by-step guide for adding a completely new instrument and/or quantity. The steps are ordered roughly by dependency — each step unlocks the next.

### Step 1 — Register the Instrument and Quantity (`definitions.py`)

Add entries to `INSTRUMENT_REGISTRY` and `QUANTITY_REGISTRY` in `wrf_ensembly/observations/definitions.py`. This is required for plotting and model-comparison support. If you only need a converter (no plotting, no validation), you can skip this step, but it is good practice to do it upfront.

**Add the instrument:**
```python
"MY_INSTRUMENT": InstrumentSpec(
    label="My Instrument Description",
    geometry=Geometry.MAP_SWATH,          # or PROFILE_CURTAIN, TIMESERIES, ...
    x=AxisSpec(dim="x", label="X", coord="longitude"),
    y=AxisSpec(dim="y", label="Y", coord="latitude"),
),
```

**Add the quantity with a direct WRF mapping (simplest case):**
```python
"MY_QUANTITY": QuantitySpec(
    label="My Quantity [unit]",
    units="unit",
    cmap="viridis",
    vmin=0,
    vmax=1,
    model_equivalent="WRF_VARIABLE_NAME",  # exact WRF field name
    dart_quantity="DART_QUANTITY_NAME",     # DART obs type string
),
```

**Add the quantity with a complex observation operator (multi-field case):**

If the observation cannot be directly mapped to a single WRF variable (e.g., a projected wind component), define an operator instead of `model_equivalent`:

```python
# In wrf_ensembly/observations/operators/my_operator.py:
def my_operator(
    model_fields: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
) -> np.ndarray:
    # model_fields keys match the `name` of each ModelFieldSpec
    # metadata keys match required_metadata
    return model_fields["field_a"] * metadata["scale"]

# In definitions.py:
"MY_QUANTITY": QuantitySpec(
    label="...",
    units="...",
    operator=OperatorSpec(
        func=my_operator,
        required_model_fields=(
            ModelFieldSpec("field_a", dims=3),  # dims=3 for vertical interpolation
            ModelFieldSpec("field_b", dims=2),  # dims=2 for surface/column fields
        ),
        required_metadata=("scale",),           # keys from the obs metadata JSON
        description="Human-readable description",
    ),
    dart_quantity="DART_QUANTITY_NAME",
),
```

### Step 2 — Write the Converter

Create `wrf_ensembly/observations/converters/my_instrument.py`. Use `template.py` as a starting point. The function must return a pandas DataFrame with these columns:

| Column | Notes |
|--------|-------|
| `instrument` | Must match the key in `INSTRUMENT_REGISTRY` |
| `quantity` | Must match the key in `QUANTITY_REGISTRY` |
| `time` | `pd.Timestamp`, timezone-aware UTC |
| `longitude` | float, degrees (-180 to 180) |
| `latitude` | float, degrees (-90 to 90) |
| `z` | float, vertical coordinate value |
| `z_type` | one of `surface`, `pressure`, `height`, `model_level`, `columnar` |
| `value` | float, may be NaN |
| `value_uncertainty` | float, may be NaN if unknown |
| `qc_flag` | int; `0` = good, positive = instrument QA issues |
| `orig_coords` | dict with keys `indices`, `shape`, `names` |
| `orig_filename` | `input_path.name` (not the full path) |
| `metadata` | JSON string for any extra fields needed by the operator |

Call `obs_io.validate_schema(df)` before returning to catch missing or mistyped columns.

For the `orig_coords` field, store the observation's position in the original data array so that `reshape_to_native()` can reconstruct it later. For a 2D satellite image of shape `(H, W)`:

```python
orig_coords = [
    {"indices": [int(i), int(j)], "shape": [H, W], "names": ["y", "x"]}
    for i, j in zip(row_indices, col_indices)
]
```

For 1D timeseries data or when reconstruction is not meaningful, use a single-element shape:
```python
orig_coords = [
    {"indices": [int(k)], "shape": [N], "names": ["obs"]}
    for k in range(N)
]
```

### Step 3 — Register the Converter CLI

Add a Click command to the same file, then register it in two places:

**In `wrf_ensembly/observations/converters/__init__.py`:**
```python
from .my_instrument import my_instrument as my_instrument_cli

__all__ = [..., "my_instrument_cli"]
```

**In `wrf_ensembly/observations/cli.py`:**
```python
from wrf_ensembly.observations.converters import ..., my_instrument_cli

# inside the file, add to the convert group:
convert_group.add_command(my_instrument_cli)
```

After this step, `wrf-ensembly-obs convert my-instrument --help` should work.

### Step 4 — Handle the DART Type (if needed)

If your quantity's `dart_quantity` string (from `QuantitySpec`) differs from what DART calls it, add a mapping to `OBS_TYPE_TABLE` in `wrf_ensembly/observations/dart.py`:

```python
OBS_TYPE_TABLE = {
    "AOD_500nm": "AIRSENSE_AOD",
    "AOD_550nm": "AIRSENSE_AOD",
    "MY_QUANTITY": "DART_TYPE_NAME",  # add here if names differ
}
```

If `dart_quantity` already matches the DART type name exactly (as with `LIDAR_EXTINCTION_355nm` and `SAT_HLOS_WIND`), no entry is needed — the quantity name is passed through directly.

You also need to ensure the DART quantity is registered in the DART source and the `wrf_ensembly` converter is compiled with that type. This is outside the Python codebase but is required before DART can ingest the observations.

### Step 5 — Test the Conversion

```bash
# Convert a test file
wrf-ensembly-obs convert my-instrument test_file.nc output.parquet

# Inspect the result
wrf-ensembly-obs dump-info output.parquet

# Add to an experiment and verify it appears in the summary
wrf-ensembly $EXP_PATH obs add output.parquet
wrf-ensembly $EXP_PATH obs show
```

### Step 6 — Enable for Assimilation

Add the instrument to `instruments_to_assimilate` in `config.toml`:
```toml
[observations]
instruments_to_assimilate = ["MY_INSTRUMENT", "AERONET"]
```

Then prepare cycles and verify the observations appear:
```bash
wrf-ensembly $EXP_PATH obs prepare-cycles --skip-dart --cycle 0
wrf-ensembly $EXP_PATH obs cycle-info 0
```

### Summary Checklist

- [ ] `InstrumentSpec` added to `INSTRUMENT_REGISTRY` in `definitions.py`
- [ ] `QuantitySpec` added to `QUANTITY_REGISTRY` in `definitions.py` (with `model_equivalent` or `operator`)
- [ ] Operator function written in `observations/operators/` (if using `OperatorSpec`)
- [ ] Converter written in `observations/converters/my_instrument.py`
- [ ] Converter registered in `converters/__init__.py`
- [ ] Converter registered in `observations/cli.py`
- [ ] `OBS_TYPE_TABLE` entry added in `dart.py` (if DART type name differs)
- [ ] DART quantity registered and converter recompiled (outside Python)
- [ ] Instrument added to `instruments_to_assimilate` in `config.toml`

## Advanced Usage

### Custom Database Queries

The observation database supports complex SQL queries:

```python
with experiment.obs._get_duckdb(read_only=True) as con:
    results = con.execute("""
        SELECT instrument, quantity, COUNT(*) as count,
               AVG(value) as mean_value,
               MIN(time) as start_time,
               MAX(time) as end_time
        FROM observations
        WHERE qc_flag = 0
          AND time BETWEEN '2023-01-01' AND '2023-01-31'
        GROUP BY instrument, quantity
        ORDER BY count DESC
    """).fetchdf()
```

### Array Reconstruction

For observations from gridded data, reconstruct the original array using `reshape_to_native()`:

```python
from wrf_ensembly.observations.definitions import reshape_to_native

# Filter to a single instrument, quantity, and original file
subset = df[
    (df["instrument"] == "AEOLUS_L2A_MLE")
    & (df["quantity"] == "LIDAR_EXTINCTION_355nm")
    & (df["orig_filename"] == "AE_TEST_ALD_U_N_2A_...DBL")
]

arr, shape, dim_names = reshape_to_native(subset, field="value")
# arr is a numpy array with the original (profile, height_bin) shape
```

## Next Steps

- [Running the ensemble](usage.md#ensemble-management) — Execute your data assimilation experiment.
- [Postprocessing](postprocess.md) — Process model output.
- [Configuration](configuration.md) — Fine-tune observation settings and other experiment parameters.
