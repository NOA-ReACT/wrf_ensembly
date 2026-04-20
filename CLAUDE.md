# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

WRF-Ensembly is a Python toolkit for conducting Ensemble Data Assimilation experiments using WRF/WRF-CHEM and NCAR DART. It acts as "glue" between WRF and DART, managing the cycling assimilation workflow through a series of CLI commands. Everything is configured through a single TOML file per experiment.

## Development Commands

### Environment Setup
The project uses a `.venv` virtual environment. Always activate it before running commands:
```bash
source .venv/bin/activate
```

### Testing
There are limited tests in the codebase. You can't rely on them for making sure things work.

```bash
# Run all tests
pytest

# Run a specific test
pytest wrf_ensembly/test_utils.py::test_int_to_letter_numeral
```

### Package Management
The project uses `uv` for dependency management (see `uv.lock`). The package is defined in `pyproject.toml` with two entry points:
- `wrf-ensembly`: Main CLI
- `wrf-ensembly-obs`: Observation handling CLI
The Main CLI always acts on an experiment directory, while the observation handling one works on individual or directory paths.

If you need to create an experiment, do it in `~/data/wrf_experiments/claude`.

### Documentation
Documentation uses MkDocs:
```bash
mkdocs serve    # Preview documentation locally
mkdocs build    # Build documentation
```

## Architecture

### Core Concepts

**Experiment-Centric Design**: All operations work with an "experiment" - a directory containing configuration, data, model binaries, and outputs for one assimilation experiment. Commands follow the pattern:
```bash
wrf-ensembly $EXPERIMENT_PATH group command
```

**Cycling Workflow**: The system performs ensemble data assimilation cycles:
1. Run WRF ensemble forward (multiple members)
2. Use DART to assimilate observations and produce analysis
3. Combine analysis with next cycle's IC/BC
4. Repeat

**State Tracking**: Experiment progress is tracked in a SQLite database (`status.db`) via the `ExperimentDatabase` class, recording which members have advanced and filter completion status.

### Key Modules

**Configuration (`config.py`)**
- Uses `mashumaro` with TOML mixins for (de)serialization
- Main class: `Config` (dataclass with nested config sections)
- Handles WRF/WPS namelists and experiment settings in one file
- Custom `UTCDatetimeStrategy` ensures timezone-aware datetime handling

**Experiment Module (`wrf_ensembly/experiment/`)**
- `Experiment`: Main class orchestrating experiment operations
- `ExperimentPaths`: Manages directory structure and paths
- `ExperimentDatabase`: SQLite interface for status tracking
- `ExperimentStatus`, `MemberStatus`: Status dataclasses

**Commands (`wrf_ensembly/commands/`)**
Commands are organized by workflow phase:
- `experiment.py`: Create, setup, cycle info
- `preprocess.py`: WPS workflow (geogrid, ungrib, metgrid, real)
- `ensemble.py`: Main cycling operations (advance-member, filter, analysis, cycle)
- `observations.py`: Observation file management
- `obs_sequence.py`: DART obs_seq file operations
- `postprocess.py`: Processing pipeline for outputs
- `slurm.py`: HPC job generation and submission
- `status.py`: Status viewing/management
- `validation.py`: Experiment validation

**Processors (`processors.py`)**
Plugin system for postprocessing data. Base class is `DataProcessor` with abstract `process()` method. Processors receive `ProcessingContext` and transform xarray datasets. Custom processors can be loaded from user Python files.

**Observations (`wrf_ensembly/observations/`)**
- `cli.py`: Observation CLI (`wrf-ensembly-obs`)
- `converters/`: Scripts to convert observations to DART format
- Separate from main CLI for preprocessing observation data

### Directory Structure in Experiments

```
experiment_dir/
├── config.toml              # Single config file
├── status.db                # SQLite status tracking
├── data/                    # Final outputs
│   ├── analysis/
│   ├── forecasts/
│   ├── diagnostics/
│   └── initial_boundary/
├── obs/                     # Observations (.toml, .obs_seq)
├── scratch/                 # Temporary/raw files (can be on different mount)
│   ├── forecasts/          # Raw wrfout files
│   ├── dart/
│   └── analysis/
├── work/                    # Model executables
│   ├── ensemble/           # One dir per member with wrf.exe
│   ├── preprocessing/      # WRF/WPS for IC/BC generation
│   ├── WRF/
│   └── WPS/
├── logs/                    # Timestamped logs per command
│   └── YYYY-MM-DD_HHMMSS-COMMAND/
└── jobfiles/               # Generated SLURM scripts
```

### Key Utilities

**Logging (`console.py`)**
- Uses Rich library for console output
- `logger.setup(command_name, experiment_path)` creates timestamped log dirs
- Format: `logs/YYYY-MM-DD_HHMMSS-COMMAND/wrf_ensembly.log`

**WRF Operations (`wrf.py`)**
- WRF-specific utilities (namelists, file operations)
- Handles wrfinput/wrfbdy files

**DART Integration (`obs_sequence.py`)**
- Reading/writing DART obs_seq files
- Observation filtering and combining

**SLURM (`jobfiles.py`)**
- Jinja2-based job file generation
- Dependency chains for cycling experiments

**Fortran Namelists (`fortran_namelists.py`)**
- Parsing/writing Fortran namelist format
- Used for WRF/WPS/DART namelists

## Important Patterns

1. **Config-first**: All settings go through the `Config` dataclass, validated on load
2. **Path management**: `ExperimentPaths` centralizes all path logic
3. **Status before action**: Check/update experiment status before operations
4. **External tool wrappers**: `external.py` provides wrappers for CDO, NCO operators
5. **Click decorators**: `@pass_experiment_path` injects experiment path from context
6. **Environment management**: `EnvironmentConfig` in config handles env vars per tool (WRF/DART/universal)

## WRF-CHEM Support

The toolkit supports WRF-CHEM through:
- Chemical IC interpolation via `interpolator-for-wrfchem` package
- Chemical observation converters in `observations/converters/`
- Configuration sections for chemistry options

## Postprocessing Pipeline

Default pipeline for wrfout files:
1. Apply xwrf for CF-compliance and diagnostics
2. Remove unneeded variables
3. Run custom processors (if configured)
4. Concatenate files per cycle/member
5. Compute ensemble statistics (mean/std)
6. Apply compression and packing

Controlled via `[postprocess]` config section. Raw wrfout files remain in `scratch/forecasts/`.

## Common Workflows

**Creating an experiment:**
```bash
wrf-ensembly $EXP_PATH experiment create <template>
wrf-ensembly $EXP_PATH experiment copy-model
wrf-ensembly $EXP_PATH experiment setup-dart
```

**Preprocessing:**
```bash
wrf-ensembly $EXP_PATH preprocess setup
wrf-ensembly $EXP_PATH preprocess geogrid
wrf-ensembly $EXP_PATH preprocess ungrib --cycle X
wrf-ensembly $EXP_PATH preprocess metgrid --cycle X
wrf-ensembly $EXP_PATH preprocess real --cycle X
```

**Running a cycle:**
```bash
wrf-ensembly $EXP_PATH ensemble advance-member --cycle X --member Y
wrf-ensembly $EXP_PATH ensemble filter --cycle X
wrf-ensembly $EXP_PATH ensemble cycle --cycle X
```

**SLURM mode:**
```bash
wrf-ensembly $EXP_PATH slurm submit-preprocessing --cycles 0-5
wrf-ensembly $EXP_PATH slurm submit-experiment --start-cycle 0
```
