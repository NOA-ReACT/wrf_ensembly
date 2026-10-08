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

Tests live in `tests/` at the repo root, one file per module under test. They are
deliberately kept out of `wrf_ensembly/` so they are not shipped in the built wheel.
`testpaths` in `pyproject.toml` restricts collection to `tests/`, so files named
`test_*` elsewhere (e.g. the exploration scripts in `scripts/`) are ignored.

```bash
# Run all tests
pytest

# Run a specific test
pytest tests/test_utils.py::test_int_to_letter_numeral
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

**State Tracking**: Experiment progress is tracked in the `status/` directory via the `ExperimentState` class (`experiment/state_store.py`), recording which members have advanced and filter completion status. It is one small file per fact, each with a single writer, written atomically with no locking — members advance as separate jobs on separate nodes, so any lock-based store is unreliable on the shared filesystems these experiments run on. Cycle state is derived from these files rather than stored, so it cannot go stale (see `experiment/state_machine.py`).

### Key Modules

**Configuration (`config.py`)**
- Uses `mashumaro` with TOML mixins for (de)serialization
- Main class: `Config` (dataclass with nested config sections)
- Handles WRF/WPS namelists and experiment settings in one file
- Custom `UTCDatetimeStrategy` ensures timezone-aware datetime handling

**Experiment Module (`wrf_ensembly/experiment/`)**
- `Experiment`: Main class orchestrating experiment operations
- `ExperimentPaths`: Manages directory structure and paths
- `ExperimentState` (`state_store.py`): file-based status store under `status/`
- `ExperimentStateMachine` (`state_machine.py`): derives cycle state from the state store
- `MemberStatus` (`dataclasses.py`): per-member status dataclass

**Commands (`wrf_ensembly/commands/`)**
Commands are organized by workflow phase:
- `experiment.py`: Create, setup, cycle info
- `preprocess.py`: WPS workflow (geogrid, ungrib, metgrid, real)
- `ensemble.py`: Main cycling operations (advance-member, filter, analysis, cycle)
- `observations.py`: Observation database management (`add`, `show`, `prepare-cycles`, ...)
- `obs_sequence.py`: Legacy DART obs_seq file operations (`obs-sequence` group)
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
├── status/                  # Status tracking, one file per fact
│   ├── experiment.json      # Current cycle
│   ├── cycles/cycle_NNN/    # Member advancement + completion markers
│   └── segments/            # Segment plans (with [segments])
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

**Cycling modes** (`assimilation.cycling_mode`): `wrfinput` makes one wrfinput/wrfbdy per cycle with real.exe and copies `cycled_variables` from the analysis into the next wrfinput. `restart` runs real.exe once for the whole experiment and continues each member from its own WRF restart file (`scratch/restart/`), writing only the analysis' `state_variables` into it.
- `restart.py`: writing into restart files (two time levels `X_1`/`X_2` for the dynamics fields)
- `rebalance.py`: recomputes P/AL after the restart file's state changed (analysis, perturbations)
- `update_bc.py`: Python port of DART's `update_wrf_bc` (hybrid-coordinate coupling), reads wrfinput or restart files

**Segments** (`[segments]`, restart mode only): members run several cycles as one WRF run and only stop at cycles with an `obs/cycle_NNN.obs_seq`, in `keep_restart_files_for_cycles`, at `--run-until`, at the end, or when `max_walltime` is reached. The cycle stays the bookkeeping unit; the segment is an overlay read from plans in `status/segments/` (no extra cycle state).
- `segments.py`: the pure planner (`plan_segment`, `find_stops`, `checkpoint_interval`, `shorten`)
- `checkpoints.py`: restart files written during a segment; `CheckpointWatcher` confirms them from rsl.out.0000 (`Timing for Writing restart`) and prunes old ones while wrf.exe runs
- In `Experiment`: `advance_member` runs the segment, `file_segment_outputs` sorts wrfouts into their cycles, `find_resume_point` continues a killed member from a confirmed checkpoint, `finish_segment` moves the pointer from the first to the last cycle (first step of the analysis job), `reset_segment`
- Tests run `advance_member` with `tests/fake_wrf.py` instead of wrf.exe

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
wrf-ensembly $EXP_PATH preprocess ungrib
wrf-ensembly $EXP_PATH preprocess metgrid
wrf-ensembly $EXP_PATH preprocess real --cycle X   # cycling_mode = "wrfinput", for every cycle
wrf-ensembly $EXP_PATH preprocess real             # cycling_mode = "restart", once
```

**Running a cycle** (these act on the experiment's current cycle):
```bash
wrf-ensembly $EXP_PATH ensemble advance-member --member Y
wrf-ensembly $EXP_PATH ensemble finish-segment      # with [segments], no-op otherwise
wrf-ensembly $EXP_PATH ensemble filter
wrf-ensembly $EXP_PATH ensemble analysis
wrf-ensembly $EXP_PATH ensemble cycle
```

**SLURM mode:**
```bash
wrf-ensembly $EXP_PATH slurm preprocessing
wrf-ensembly $EXP_PATH slurm run-experiment --all-cycles
```
