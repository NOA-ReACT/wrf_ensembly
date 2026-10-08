# Usage

This document provides a quick guide on all commands of `wrf-ensembly`. They are written here mostly in the order they are used in a typical workflow, so this could also be used as a quick start guide.
Commands are grouped by their functionality in the following sections:

- [Experiment Management](#experiment-management)
- [Preprocessing](#preprocessing)
- [Observations](#observations)
- [Ensemble Management](#ensemble-management)
- [Postprocessing](#postprocessing)
- [Status](#status)
- [SLURM](#slurm)
- [Plots](#plots)
- [Validation](#validation)
- [Legacy obs_seq tools](#legacy-obs_seq-tools)

For a new experiment, you will typically start with creating it and copying the model ([experiment management](#experiment-management)), then preprocess the input data ([preprocessing](#preprocessing)), preprocess observations ([observations](#observations)), run the ensemble ([ensemble management](#ensemble-management)), and finally postprocess the results ([postprocessing](#postprocessing)). You can check the experiment status at any time using the [status](#status) commands. If you are using SLURM, you can also find commands for that in the [SLURM](#slurm) section (preprocess, run ensemble, postprocess).

All commands will take the path to the experiment directory as the first argument. This directory will contain the model data, input and output forecasts, configuration and anything else related to the experiment. It must be writable by the current user.

## Experiment Management

::: mkdocs-click
    :module: wrf_ensembly.commands.experiment
    :command: create
    :prog_name: wrf-ensembly EXPERIMENT_PATH experiment create
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.experiment
    :command: copy_model
    :prog_name: wrf-ensembly EXPERIMENT_PATH experiment copy-model
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.experiment
    :command: cycle_info
    :prog_name: wrf-ensembly EXPERIMENT_PATH experiment cycle-info
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.experiment
    :command: setup_dart
    :prog_name: wrf-ensembly EXPERIMENT_PATH experiment setup-dart
    :depth: 2

## Preprocessing

::: mkdocs-click
    :module: wrf_ensembly.commands.preprocess
    :command: setup
    :prog_name: wrf-ensembly EXPERIMENT_PATH preprocess setup
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.preprocess
    :command: geogrid
    :prog_name: wrf-ensembly EXPERIMENT_PATH preprocess geogrid
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.preprocess
    :command: ungrib
    :prog_name: wrf-ensembly EXPERIMENT_PATH preprocess ungrib
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.preprocess
    :command: metgrid
    :prog_name: wrf-ensembly EXPERIMENT_PATH preprocess metgrid
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.preprocess
    :command: real
    :prog_name: wrf-ensembly EXPERIMENT_PATH preprocess real
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.preprocess
    :command: icbc_status
    :prog_name: wrf-ensembly EXPERIMENT_PATH preprocess icbc-status
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.preprocess
    :command: interpolate_chem
    :prog_name: wrf-ensembly EXPERIMENT_PATH preprocess interpolate-chem
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.preprocess
    :command: clean
    :prog_name: wrf-ensembly EXPERIMENT_PATH preprocess clean
    :depth: 2

## Observations

Observations are first converted to the WRF-Ensembly parquet format with the separate `wrf-ensembly-obs` CLI, then added to the experiment's observation database. See [Observations](observations.md) for the full workflow.

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: add
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations add
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: show
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations show
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: delete
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations delete
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: prepare_cycles
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations prepare-cycles
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: cycle_summary
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations cycle-summary
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: cycle_info
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations cycle-info
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: plot
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations plot
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: plot_cycle_locations
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations plot-cycle-locations
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: plot_compare_obs_to_grid
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations plot-compare-obs-to-grid
    :depth: 2

## Ensemble Management

::: mkdocs-click
    :module: wrf_ensembly.commands.ensemble
    :command: setup
    :prog_name: wrf-ensembly EXPERIMENT_PATH ensemble setup
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.ensemble
    :command: setup_from_other_experiment
    :prog_name: wrf-ensembly EXPERIMENT_PATH ensemble setup-from-other-experiment
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.ensemble
    :command: plan_segment
    :prog_name: wrf-ensembly EXPERIMENT_PATH ensemble plan-segment
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.ensemble
    :command: generate_perturbations
    :prog_name: wrf-ensembly EXPERIMENT_PATH ensemble generate-perturbations
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.ensemble
    :command: apply_perturbations
    :prog_name: wrf-ensembly EXPERIMENT_PATH ensemble apply-perturbations
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.ensemble
    :command: update_bc
    :prog_name: wrf-ensembly EXPERIMENT_PATH ensemble update-bc
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.ensemble
    :command: advance_member
    :prog_name: wrf-ensembly EXPERIMENT_PATH ensemble advance-member
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.ensemble
    :command: finish_segment
    :prog_name: wrf-ensembly EXPERIMENT_PATH ensemble finish-segment
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.ensemble
    :command: filter
    :prog_name: wrf-ensembly EXPERIMENT_PATH ensemble filter
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.ensemble
    :command: analysis
    :prog_name: wrf-ensembly EXPERIMENT_PATH ensemble analysis
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.ensemble
    :command: cycle
    :prog_name: wrf-ensembly EXPERIMENT_PATH ensemble cycle
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.ensemble
    :command: reset_cycle
    :prog_name: wrf-ensembly EXPERIMENT_PATH ensemble reset-cycle
    :depth: 2

## Postprocessing

::: mkdocs-click
    :module: wrf_ensembly.commands.postprocess
    :command: print_variables_to_keep
    :prog_name: wrf-ensembly EXPERIMENT_PATH postprocess print-variables-to-keep
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.postprocess
    :command: run
    :prog_name: wrf-ensembly EXPERIMENT_PATH postprocess run
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.postprocess
    :command: clean
    :prog_name: wrf-ensembly EXPERIMENT_PATH postprocess clean
    :depth: 2

## Status

::: mkdocs-click
    :module: wrf_ensembly.commands.status
    :command: show
    :prog_name: wrf-ensembly EXPERIMENT_PATH status show
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.status
    :command: runtime_stats
    :prog_name: wrf-ensembly EXPERIMENT_PATH status runtime-stats
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.status
    :command: clear_runtime_stats
    :prog_name: wrf-ensembly EXPERIMENT_PATH status clear-runtime-stats
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.status
    :command: reset
    :prog_name: wrf-ensembly EXPERIMENT_PATH status reset
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.status
    :command: set_member
    :prog_name: wrf-ensembly EXPERIMENT_PATH status set-member
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.status
    :command: set_all_members
    :prog_name: wrf-ensembly EXPERIMENT_PATH status set-all-members
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.status
    :command: set_experiment
    :prog_name: wrf-ensembly EXPERIMENT_PATH status set-experiment
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.status
    :command: reconcile
    :prog_name: wrf-ensembly EXPERIMENT_PATH status reconcile
    :depth: 2

## SLURM

::: mkdocs-click
    :module: wrf_ensembly.commands.slurm
    :command: preprocessing
    :prog_name: wrf-ensembly EXPERIMENT_PATH slurm preprocessing
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.slurm
    :command: advance_members
    :prog_name: wrf-ensembly EXPERIMENT_PATH slurm advance-members
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.slurm
    :command: make_analysis
    :prog_name: wrf-ensembly EXPERIMENT_PATH slurm make-analysis
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.slurm
    :command: postprocess
    :prog_name: wrf-ensembly EXPERIMENT_PATH slurm postprocess
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.slurm
    :command: queue_all_postprocessing
    :prog_name: wrf-ensembly EXPERIMENT_PATH slurm queue-all-postprocessing
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.slurm
    :command: run_experiment
    :prog_name: wrf-ensembly EXPERIMENT_PATH slurm run-experiment
    :depth: 2

## Plots

::: mkdocs-click
    :module: wrf_ensembly.commands.plots
    :command: cycle_consistency
    :prog_name: wrf-ensembly EXPERIMENT_PATH plots cycle-consistency
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.plots
    :command: cycle_filter_stats
    :prog_name: wrf-ensembly EXPERIMENT_PATH plots cycle-filter-stats
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.plots
    :command: ensemble_spread
    :prog_name: wrf-ensembly EXPERIMENT_PATH plots ensemble-spread
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.plots
    :command: forecast
    :prog_name: wrf-ensembly EXPERIMENT_PATH plots forecast
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.plots
    :command: forecast_vs_analysis
    :prog_name: wrf-ensembly EXPERIMENT_PATH plots forecast-vs-analysis
    :depth: 2

## Validation

::: mkdocs-click
    :module: wrf_ensembly.commands.validation
    :command: interpolate_model
    :prog_name: wrf-ensembly EXPERIMENT_PATH validation interpolate-model
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.validation
    :command: interpolate_model_per_member
    :prog_name: wrf-ensembly EXPERIMENT_PATH validation interpolate-model-per-member
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.validation
    :command: analyze_first_departures
    :prog_name: wrf-ensembly EXPERIMENT_PATH validation analyze-first-departures
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.validation
    :command: analyze_lead_time_skill
    :prog_name: wrf-ensembly EXPERIMENT_PATH validation analyze-lead-time-skill
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.validation
    :command: obs_curtain
    :prog_name: wrf-ensembly EXPERIMENT_PATH validation obs-curtain
    :depth: 2

## Legacy obs_seq tools

These commands belong to the older workflow where each observation file was converted to `obs_seq` directly by a DART converter. They are kept for existing experiments; new experiments should use the [observations](#observations) commands instead.

::: mkdocs-click
    :module: wrf_ensembly.commands.obs_sequence
    :command: convert_obs
    :prog_name: wrf-ensembly EXPERIMENT_PATH obs-sequence convert-obs
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.obs_sequence
    :command: combine_obs
    :prog_name: wrf-ensembly EXPERIMENT_PATH obs-sequence combine-obs
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.obs_sequence
    :command: preprocess_for_wrf
    :prog_name: wrf-ensembly EXPERIMENT_PATH obs-sequence preprocess-for-wrf
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.obs_sequence
    :command: prepare_custom_window
    :prog_name: wrf-ensembly EXPERIMENT_PATH obs-sequence prepare-custom-window
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.obs_sequence
    :command: obs_seq_to_nc
    :prog_name: wrf-ensembly EXPERIMENT_PATH obs-sequence obs-seq-to-nc
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.obs_sequence
    :command: list_files
    :prog_name: wrf-ensembly EXPERIMENT_PATH obs-sequence list-files
    :depth: 2
