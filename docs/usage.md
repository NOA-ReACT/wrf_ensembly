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

For a new experiment, you will typically start with creating it and copying the model ([experiment management](#experiment-management)), then preprocess the input data ([preprocessing](#preprocessing)), preprocess observations ([observations](#observations)), run the ensemble ([ensemble management](#ensemble-management)), and finally postprocess the results ([postprocessing](#postprocessing)). You can check the experiment status at any time using the [status](#status) commands. If you are using SLURM, you can also find commands for that in the last section (preprocess, run ensemble, postprocess).

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
    :command: interpolate_chem
    :prog_name: wrf-ensembly EXPERIMENT_PATH preprocess interpolate-chem
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.preprocess
    :command: clean
    :prog_name: wrf-ensembly EXPERIMENT_PATH preprocess clean
    :depth: 2

## Observations

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: convert_obs
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations convert-obs
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: combine_obs
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations combine-obs
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: prepare_custom_window
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations prepare-custom-window
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: obs_seq_to_nc
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations obs-seq-to-nc
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: preprocess_for_wrf
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations preprocess-for-wrf
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.observations
    :command: list_files
    :prog_name: wrf-ensembly EXPERIMENT_PATH observations list-files
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

## Postprocessing

::: mkdocs-click
    :module: wrf_ensembly.commands.postprocess
    :command: print_variables_to_keep
    :prog_name: wrf-ensembly EXPERIMENT_PATH postprocess print-variables-to-keep
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.postprocess
    :command: process_pipeline
    :prog_name: wrf-ensembly EXPERIMENT_PATH postprocess process-pipeline
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.postprocess
    :command: statistics
    :prog_name: wrf-ensembly EXPERIMENT_PATH postprocess statistics
    :depth: 2

::: mkdocs-click
    :module: wrf_ensembly.commands.postprocess
    :command: concatenate
    :prog_name: wrf-ensembly EXPERIMENT_PATH postprocess concatenate
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

