# Usage

This document provides a quick guide on all commands of `wrf-ensembly`. They are written here mostly in the order they are used in a typical workflow, so this could also be used as a quick start guide.
Commands are grouped by their functionality in the following sections:

- [Experiment Management](#experiment-management)
- [Preprocessing](#preprocessing)
- [Observations](#observations)
- [Ensemble Management](#ensemble-management)
- [Postprocessing](#postprocessing)
- [SLURM](#slurm)

For a new experiment, you will typically start with creating it and copying the model ([experiment management](#experiment-management)), then preprocess the input data ([preprocessing](#preprocessing)), preprocess observations ([observations](#observations)), run the ensemble ([ensemble management](#ensemble-management)), and finally postprocess the results ([postprocessing](#postprocessing)). If you are using SLURM, you can also find commands for that in the last section (preprocess, run ensemble, postprocess).

All commands will take the path to the experiment directory as the first argument. This directory will contain the model data, input and output forecasts, configuration and anything else related to the experiment. It must be writable by the current user.

## Experiment Management

::: mkdocs-click
    :module: wrf_ensembly.commands.experiment
    :command: create
    :prog_name: wrf-ensembly EXPERIMENT_PATH experiment create
    :depth: 2

