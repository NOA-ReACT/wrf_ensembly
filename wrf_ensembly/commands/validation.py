from pathlib import Path

import click

from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.console import logger
from wrf_ensembly.experiment import experiment
from wrf_ensembly.validation import ModelInterpolation


@click.group(name="validation", cls=GroupWithStartEndPrint)
def validation_cli():
    """Commands related to validating an experiment, i.e. comparing model output to observations"""
    pass


@validation_cli.command()
@pass_experiment_path
def interpolate_model(experiment_path: Path):
    """
    Interpolate the model outputs to the observation locations and times.

    This will create a `validation` directory in the experiment directory, containing
    the results of the validation in parquet format.

    The output file will be in WRF ensembly observation file format but include additional columns:
    - model_value: The value from the model at the observation location and time
    - used_in_da: A boolean indicating if the observation was used in data assimilation
    - cycle: The index of the assimilation cycle the observation was used in (if any)
    """
    logger.setup("validation-interpolate-model", experiment_path)
    exp = experiment.Experiment(experiment_path)

    interpolation = ModelInterpolation(exp)
    interpolation.run()
