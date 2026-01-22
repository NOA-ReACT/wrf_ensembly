from pathlib import Path

import click
import pandas as pd

from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.console import logger
from wrf_ensembly.experiment import experiment
from wrf_ensembly.validation import ModelInterpolation, FirstDeparturesAnalysis


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


@validation_cli.command()
@click.option(
    "--quantities",
    multiple=True,
    help="Specific quantities to analyze (overrides config). Can be specified multiple times.",
)
@pass_experiment_path
def analyze_first_departures(experiment_path: Path, quantities: tuple):
    """
    Analyze first departures (O-B) statistics for validation.

    Generates statistical analysis and plots for each observation type in the
    model_interpolated.parquet file. This includes:
    - Overall statistics (bias, std, RMSE)
    - Histogram of first departure values
    - Time series of mean and std deviation
    - Spatial maps of bias and std deviation
    - Regime-based analysis (if configured)

    Results are saved to data/validation/first_departures/{quantity}/

    Quantities to analyze are configured in config.toml under
    [validation.first_departures]. Use --quantities to override.
    """
    logger.setup("validation-analyze-first-departures", experiment_path)
    exp = experiment.Experiment(experiment_path)

    # Load model_interpolated.parquet
    model_interpolated_file = exp.paths.data / "model_interpolated.parquet"
    if not model_interpolated_file.exists():
        logger.error(
            f"Model interpolated file not found at {model_interpolated_file}. "
            "Run 'wrf-ensembly validation interpolate-model' first."
        )
        return

    logger.info(f"Loading model interpolated data from {model_interpolated_file}")
    df = pd.read_parquet(model_interpolated_file)

    # Determine which quantities to analyze
    if quantities:
        # Override from command line
        quantities_to_analyze = list(quantities)
        logger.info(f"Analyzing quantities from command line: {quantities_to_analyze}")
    elif exp.cfg.validation.first_departures.quantities:
        # Use config
        quantities_to_analyze = exp.cfg.validation.first_departures.quantities
        logger.info(f"Analyzing quantities from config: {quantities_to_analyze}")
    else:
        # Use all available quantities
        quantities_to_analyze = df["quantity"].unique().tolist()
        logger.info(f"No quantities specified, analyzing all: {quantities_to_analyze}")

    # Filter to quantities that exist in the data
    available_quantities = df["quantity"].unique()
    quantities_to_analyze = [q for q in quantities_to_analyze if q in available_quantities]

    if not quantities_to_analyze:
        logger.warning("No quantities to analyze!")
        return

    # Analyze each quantity
    for quantity in quantities_to_analyze:
        logger.info(f"\n{'='*60}")
        logger.info(f"Analyzing {quantity}")
        logger.info(f"{'='*60}")

        # Filter data for this quantity
        quantity_df = df[df["quantity"] == quantity].copy()

        if len(quantity_df) == 0:
            logger.warning(f"No data found for {quantity}, skipping")
            continue

        # Run analysis
        analysis = FirstDeparturesAnalysis(exp, quantity)
        results = analysis.run(quantity_df)

        logger.info(f"Results for {quantity}:")
        logger.info(f"  Output directory: {results['output_dir']}")
        if "statistics_file" in results:
            logger.info(f"  Statistics: {results['statistics_file']}")
        if "histogram" in results:
            logger.info(f"  Histogram: {results['histogram']}")
        if "timeseries" in results:
            logger.info(f"  Time series: {results['timeseries']}")
        if "spatial_maps" in results:
            logger.info(f"  Spatial maps: {results['spatial_maps']}")
        if "regime_plots" in results:
            logger.info(f"  Regime analysis: {results['regime_plots']}")

    logger.info(f"\n{'='*60}")
    logger.info("First departures analysis complete!")
