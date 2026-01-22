from pathlib import Path

import click
import pandas as pd

from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.console import logger
from wrf_ensembly.experiment import experiment
from wrf_ensembly.validation import FirstDeparturesAnalysis, ModelInterpolation


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
    "--instrument-quantity",
    multiple=True,
    help="Specific instrument-quantity pairs to analyze in dot notation (e.g., MODIS.AOD_550nm). Can be specified multiple times. Overrides config.",
)
@pass_experiment_path
def analyze_first_departures(experiment_path: Path, instrument_quantity: tuple):
    """
    Analyze first departures (O-B) statistics for validation.

    Generates statistical analysis and plots for each instrument-quantity pair in the
    model_interpolated.parquet file. This includes:
    - Overall statistics (bias, std, RMSE)
    - Histogram of first departure values
    - Time series of mean and std deviation
    - Spatial maps of bias and std deviation
    - Regime-based analysis (if configured)

    Results are saved to data/validation/first_departures/{instrument}/{quantity}/

    Pairs to analyze are configured in config.toml under
    [validation.first_departures.instrument_quantity_pairs]. Use --instrument-quantity to override.
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

    # Determine which instrument-quantity pairs to analyze
    pairs_to_analyze = []

    if instrument_quantity:
        # Override from command line
        for pair_str in instrument_quantity:
            if "." not in pair_str:
                logger.error(
                    f"Invalid pair format: {pair_str}. Expected format: 'instrument.quantity'"
                )
                return
            instrument, quantity = pair_str.split(".", 1)
            pairs_to_analyze.append((instrument, quantity))
        logger.info(
            f"Analyzing pairs from command line: {[f'{i}.{q}' for i, q in pairs_to_analyze]}"
        )
    elif exp.cfg.validation.first_departures.instrument_quantity_pairs:
        # Use config
        for pair_str in exp.cfg.validation.first_departures.instrument_quantity_pairs:
            if "." not in pair_str:
                logger.error(
                    f"Invalid pair format in config: {pair_str}. Expected format: 'instrument.quantity'"
                )
                return
            instrument, quantity = pair_str.split(".", 1)
            pairs_to_analyze.append((instrument, quantity))
        logger.info(
            f"Analyzing pairs from config: {[f'{i}.{q}' for i, q in pairs_to_analyze]}"
        )
    else:
        # Use all available pairs
        pairs_to_analyze = df.groupby(["instrument", "quantity"]).size().index.tolist()
        logger.info(
            f"No pairs specified, analyzing all available: {[f'{i}.{q}' for i, q in pairs_to_analyze]}"
        )

    # Filter to pairs that exist in the data
    available_pairs = set(df.groupby(["instrument", "quantity"]).size().index.tolist())
    pairs_to_analyze = [
        (i, q) for i, q in pairs_to_analyze if (i, q) in available_pairs
    ]

    if not pairs_to_analyze:
        logger.warning("No pairs to analyze!")
        return

    # Analyze each instrument-quantity pair
    for instrument, quantity in pairs_to_analyze:
        logger.info(f"\n{'-' * 60}")
        logger.info(f"Analyzing {instrument}.{quantity}")

        # Filter data for this instrument-quantity pair
        pair_df = df[
            (df["instrument"] == instrument) & (df["quantity"] == quantity)
        ].copy()

        if len(pair_df) == 0:
            logger.warning(f"No data found for {instrument}.{quantity}, skipping")
            continue

        # Run analysis
        analysis = FirstDeparturesAnalysis(exp, instrument, quantity)
        results = analysis.run(pair_df)

        logger.info(f"Results for {instrument}.{quantity}:")
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

    logger.info(f"\n{'=' * 60}")
    logger.info("First departures analysis complete!")
