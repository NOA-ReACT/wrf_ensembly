import datetime as dt
from pathlib import Path

import click

from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.console import logger
from wrf_ensembly.experiment import experiment
from wrf_ensembly.validation import FirstDeparturesAnalysis, ModelInterpolation, PerMemberModelInterpolation
from wrf_ensembly.wrf import get_wrf_cartopy_crs


@click.group(name="validation", cls=GroupWithStartEndPrint)
def validation_cli():
    """Commands related to validating an experiment, i.e. comparing model output to observations"""
    pass


def _parse_metadata_filters(raw: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """Parse 'key=value' / 'key!=value' strings into (key, op, value) tuples."""
    parsed: list[tuple[str, str, str]] = []
    for f in raw:
        if "!=" in f:
            key, value = f.split("!=", 1)
            op = "!="
        elif "=" in f:
            key, value = f.split("=", 1)
            op = "="
        else:
            raise click.BadParameter(
                f"Invalid metadata filter '{f}'. Expected 'key=value' or 'key!=value'."
            )
        parsed.append((key.strip(), op, value.strip()))
    return parsed


@validation_cli.command()
@pass_experiment_path
def interpolate_model(experiment_path: Path):
    """
    Interpolate the model outputs to the observation locations and times.

    Updates the `model_forecast` and `model_analysis` columns in the experiment's
    DuckDB observations table with the interpolated model values at each observation
    location and time. Analysis interpolation is skipped if no analysis mean files exist.
    """
    logger.setup("validation-interpolate-model", experiment_path)
    exp = experiment.Experiment(experiment_path)

    interpolation = ModelInterpolation(exp)
    interpolation.run()


@validation_cli.command()
@pass_experiment_path
def interpolate_model_per_member(experiment_path: Path):
    """
    Interpolate each ensemble member's model output to observation locations.

    Reads the per-member ensemble files produced when postprocess.keep_per_member
    is true and writes results to data/validation/model_member_{forecast,analysis}.parquet.
    Each parquet file is in long form: one row per (observation × member).

    The existing model_forecast / model_analysis columns in DuckDB are not modified.
    Run interpolate-model first to populate those.
    """
    logger.setup("validation-interpolate-model-per-member", experiment_path)
    exp = experiment.Experiment(experiment_path)

    interpolation = PerMemberModelInterpolation(exp)
    interpolation.run()


@validation_cli.command()
@click.option(
    "--instrument-quantity",
    multiple=True,
    help="Specific instrument-quantity pairs to analyze in dot notation (e.g., MODIS.AOD_550nm). Can be specified multiple times. Overrides config.",
)
@click.option(
    "--start-time",
    type=click.DateTime(),
    help="First timestamp to consider in the analysis (ISO format)",
)
@click.option(
    "--end-time",
    type=click.DateTime(),
    help="Last timestamp to consider in the analysis (ISO format)",
)
@click.option(
    "--metadata-filter",
    "metadata_filter",
    multiple=True,
    help="Filter observations on a metadata JSON key. Use 'key=value' to keep only "
    "matching obs, or 'key!=value' to exclude them. Repeatable. Observations whose "
    "metadata lacks the key are always kept. Values are matched as text, so use "
    "is_over_land=1 (not =true). E.g. --metadata-filter is_over_land=1 keeps only "
    "sea observations.",
)
@pass_experiment_path
def analyze_first_departures(
    experiment_path: Path,
    instrument_quantity: tuple[str, ...],
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None,
    metadata_filter: tuple[str, ...] = (),
):
    """
    Analyze first departures (O-B) statistics for validation.

    Generates statistical analysis and plots for each instrument-quantity pair with
    model-interpolated values in DuckDB. This includes:
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

    metadata_filters = _parse_metadata_filters(metadata_filter)
    if metadata_filters:
        logger.info(
            f"Applying metadata filters: {['{} {} {}'.format(*f) for f in metadata_filters]}"
        )

    # Get cartopy projection
    try:
        proj = get_wrf_cartopy_crs(exp.cfg.domain_control)
    except NotImplementedError:
        logger.warning(
            "Could not create cartopy projection for this domain, falling back to PlateCarree"
        )
        proj = None

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
            instr, qty = pair_str.split(".", 1)
            pairs_to_analyze.append((instr, qty))
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
            instr, qty = pair_str.split(".", 1)
            pairs_to_analyze.append((instr, qty))
        logger.info(
            f"Analyzing pairs from config: {[f'{i}.{q}' for i, q in pairs_to_analyze]}"
        )
    else:
        # Use all available pairs from DuckDB
        pairs_to_analyze = exp.obs.get_model_interpolated_pairs(start_time, end_time)
        logger.info(
            f"No pairs specified, analyzing all available: {[f'{i}.{q}' for i, q in pairs_to_analyze]}"
        )

    if not pairs_to_analyze:
        logger.warning("No pairs to analyze!")
        return

    # When pairs were specified (not discovered), validate against what's in the DB
    if (
        instrument_quantity
        or exp.cfg.validation.first_departures.instrument_quantity_pairs
    ):
        available_pairs = set(
            exp.obs.get_model_interpolated_pairs(start_time, end_time)
        )
        pairs_to_analyze = [
            (i, q) for i, q in pairs_to_analyze if (i, q) in available_pairs
        ]
        if not pairs_to_analyze:
            logger.warning("None of the specified pairs have model-interpolated data!")
            logger.warning("Run 'wrf-ensembly validation interpolate-model' first.")
            return

    # Analyze each instrument-quantity pair, loading data from DuckDB per pair
    for instrument, quantity in pairs_to_analyze:
        logger.info(f"\n{'-' * 60}")
        logger.info(f"Analyzing {instrument}.{quantity}")

        pair_df = exp.obs.get_model_interpolated_for_pair(
            instrument,
            quantity,
            qc_flags=[0, -1],
            start_date=start_time,
            end_date=end_time,
            metadata_filters=metadata_filters,
        )

        if pair_df is None or len(pair_df) == 0:
            logger.warning(f"No data found for {instrument}.{quantity}, skipping")
            continue

        # Run analysis
        analysis = FirstDeparturesAnalysis(exp, instrument, quantity, proj=proj)
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
