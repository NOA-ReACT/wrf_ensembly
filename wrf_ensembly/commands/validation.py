import datetime as dt
from pathlib import Path

import click

from wrf_ensembly.click_utils import GroupWithStartEndPrint, pass_experiment_path
from wrf_ensembly.console import logger
from wrf_ensembly.experiment import experiment
from wrf_ensembly.validation import (
    FirstDeparturesAnalysis,
    LeadTimeSkillAnalysis,
    ModelInterpolation,
    ObsCurtainAnalysis,
    PerMemberModelInterpolation,
)
from wrf_ensembly.wrf import get_wrf_cartopy_crs


@click.group(name="validation", cls=GroupWithStartEndPrint)
def validation_cli():
    """Commands related to validating an experiment, i.e. comparing model output to observations"""
    pass


def _resolve_pairs(
    exp,
    instrument_quantity: tuple[str, ...],
    config_pairs: list[str],
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None,
) -> list[tuple[str, str]] | None:
    """
    Work out which instrument-quantity pairs to analyse.

    Command line beats config, and config beats discovering everything in the
    database. Explicitly requested pairs are checked against what actually has
    model-interpolated values.

    Args:
        exp: The experiment.
        instrument_quantity: Pairs in dot notation from the command line.
        config_pairs: Pairs in dot notation from the configuration.
        start_time: If set, only consider observations at or after this time.
        end_time: If set, only consider observations before or at this time.

    Returns:
        The pairs to analyse, or None if there is nothing usable (already logged).
    """
    pairs: list[tuple[str, str]] = []
    source = instrument_quantity or config_pairs
    origin = "command line" if instrument_quantity else "config"

    if source:
        for pair_str in source:
            if "." not in pair_str:
                logger.error(
                    f"Invalid pair format in {origin}: {pair_str}. "
                    "Expected format: 'instrument.quantity'"
                )
                return None
            instr, qty = pair_str.split(".", 1)
            pairs.append((instr, qty))
        logger.info(f"Analyzing pairs from {origin}: {[f'{i}.{q}' for i, q in pairs]}")
    else:
        pairs = exp.obs.get_model_interpolated_pairs(start_time, end_time)
        logger.info(
            f"No pairs specified, analyzing all available: {[f'{i}.{q}' for i, q in pairs]}"
        )

    if not pairs:
        logger.warning("No pairs to analyze!")
        return None

    # When pairs were specified (not discovered), validate against what's in the DB
    if source:
        available = set(exp.obs.get_model_interpolated_pairs(start_time, end_time))
        pairs = [(i, q) for i, q in pairs if (i, q) in available]
        if not pairs:
            logger.warning("None of the specified pairs have model-interpolated data!")
            logger.warning("Run 'wrf-ensembly validation interpolate-model' first.")
            return None

    return pairs


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
@click.option(
    "--extended/--no-extended",
    "prefer_extended",
    default=None,
    help="When forecasts overlap (forecast_extension > 0), use the longer-lead "
    "independent forecast (--extended) or the shorter-lead analysis-driven one "
    "(--no-extended). Overrides validation.prefer_extended_forecast in the config.",
)
@pass_experiment_path
def interpolate_model(experiment_path: Path, prefer_extended: bool | None):
    """
    Interpolate the model outputs to the observation locations and times.

    Updates the `model_forecast` and `model_analysis` columns in the experiment's
    DuckDB observations table with the interpolated model values at each observation
    location and time. Analysis interpolation is skipped if no analysis mean files exist.
    """
    logger.setup("validation-interpolate-model", experiment_path)
    exp = experiment.Experiment(experiment_path)

    if prefer_extended is not None:
        exp.cfg.validation.prefer_extended_forecast = prefer_extended

    interpolation = ModelInterpolation(exp)
    interpolation.run()


@validation_cli.command()
@click.option(
    "--extended/--no-extended",
    "prefer_extended",
    default=None,
    help="When forecasts overlap (forecast_extension > 0), use the longer-lead "
    "independent forecast (--extended) or the shorter-lead analysis-driven one "
    "(--no-extended). Overrides validation.prefer_extended_forecast in the config.",
)
@pass_experiment_path
def interpolate_model_per_member(experiment_path: Path, prefer_extended: bool | None):
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

    if prefer_extended is not None:
        exp.cfg.validation.prefer_extended_forecast = prefer_extended

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

    pairs_to_analyze = _resolve_pairs(
        exp,
        instrument_quantity,
        exp.cfg.validation.first_departures.instrument_quantity_pairs,
        start_time,
        end_time,
    )
    if pairs_to_analyze is None:
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


@validation_cli.command()
@click.option(
    "--instrument-quantity",
    multiple=True,
    help="Instrument-quantity pair in dot notation (e.g. EarthCARE_ATLID_EBD."
    "LIDAR_EXTINCTION_355nm). Can be specified multiple times. Defaults to every "
    "pair with model-interpolated values.",
)
@click.option(
    "--file",
    "filenames",
    multiple=True,
    help="Draw only these source observation files. Can be specified multiple "
    "times. Defaults to every file that has both a prior and an analysis.",
)
@click.option(
    "--z-min",
    type=float,
    help="Drop vertical bins below this altitude, in the observation's z units.",
)
@click.option(
    "--z-max",
    type=float,
    help="Drop vertical bins above this altitude, in the observation's z units. "
    "Useful when the instrument profiles far above the layer of interest.",
)
@pass_experiment_path
def obs_curtain(
    experiment_path: Path,
    instrument_quantity: tuple[str, ...],
    filenames: tuple[str, ...],
    z_min: float | None,
    z_max: float | None,
):
    """
    Draw observation-space curtains for profile instruments.

    For each source observation file, produces a four-panel figure on the
    instrument's native grid: the observed field, the model prior and analysis
    interpolated to the same pixels, and the increment between them. This shows
    where in the vertical the filter moved the model.

    Only works for instruments that sample a two-dimensional curtain, such as
    lidars and radar profilers. Requires `validation interpolate-model` to have
    populated the model columns, and an analysis to exist for the cycle.

    Results are saved to data/validation/obs_curtain/{instrument}/{quantity}/
    """

    logger.setup("validation-obs-curtain", experiment_path)
    exp = experiment.Experiment(experiment_path)

    pairs_to_analyze = _resolve_pairs(exp, instrument_quantity, [])
    if pairs_to_analyze is None:
        return

    for instrument, quantity in pairs_to_analyze:
        logger.info(f"\n{'-' * 60}")
        logger.info(f"Drawing curtains for {instrument}.{quantity}")

        analysis = ObsCurtainAnalysis(
            exp, instrument, quantity, z_min=z_min, z_max=z_max
        )
        results = analysis.run(list(filenames) or None)

        logger.info(
            f"  {len(results['figures'])} figure(s) in {results['output_dir']}"
        )

    logger.info(f"\n{'=' * 60}")
    logger.info("Observation curtains complete!")


@validation_cli.command()
@click.option(
    "--instrument-quantity",
    multiple=True,
    help="Instrument-quantity pair in dot notation (e.g. MODIS.AOD_550nm). Can be "
    "specified multiple times. Defaults to every pair with model-interpolated values.",
)
@click.option(
    "--bin-hours",
    type=float,
    default=1.0,
    show_default=True,
    help="Lead-time bin width in hours.",
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
    "matching obs, or 'key!=value' to exclude them. Repeatable.",
)
@click.option(
    "--extended/--no-extended",
    "prefer_extended",
    default=None,
    help="When forecasts overlap (forecast_extension > 0), attribute each "
    "observation to the longer-lead independent forecast (--extended) or the "
    "shorter-lead analysis-driven one (--no-extended). Overrides "
    "validation.prefer_extended_forecast in the config.",
)
@pass_experiment_path
def analyze_lead_time_skill(
    experiment_path: Path,
    instrument_quantity: tuple[str, ...],
    bin_hours: float,
    start_time: dt.datetime | None = None,
    end_time: dt.datetime | None = None,
    metadata_filter: tuple[str, ...] = (),
    prefer_extended: bool | None = None,
):
    """
    Score forecast departures as a function of lead time.

    Pooling every observation together conflates a genuinely skillful model with
    one that simply had a recent analysis. This bins the departures by how long the
    owning cycle's forecast had been running, and reports the bias, RMS and
    correlation per bin, so error growth with lead is visible.

    Lead time is reconstructed from the cycle definitions, replaying the same rule
    the model interpolation uses to pick between overlapping forecasts. Note that
    an analysis only exists for cycles where the filter ran, so the O-A curve
    usually covers only the shortest leads.

    Results are saved to data/validation/lead_time_skill/{instrument}/{quantity}/
    """

    logger.setup("validation-analyze-lead-time-skill", experiment_path)
    exp = experiment.Experiment(experiment_path)

    if prefer_extended is not None:
        exp.cfg.validation.prefer_extended_forecast = prefer_extended

    metadata_filters = _parse_metadata_filters(metadata_filter)
    if metadata_filters:
        logger.info(
            f"Applying metadata filters: {['{} {} {}'.format(*f) for f in metadata_filters]}"
        )

    pairs_to_analyze = _resolve_pairs(
        exp, instrument_quantity, [], start_time, end_time
    )
    if pairs_to_analyze is None:
        return

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

        analysis = LeadTimeSkillAnalysis(exp, instrument, quantity, bin_hours)
        results = analysis.run(pair_df)

        logger.info(f"  Output directory: {results['output_dir']}")
        if "skill" in results:
            for line in results["skill"].to_string(
                index=False, float_format=lambda x: f"{x:.4g}"
            ).splitlines():
                logger.info(line)

    logger.info(f"\n{'=' * 60}")
    logger.info("Lead-time skill analysis complete!")
