"""
Streaming postprocessing pipeline for ensemble WRF output.

This module implements a memory-efficient streaming approach that processes
ensemble members one at a time, computes statistics on-the-fly using Welford's
algorithm, and writes results directly to final output files without intermediate
files.
"""

import re
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from wrf_ensembly.config import Config, PostprocessConfig
from wrf_ensembly.console import logger
from wrf_ensembly.experiment import Experiment
from wrf_ensembly.postprocess.streaming_writer import (
    StreamingEnsembleWriter,
    StreamingNetCDFWriter,
)
from wrf_ensembly.processors import ProcessingContext, ProcessorPipeline
from wrf_ensembly.statistics import (
    COORDINATE_VARIABLES,
    NetCDFFile,
    WelfordState,
    add_member_dimension,
    create_welford_accumulators,
    finalize_accumulators,
    get_structure_from_xarray,
    update_accumulators_from_dataset,
)


def _create_writer(
    path: Path,
    template: NetCDFFile,
    postprocess_cfg: PostprocessConfig,
) -> StreamingNetCDFWriter:
    """
    Create a StreamingNetCDFWriter with compression settings from config.

    Args:
        path: Path where the output file will be created.
        template: NetCDFFile structure to use as template.
        postprocess_cfg: Postprocess configuration with compression settings.

    Returns:
        Configured StreamingNetCDFWriter instance.
    """
    return StreamingNetCDFWriter(
        path,
        template,
        compression=postprocess_cfg.compression,
        complevel=postprocess_cfg.compression_level,
        shuffle=postprocess_cfg.shuffle,
        significant_digits=postprocess_cfg.significant_digits
        if postprocess_cfg.significant_digits != 0
        else None,
        significant_digits_overrides=postprocess_cfg.significant_digits_overrides,
        quantize_mode=postprocess_cfg.quantize_mode,
    )


def _create_ensemble_writer(
    path: Path,
    template: NetCDFFile,
    n_members: int,
    postprocess_cfg: PostprocessConfig,
) -> StreamingEnsembleWriter:
    """
    Create a StreamingEnsembleWriter with compression settings from config.

    Args:
        path: Path where the output file will be created.
        template: NetCDFFile structure (already with member dimension) to use as template.
        n_members: Number of ensemble members.
        postprocess_cfg: Postprocess configuration with compression settings.

    Returns:
        Configured StreamingEnsembleWriter instance.
    """
    return StreamingEnsembleWriter(
        path,
        template,
        n_members,
        compression=postprocess_cfg.compression,
        complevel=postprocess_cfg.compression_level,
        shuffle=postprocess_cfg.shuffle,
        significant_digits=postprocess_cfg.significant_digits
        if postprocess_cfg.significant_digits != 0
        else None,
        significant_digits_overrides=postprocess_cfg.significant_digits_overrides,
        quantize_mode=postprocess_cfg.quantize_mode,
    )


def process_members_for_timestep(
    wrfout_paths: list[Path],
    pipeline: ProcessorPipeline,
    cycle: int,
    config: Config,
    ensemble_writer: StreamingEnsembleWriter | None = None,
    precomputed_first_member: xr.Dataset | None = None,
) -> tuple[dict[str, WelfordState], np.ndarray]:
    """
    Process all ensemble members for a single output timestep.

    Opens each member's wrfout file, applies the processor pipeline,
    and accumulates statistics using Welford's algorithm.

    Args:
        wrfout_paths: List of paths to wrfout files, one per member.
        pipeline: Configured processor pipeline to apply.
        cycle: Current cycle number.
        config: Experiment configuration.
        ensemble_writer: Optional ensemble writer for per-member output. If provided,
            each member's processed data is written as a slice before finalizing.
        precomputed_first_member: If provided, used directly for member 0 instead of
            re-opening and re-processing its wrfout file.

    Returns:
        Tuple of (accumulators, time_value).
    """
    accumulators: dict[str, WelfordState] | None = None
    time_val: np.ndarray | None = None

    for member_i, wrfout_path in enumerate(wrfout_paths):
        # Use precomputed result for member 0 when available
        if member_i == 0 and precomputed_first_member is not None:
            logger.debug(
                f"Using cached pipeline result for member 0: {wrfout_path.name}"
            )
            processed = precomputed_first_member
        else:
            if not wrfout_path.exists():
                logger.warning(f"Member file not found, skipping: {wrfout_path}")
                continue

            logger.debug(f"Processing member {member_i}: {wrfout_path.name}")

            with xr.open_dataset(wrfout_path) as ds:
                context = ProcessingContext(
                    member=member_i,
                    cycle=cycle,
                    input_file=wrfout_path,
                    output_file=Path("/dev/null"),  # Not used in streaming mode
                    config=config,
                )
                processed = pipeline.process(ds, context)

        # Squeeze the time dimension — each wrfout file is a single
        # timestep so the leading t-dim is always size 1.  Removing it
        # here keeps shapes consistent with what the writers and Welford
        # accumulators expect (no extra leading dim).
        time_val = processed["t"].values
        processed = processed.squeeze("t", drop=False)

        # Initialize accumulators from first member's structure
        if accumulators is None:
            accumulators = create_welford_accumulators(processed)

        # Update running statistics
        update_accumulators_from_dataset(accumulators, processed)

        # Write per-member slice if ensemble writer is active
        if ensemble_writer is not None:
            member_data = {
                var: processed[var].values for var in processed.data_vars
            }
            ensemble_writer.write_member(member_data, time_val, member_i)

    if accumulators is None or time_val is None:
        raise ValueError("No member files were successfully processed")

    if ensemble_writer is not None:
        ensemble_writer.finalize_timestep()

    return accumulators, time_val


def _build_ensemble_template(
    template: NetCDFFile,
    n_members: int,
    postprocess_cfg: PostprocessConfig,
) -> NetCDFFile:
    """
    Build the ensemble file template, optionally filtering variables.

    Args:
        template: Base NetCDFFile template (from mean/sd structure).
        n_members: Number of ensemble members.
        postprocess_cfg: Config, used to check variables_to_keep_ensemble.

    Returns:
        NetCDFFile with member dimension added to time-varying variables.
    """
    filtered_template = template

    if postprocess_cfg.variables_to_keep_ensemble is not None:
        patterns = [re.compile(v) for v in postprocess_cfg.variables_to_keep_ensemble]
        filtered_vars = {
            name: var
            for name, var in template.variables.items()
            if name in COORDINATE_VARIABLES or any(p.match(name) for p in patterns)
        }
        filtered_template = NetCDFFile(
            dimensions=template.dimensions,
            variables=filtered_vars,
            global_attributes=template.global_attributes,
        )

    return add_member_dimension(filtered_template, n_members)


def process_cycle_streaming(
    exp: Experiment,
    cycle: int,
    pipeline: ProcessorPipeline,
    source: Literal["forecast", "analysis"],
    only_last_timestep: bool = False,
) -> tuple[Path, Path, Path | None] | None:
    """
    Process an entire cycle in streaming fashion.

    For each output timestep:
    1. Process all members through the pipeline (one at a time)
    2. Accumulate statistics using Welford's algorithm
    3. Append results directly to output files

    If ``keep_per_member`` is enabled in the config, also writes a per-member
    ensemble file with shape ``(t, member, ...)`` for all data variables.

    Args:
        exp: Experiment object with configuration and paths.
        cycle: Cycle number to process.
        pipeline: Configured processor pipeline.
        source: Either "forecast" or "analysis".
        only_last_timestep: If True, only process the last timestep of the cycle.

    Returns:
        Tuple of (mean_path, sd_path, ensemble_path) or None if no files to process.
        ensemble_path is None when keep_per_member is False.
    """
    n_members = exp.cfg.assimilation.n_members

    # Determine paths based on source type
    if source == "forecast":
        scratch_dir = exp.paths.scratch_forecasts_path(cycle)
        output_dir = exp.paths.forecast_path(cycle)
    else:
        scratch_dir = exp.paths.scratch_analysis_path(cycle)
        output_dir = exp.paths.analysis_path(cycle)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all output timesteps by checking member_00
    member_00_dir = scratch_dir / "member_00"
    if not member_00_dir.exists():
        logger.warning(f"Member 00 directory not found: {member_00_dir}")
        return None

    member_00_files = sorted(member_00_dir.glob("wrfout_d01_*"))

    # Filter out cycle start time file (should not be included in output)
    cycle_start_str = exp.cycles[cycle].start.strftime("%Y-%m-%d_%H:%M:%S")
    member_00_files = [f for f in member_00_files if cycle_start_str not in f.name]

    if not member_00_files:
        logger.warning(f"No wrfout files found in {member_00_dir}")
        return None

    # If only processing last timestep, filter to just that file
    if only_last_timestep:
        cycle_end_str = exp.cycles[cycle].end.strftime("%Y-%m-%d_%H:%M:%S")
        member_00_files = [f for f in member_00_files if cycle_end_str in f.name]
        if not member_00_files:
            logger.warning(f"Last timestep file not found for cycle {cycle}")
            return None

    logger.info(
        f"Processing {len(member_00_files)} timesteps for {source}, cycle {cycle}"
    )

    # Process first timestep to get template structure
    first_file = member_00_files[0]
    first_member_paths = [
        scratch_dir / f"member_{i:02d}" / first_file.name for i in range(n_members)
    ]

    logger.info(f"Building template from first timestep: {first_file.name}")

    # Extract template structure for output files
    # Use cycle start time as reference for time coordinate
    reference_time = np.datetime64(exp.cycles[cycle].start.replace(tzinfo=None))

    # Create output files
    mean_path = output_dir / f"{source}_mean_cycle_{cycle:03d}.nc"
    sd_path = output_dir / f"{source}_sd_cycle_{cycle:03d}.nc"

    keep_per_member = exp.cfg.postprocess.keep_per_member
    ensemble_path: Path | None = None

    if keep_per_member:
        ensemble_path = output_dir / f"{source}_ensemble_cycle_{cycle:03d}.nc"

    # Process member_00 through the pipeline once to discover the post-pipeline
    # structure (variables, dims, dtypes).  The result is reused for the first
    # timestep so member_00 is not processed twice.
    with xr.open_dataset(first_member_paths[0]) as ds:
        context = ProcessingContext(
            member=0,
            cycle=cycle,
            input_file=first_member_paths[0],
            output_file=Path("/dev/null"),
            config=exp.cfg,
        )
        template_ds = pipeline.process(ds, context)

    template = get_structure_from_xarray(template_ds, reference_time=reference_time)

    log_files = f"{mean_path.name}, {sd_path.name}"
    if ensemble_path:
        log_files += f", {ensemble_path.name}"
    logger.info(f"Creating output files: {log_files}")

    # Build ensemble context manager
    if keep_per_member:
        assert ensemble_path is not None
        ensemble_template = _build_ensemble_template(
            template, n_members, exp.cfg.postprocess
        )
        ensemble_ctx = _create_ensemble_writer(
            ensemble_path, ensemble_template, n_members, exp.cfg.postprocess
        )
    else:
        ensemble_ctx = nullcontext()

    with (
        _create_writer(mean_path, template, exp.cfg.postprocess) as mean_writer,
        _create_writer(sd_path, template, exp.cfg.postprocess) as sd_writer,
        ensemble_ctx as ensemble_writer,
    ):
        # Process first timestep, reusing the template result for member_00
        first_accumulators, time_val = process_members_for_timestep(
            first_member_paths,
            pipeline,
            cycle,
            exp.cfg,
            ensemble_writer=ensemble_writer,
            precomputed_first_member=template_ds,
        )
        del template_ds
        means, stddevs = finalize_accumulators(first_accumulators)
        mean_writer.append_timestep(means, time_val)
        sd_writer.append_timestep(stddevs, time_val)

        # Process remaining timesteps
        for wrfout_file in member_00_files[1:]:
            filename = wrfout_file.name
            logger.info(f"Processing timestep: {filename}")

            # Collect paths for all members
            member_paths = [
                scratch_dir / f"member_{i:02d}" / filename for i in range(n_members)
            ]

            # Process all members, accumulate statistics
            accumulators, time_val = process_members_for_timestep(
                member_paths,
                pipeline,
                cycle,
                exp.cfg,
                ensemble_writer=ensemble_writer,
            )

            # Finalize and write this timestep
            means, stddevs = finalize_accumulators(accumulators)
            mean_writer.append_timestep(means, time_val)
            sd_writer.append_timestep(stddevs, time_val)

    logger.info(f"Completed {source} processing for cycle {cycle}")
    return mean_path, sd_path, ensemble_path


def process_cycle_single_member(
    exp: Experiment,
    cycle: int,
    pipeline: ProcessorPipeline,
    source: Literal["forecast", "analysis"],
    only_last_timestep: bool = False,
) -> Path | None:
    """
    Process a cycle with only one ensemble member.

    When n_members=1, we skip statistics computation and just write
    the processed data directly as the "mean" (no standard deviation).

    Args:
        exp: Experiment object with configuration and paths.
        cycle: Cycle number to process.
        pipeline: Configured processor pipeline.
        source: Either "forecast" or "analysis".
        only_last_timestep: If True, only process the last timestep.

    Returns:
        Path to output file, or None if no files to process.
    """
    # Determine paths
    if source == "forecast":
        scratch_dir = exp.paths.scratch_forecasts_path(cycle)
        output_dir = exp.paths.forecast_path(cycle)
    else:
        scratch_dir = exp.paths.scratch_analysis_path(cycle)
        output_dir = exp.paths.analysis_path(cycle)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all timesteps
    member_00_dir = scratch_dir / "member_00"
    if not member_00_dir.exists():
        logger.warning(f"Member 00 directory not found: {member_00_dir}")
        return None

    wrfout_files = sorted(member_00_dir.glob("wrfout_d01_*"))

    # Filter out cycle start time
    cycle_start_str = exp.cycles[cycle].start.strftime("%Y-%m-%d_%H:%M:%S")
    wrfout_files = [f for f in wrfout_files if cycle_start_str not in f.name]

    if not wrfout_files:
        logger.warning(f"No wrfout files found in {member_00_dir}")
        return None

    if only_last_timestep:
        cycle_end_str = exp.cycles[cycle].end.strftime("%Y-%m-%d_%H:%M:%S")
        wrfout_files = [f for f in wrfout_files if cycle_end_str in f.name]

    if not wrfout_files:
        return None

    logger.info(
        f"Processing {len(wrfout_files)} timesteps for {source}, cycle {cycle} (single member)"
    )

    # Process first file to get template
    first_file = wrfout_files[0]
    with xr.open_dataset(first_file) as ds:
        context = ProcessingContext(
            member=0,
            cycle=cycle,
            input_file=first_file,
            output_file=Path("/dev/null"),
            config=exp.cfg,
        )
        template_ds = pipeline.process(ds, context)

    # Use cycle start time as reference for time coordinate
    reference_time = np.datetime64(exp.cycles[cycle].start.replace(tzinfo=None))
    template = get_structure_from_xarray(template_ds, reference_time=reference_time)

    # Create output file (only mean, no sd for single member)
    mean_path = output_dir / f"{source}_mean_cycle_{cycle:03d}.nc"

    logger.info(f"Creating output file: {mean_path.name}")

    # Squeeze time dim from the template (single timestep per file)
    first_time_val = template_ds["t"].values
    template_ds = template_ds.squeeze("t", drop=False)

    with _create_writer(mean_path, template, exp.cfg.postprocess) as writer:
        # Write first timestep (already processed)
        data = {var: template_ds[var].values for var in template_ds.data_vars}
        writer.append_timestep(data, first_time_val)

        # Process remaining timesteps
        for wrfout_file in wrfout_files[1:]:
            logger.info(f"Processing timestep: {wrfout_file.name}")

            with xr.open_dataset(wrfout_file) as ds:
                context = ProcessingContext(
                    member=0,
                    cycle=cycle,
                    input_file=wrfout_file,
                    output_file=Path("/dev/null"),
                    config=exp.cfg,
                )
                processed = pipeline.process(ds, context)

                time_val = processed["t"].values
                processed = processed.squeeze("t", drop=False)
                data = {var: processed[var].values for var in processed.data_vars}
                writer.append_timestep(data, time_val)

    logger.info(f"Completed {source} processing for cycle {cycle}")
    return mean_path
