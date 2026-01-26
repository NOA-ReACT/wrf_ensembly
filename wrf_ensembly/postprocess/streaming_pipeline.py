"""
Streaming postprocessing pipeline for ensemble WRF output.

This module implements a memory-efficient streaming approach that processes
ensemble members one at a time, computes statistics on-the-fly using Welford's
algorithm, and writes results directly to final output files without intermediate
files.
"""

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import xarray as xr

from wrf_ensembly.config import Config
from wrf_ensembly.console import logger
from wrf_ensembly.experiment import Experiment
from wrf_ensembly.postprocess.streaming_writer import StreamingNetCDFWriter
from wrf_ensembly.processors import ProcessingContext, ProcessorPipeline
from wrf_ensembly.statistics import (
    WelfordState,
    create_welford_accumulators,
    finalize_accumulators,
    get_structure_from_xarray,
    update_accumulators_from_dataset,
)


def process_members_for_timestep(
    wrfout_paths: list[Path],
    pipeline: ProcessorPipeline,
    cycle: int,
    config: Config,
) -> tuple[dict[str, WelfordState], xr.Dataset]:
    """
    Process all ensemble members for a single output timestep.

    Opens each member's wrfout file, applies the processor pipeline,
    and accumulates statistics using Welford's algorithm.

    Args:
        wrfout_paths: List of paths to wrfout files, one per member.
        pipeline: Configured processor pipeline to apply.
        cycle: Current cycle number.
        config: Experiment configuration.

    Returns:
        Tuple of (accumulators, last_processed_dataset).
        The last dataset is returned to provide template/coordinate information.
    """
    accumulators: Optional[dict[str, WelfordState]] = None
    last_processed: Optional[xr.Dataset] = None

    for member_i, wrfout_path in enumerate(wrfout_paths):
        if not wrfout_path.exists():
            logger.warning(f"Member file not found, skipping: {wrfout_path}")
            continue

        logger.debug(f"Processing member {member_i}: {wrfout_path.name}")

        # Load and process through pipeline
        with xr.open_dataset(wrfout_path) as ds:
            context = ProcessingContext(
                member=member_i,
                cycle=cycle,
                input_file=wrfout_path,
                output_file=Path("/dev/null"),  # Not used in streaming mode
                config=config,
            )
            processed = pipeline.process(ds, context)

            # Initialize accumulators from first member's structure
            if accumulators is None:
                accumulators = create_welford_accumulators(processed)

            # Update running statistics
            update_accumulators_from_dataset(accumulators, processed)
            last_processed = processed

    if accumulators is None:
        raise ValueError("No member files were successfully processed")

    return accumulators, last_processed


def process_cycle_streaming(
    exp: Experiment,
    cycle: int,
    pipeline: ProcessorPipeline,
    source: Literal["forecast", "analysis"],
    only_last_timestep: bool = False,
) -> Optional[tuple[Path, Path]]:
    """
    Process an entire cycle in streaming fashion.

    For each output timestep:
    1. Process all members through the pipeline (one at a time)
    2. Accumulate statistics using Welford's algorithm
    3. Append results directly to output files

    Args:
        exp: Experiment object with configuration and paths.
        cycle: Cycle number to process.
        pipeline: Configured processor pipeline.
        source: Either "forecast" or "analysis".
        only_last_timestep: If True, only process the last timestep of the cycle.

    Returns:
        Tuple of (mean_path, sd_path) or None if no files to process.
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
    first_accumulators, template_ds = process_members_for_timestep(
        first_member_paths, pipeline, cycle, exp.cfg
    )

    # Extract template structure for output files
    # Use cycle start time as reference for time coordinate
    reference_time = np.datetime64(exp.cycles[cycle].start.replace(tzinfo=None))
    template = get_structure_from_xarray(template_ds, reference_time=reference_time)

    # Create output files
    mean_path = output_dir / f"{source}_mean_cycle_{cycle:03d}.nc"
    sd_path = output_dir / f"{source}_sd_cycle_{cycle:03d}.nc"

    logger.info(f"Creating output files: {mean_path.name}, {sd_path.name}")

    with (
        StreamingNetCDFWriter(mean_path, template) as mean_writer,
        StreamingNetCDFWriter(sd_path, template) as sd_writer,
    ):
        # Write first timestep (already processed)
        means, stddevs = finalize_accumulators(first_accumulators)
        time_val = template_ds["t"].values
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
            accumulators, last_ds = process_members_for_timestep(
                member_paths, pipeline, cycle, exp.cfg
            )

            # Finalize and write this timestep
            means, stddevs = finalize_accumulators(accumulators)
            time_val = last_ds["t"].values
            mean_writer.append_timestep(means, time_val)
            sd_writer.append_timestep(stddevs, time_val)

    logger.info(f"Completed {source} processing for cycle {cycle}")
    return mean_path, sd_path


def process_cycle_single_member(
    exp: Experiment,
    cycle: int,
    pipeline: ProcessorPipeline,
    source: Literal["forecast", "analysis"],
    only_last_timestep: bool = False,
) -> Optional[Path]:
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

    with StreamingNetCDFWriter(mean_path, template) as writer:
        # Write first timestep (already processed)
        data = {var: template_ds[var].values for var in template_ds.data_vars}
        time_val = template_ds["t"].values
        writer.append_timestep(data, time_val)

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

                data = {var: processed[var].values for var in processed.data_vars}
                time_val = processed["t"].values
                writer.append_timestep(data, time_val)

    logger.info(f"Completed {source} processing for cycle {cycle}")
    return mean_path
