"""
Utility functions for postprocessing operations.
"""

from wrf_ensembly import external
from wrf_ensembly.console import logger
from wrf_ensembly.experiment import Experiment


def apply_compression(exp: Experiment, cycle: int) -> None:
    """
    Apply NCO compression to final output files.

    Uses ncks to recompress files with the configured compression filters
    and precision-preserving compression (PPC) settings.

    Args:
        exp: Experiment object with configuration and paths.
        cycle: Cycle number to process.
    """
    cmp_args = []
    if exp.cfg.postprocess.ppc_filter:
        cmp_args.extend(["--ppc", exp.cfg.postprocess.ppc_filter])
    if exp.cfg.postprocess.compression_filters:
        cmp_args.append(f"--cmp={exp.cfg.postprocess.compression_filters}")

    if not cmp_args:
        return

    logger.info(f"Compression args: {' '.join(cmp_args)}")

    # Get ncks command from ncrcat command (they're usually in the same location)
    ncks_cmd = exp.cfg.postprocess.ncrcat_cmd.replace("ncrcat", "ncks")

    # Find all output files for this cycle
    output_files = []
    output_files.extend(exp.paths.forecast_path(cycle).glob("*.nc"))
    output_files.extend(exp.paths.analysis_path(cycle).glob("*.nc"))

    for f in output_files:
        logger.info(f"Compressing {f.name}...")
        temp_path = f.with_suffix(".nc.tmp")

        cmd = [ncks_cmd, "-O", *cmp_args, str(f), str(temp_path)]
        res = external.runc(cmd)

        if res.returncode != 0:
            logger.error(f"ncks failed for {f}: {res.output}")
            # Clean up temp file if it exists
            temp_path.unlink(missing_ok=True)
            continue

        # Replace original with compressed version
        temp_path.rename(f)
        logger.debug(f"Compressed {f.name}")
