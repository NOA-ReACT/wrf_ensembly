from pathlib import Path

import typer

from wrf_ensembly.utils import call_external_process, ExternalProcessResult
from wrf_ensembly.console import logger


def average(input_files: list[Path], output_file: Path) -> ExternalProcessResult:
    """Average a set of netCDF files using NCO"""
    # Call the ncra command using call_external_command from utils.py
    res = call_external_process(
        ["nces", *[str(x.resolve()) for x in input_files], str(output_file.resolve())]
    )
    if not res.success:
        logger.error("Failed to compute average w/ NCO (nces)")
        logger.error(res.stderr)
        raise typer.Exit(1)


def standard_deviation(
    input_files: list[Path], output_file: Path
) -> ExternalProcessResult:
    """Calculate the standard deviation of a set of netCDF files using NCO"""

    res = call_external_process(
        [
            "nces",
            "-y",
            "rmssdn",
            *[str(x.resolve()) for x in input_files],
            str(output_file.resolve()),
        ]
    )
    if not res.success:
        logger.error("Failed to compute stddev w/ NCO (nces)")
        logger.error(res.stderr)
        raise typer.Exit(1)
