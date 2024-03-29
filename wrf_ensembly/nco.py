from pathlib import Path

from wrf_ensembly.console import logger
from wrf_ensembly.external import ExternalProcess


def average(input_files: list[Path], output_file: Path) -> ExternalProcess:
    """
    Average a set of netCDF files using NCO
    You need to execute the returned object to run the command using external.run()
    """

    return ExternalProcess(
        [
            "nces",
            *[str(x.resolve()) for x in input_files],
            str(output_file.resolve()),
        ]
    )


def standard_deviation(input_files: list[Path], output_file: Path) -> ExternalProcess:
    """
    Calculate the standard deviation of a set of netCDF files using NCO
    You need to execute the returned object to run the command using external.run()
    """

    return ExternalProcess(
        [
            "nces",
            "-y",
            "rmssdn",
            *[str(x.resolve()) for x in input_files],
            str(output_file.resolve()),
        ]
    )
