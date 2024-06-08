from pathlib import Path

from wrf_ensembly.external import ExternalProcess


def average(input_files: list[Path], output_file: Path) -> ExternalProcess:
    """
    Average a set of netCDF files using NCO
    You need to execute the returned object to run the command using external.run()
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    return ExternalProcess(
        [
            "nces",
            "-4",
            *[str(x.resolve()) for x in input_files],
            str(output_file.resolve()),
        ]
    )


def standard_deviation(input_files: list[Path], output_file: Path) -> ExternalProcess:
    """
    Calculate the standard deviation of a set of netCDF files using NCO
    You need to execute the returned object to run the command using external.run()
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    return ExternalProcess(
        [
            "nces",
            "-4",
            "-y",
            "rmssdn",
            *[str(x.resolve()) for x in input_files],
            str(output_file.resolve()),
        ]
    )


def concatenate(
    input_files: list[Path], output_file: Path, args: list[str] = []
) -> ExternalProcess:
    """
    Concatenate a set of netCDF files using NCO
    You need to execute the returned object to run the command using external.run()

    Args:
        input_files: List of files to concatenate
        output_file: Where to write the output
        args: Additional arguments to pass to ncrcat (e.g., compression)
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    return ExternalProcess(
        [
            "ncrcat",
            "-4",
            *args,
            *[str(x.resolve()) for x in input_files],
            str(output_file.resolve()),
        ]
    )
