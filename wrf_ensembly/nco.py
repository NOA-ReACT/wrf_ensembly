from pathlib import Path

from wrf_ensembly.external import ExternalProcess


def concatenate(
    ncrcat_cmd: str, input_files: list[Path], output_file: Path, args: list[str] = []
) -> ExternalProcess:
    """
    Concatenate a set of netCDF files using NCO
    You need to execute the returned object to run the command using external.run()

    Args:
        ncrcat_cmd: Path to the ncrcat command to use
        input_files: List of files to concatenate
        output_file: Where to write the output
        args: Additional arguments to pass to ncrcat (e.g., compression)
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    return ExternalProcess(
        [
            *ncrcat_cmd.split(" "),
            "-4",
            "-O",
            *args,
            *[str(x.resolve()) for x in input_files],
            str(output_file.resolve()),
        ]
    )
