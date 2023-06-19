from dataclasses import dataclass
from logging import Logger
from pathlib import Path
import subprocess


def rm_tree(p: Path):
    """
    Removes a directory tree recursively.
    Only handles files, symbolic links, and directories.

    Args:
        p: Path to remove
    """

    if p.is_file() or p.is_symlink():
        p.unlink()
    else:
        for child in p.iterdir():
            rm_tree(child)
        p.rmdir()


@dataclass
class ExternalProcessResult:
    """
    Represents the result of an external process call
    """

    returncode: int
    """Return code of the process"""

    success: bool
    """Whether the process was successful (returncode == 0)"""

    stdout: str
    """Standard output of the process"""

    stderr: str
    """Standard error of the process"""


def call_external_process(
    command: list[str], cwd: Path, logger: Logger, log_failure=True
):
    """
    Calls an external process and handles failures gracefully.

    Args:
        command: Command to run as an array of strings
        cwd: Working directory to run the command in
        log_failure: Whether to log a failure message if the process fails, defaults to True

    Returns:
        ExternalProcessResult object containing the result of the process (stdout, stderr, code)
    """

    command_str = " ".join(map(str, command))
    logger.debug(f"Calling external process: {command_str}")

    if isinstance(command[0], Path):
        command[0] = str(command[0].resolve())

    proc = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0 and log_failure:
        logger.error(
            f"External process failed with return code {proc.returncode}: {command_str}"
        )
        if len(proc.stdout) > 0:
            logger.error(f"stdout:\n{proc.stdout.strip()}")

    return ExternalProcessResult(
        returncode=proc.returncode,
        success=proc.returncode == 0,
        stdout=proc.stdout,
        stderr=proc.stderr,
    )


def int_to_letter_numeral(i: int, length=3) -> str:
    """
    Converts the given integer to a letter numeral, similar to how excel names the rows. For example, 1 becomes "A", 27 becomes "AA", etc.

    This is used by WPS when ungribing files (e.g., GRIBFILE.AAB), ...

    Args:
        i: Integer to convert
        length: Length of the output string, defaults to 3
    """

    letters = []
    while i > 0:
        i, remainder = divmod(i - 1, 26)
        letters.append(chr(ord("A") + remainder))

    return "".join(reversed(letters)).rjust(length, "A")
