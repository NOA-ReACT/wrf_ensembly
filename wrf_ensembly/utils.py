from dataclasses import dataclass
import logging
import shutil
from pathlib import Path
import subprocess

from wrf_ensembly.console import logger


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


def call_external_process(command: list[str], cwd: Path, log_filename: str = None):
    """
    Calls an external process and handles failures gracefully.

    Args:
        command: Command to run as an array of strings
        cwd: Working directory to run the command in
        log_filename: Filename to log the stdout and stderr to, inside logger's directory

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
    if proc.returncode != 0:
        logger.error(
            f"External process failed with return code {proc.returncode}: {command_str}"
        )
        if len(proc.stdout) > 0:
            logger.error(f"stdout:\n{proc.stdout.strip()}")

    # Write stdout/err to this command's log directory
    if log_filename is None:
        if isinstance(command[0], Path):
            log_filename = str(command[0].resolve().name) + ".log"
        else:
            log_filename = command[0].split("/")[-1] + ".log"
    logger.write_log_file(log_filename, proc.stdout)

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


def copy(src: Path, dest: Path, ensure_dest_parent_exists=True):
    """
    Copies the file from `src` to `dest` using `shutil.copy` and logs the operation.

    Args:
        src: Source file
        dest: Destination file.
        ensure_dest_parent_exists: Whether to create the parent directory of `dest` if it doesn't exist.
    """

    logger.debug(f"Copying {src} to {dest}")
    if ensure_dest_parent_exists:
        dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)


def filter_none_from_dict(dict: dict) -> dict:
    """
    Returns a new dictionary that does not contain any keys with a value of None.

    Args:
        dict: Dictionary to filter
    """

    return {k: v for k, v in dict.items() if v is not None}


def seconds_to_pretty_hours(seconds: int) -> str:
    """
    Converts seconds to hours minutes in the `HHh MMm` format
    """

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60

    return f"{hours:.0f}h {minutes:02.0f}m"
