import itertools
import shutil
import string
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


def call_external_process(
    command: Sequence[str | Path],
    cwd: Path = Path.cwd(),
    log_filename: Optional[str] = None,
) -> ExternalProcessResult:
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
        command = [str(command[0].resolve()), *command[1:]]

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


def int_to_letter_numeral(i: int) -> str:
    """
    Converts the given integer to a letter numeral, This is used by WPS when ungribing files (e.g., GRIBFILE.AAB), ...

    The first file (1) would be A, the second would be B, and so on. After Z, the next file would be BA, then BB, and so on.
    """

    if i < 1 or i > 17576:
        raise ValueError("i must be between 1 and 704 (AAA and ZZZ)")

    letters = list(itertools.product(string.ascii_uppercase, repeat=3))[i - 1]

    return "".join(letters).rjust(3, "A")


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


def seconds_to_pretty_hours(seconds: int | float) -> str:
    """
    Converts seconds to hours minutes in the `HHh MMm` format
    """

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60

    return f"{hours:.0f}h {minutes:02.0f}m"
