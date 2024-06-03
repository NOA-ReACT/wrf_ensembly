import itertools
import shutil
import string
from contextlib import contextmanager
from pathlib import Path
import time

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


@contextmanager
def atomic_binary_open(path: Path, mode="wb"):
    """
    Opens a file in atomic mode, which means that the file is first opened in a temporary
    location and then moved to the final location after the file is closed.

    This function will attempt to create the temp file in the same directory but with a
    random prefix. So the process/user must have the appropriate permissions to create
    files in the same directory.

    Args:
        path: Path to the file
        mode: Mode to open the file in, default 'wb'
    """

    # Create a temp file
    tmp_file = path.parent / (path.name + ".tmp")
    with open(tmp_file, mode) as f:
        yield f

    # If the target exists, remove it
    if path.exists():
        path.unlink()

    # Move the temp file to the target location
    tmp_file.rename(path)


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


def bool_to_console_str(b: bool):
    """
    Converts a boolean to a string that can be printed to the console.
    """

    return "✅" if b else "❌"


class LockFile:
    """
    A simple lock file that can be used to prevent multiple processes from running the same code at the same time. It is implemented by creating a second file with the .lock suffix.

    Args:
        path: Path to the lock file
    """

    path: Path
    """The file to lock"""

    lockfile: Path
    """Path to the lockfile (.lock)"""

    timeout: int
    """Max amount of seconds to wait for a lockfile to clear"""

    pooling_interval = 5
    """How often to pool an existing lockfile"""

    def __init__(self, path: Path, timeout=10 * 60):
        self.path = path
        self.lockfile = path.with_suffix(".lock")
        self.timeout = timeout

    def __del__(self):
        self.path.unlink(missing_ok=True)

    def __enter__(self):
        waiting_for = 0
        while self.lockfile.exists() and waiting_for <= self.timeout:
            logger.debug(f"Waiting for lockfile to be removed: {self.lockfile}")
            time.sleep(self.pooling_interval)
            waiting_for += self.pooling_interval

        logger.debug(f"Creating lockfile: {self.lockfile}")
        self.lockfile.touch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug(f"Removing lockfile: {self.lockfile}")

        if exc_val is not None:
            logger.error(f"An exception occurred: {exc_val}")
        if self.lockfile.is_file:
            self.lockfile.unlink()
