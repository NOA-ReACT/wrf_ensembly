"""
Functions related to running external commands
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from wrf_ensembly.console import logger


@dataclass
class ExternalProcess:
    """Represents a command to run in parallel"""

    command: Sequence[str | Path]
    cwd: Path = Path(os.getcwd())
    log_filename: Optional[str] = None
    stdin: Optional[str] = None


@dataclass
class ExternalProcessResult:
    """Represents a command that has been run in parallel"""

    command: Sequence[str]
    cwd: Path
    returncode: int
    output: str
    log_filename: Optional[str] = None


class ExternalProcessFailed(Exception):
    """Raise this exception when an external command fails and we cannot handle the issue"""

    res: ExternalProcessResult

    def __init__(self, res: ExternalProcessResult):
        self.res = res

    def __str__(self):
        return (
            f"Command {self.res.command} failed with return code {self.res.returncode}"
        )


def run(proc: ExternalProcess):
    """
    Runs a command and returns the output
    """

    command = []
    for c in proc.command:
        if isinstance(c, Path):
            command.append(str(c.resolve()))
        else:
            command.append(c)

    logger.debug(f"Running command: {' '.join(command)}")

    p = subprocess.run(
        command,
        cwd=proc.cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        input=proc.stdin,
    )
    if p.returncode != 0:
        logger.error(f"Command {proc.command} failed with return code {p.returncode}")
        logger.error(p.stdout)

    # Write stdout/err to this command's log directory
    if proc.log_filename is not None:
        logger.write_log_file(proc.log_filename, p.stdout)

    return ExternalProcessResult(
        command=command,
        cwd=proc.cwd,
        returncode=p.returncode,
        output=p.stdout,
    )


def runc(
    command: Sequence[str | Path],
    cwd: Path = Path(os.getcwd()),
    log_filename: Optional[str] = None,
):
    """Brief form of run() in case you don't need the ExternalProcess object"""
    return run(ExternalProcess(command, cwd, log_filename))


def run_in_parallel(
    commands: list[ExternalProcess],
    max_processes: int = 1,
    stop_on_failure: bool = False,
):
    """
    Runs a list of commands in parallel, with a maximum of `max_processes` processes running at the same time.
    If stop_on_failure is True, cancels remaining jobs when the first failure occurs.
    """

    with ThreadPoolExecutor(max_workers=max_processes) as executor:
        futures = [executor.submit(run, command) for command in commands]

        try:
            for future in as_completed(futures):
                res = future.result()
                str_cmd = " ".join(res.command)

                if res.returncode != 0:
                    logger.error(
                        f"Command {str_cmd} failed with return code {res.returncode}"
                    )
                    if stop_on_failure:
                        # Cancel all remaining futures
                        for f in futures:
                            f.cancel()
                        yield res
                        return
                else:
                    logger.debug(
                        f"Command {str_cmd} finished with return code {res.returncode}"
                    )
                yield res
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, cancelling remaining jobs...")
            # Cancel all futures
            for f in futures:
                f.cancel()
            raise


def assert_all_successful(results: list[ExternalProcessResult]):
    """
    Assert that all commands in the list of results were successful
    """

    for res in results:
        if res.returncode != 0:
            sys.exit(1)
