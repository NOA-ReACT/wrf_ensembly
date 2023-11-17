"""
Handles interactions with the "update_wrf_bc" executable that updates the boundary conditions
to match the initial conditions, that might be modified.
"""

from pathlib import Path
from typing import Optional

from wrf_ensembly.config import Config
from wrf_ensembly.utils import call_external_process


def update_wrf_bc(
    cfg: Config,
    wrfinput: Path,
    wrfbdy: Path,
    log_filename: Optional[str] = None,
):
    """
    Updates the given `wrfbdy` file to match the `wrfinput` file, using `update_wrf_bc`.
    Required if you have modified the `wrfinput` file.

    The `update_wrf_bc` executable is expected to be found inside the DART work directory.
    ! Any existing `wrfinput`/`wrfbdy` files in the work dir will be overwritten !

    Args:
        cfg: The configuration object.
        wrfinput: The wrfinput file to update from.
        wrfbdy: The wrfbdy file to update. Will be mutated.
        log_dir: The directory to log the stdout and stderr of the process to.
        log_filename: The filename inside `log_dir` to log the stdout and stderr

    Returns:
        The result of the external process call (see `ExternalProcessResult`).
    """

    work_dir = cfg.directories.dart_root / "models" / "wrf" / "work"
    sym_wrfinput = work_dir / "wrfinput_d01"
    sym_wrfbdy = work_dir / "wrfbdy_d01"

    sym_wrfinput.unlink(missing_ok=True)
    sym_wrfbdy.unlink(missing_ok=True)

    sym_wrfinput.symlink_to(wrfinput.resolve())
    sym_wrfbdy.symlink_to(wrfbdy.resolve())

    command = work_dir / "update_wrf_bc"
    if not command.is_file():
        raise RuntimeError(
            "update_wrf_bc executable not found in DART/wrf/work directory"
        )

    res = call_external_process(
        [str(command.resolve())],
        cwd=work_dir,
        log_filename=log_filename,
    )
    return res
