"""
Handles interactions with the "update_wrf_bc" executable that updates the boundary conditions
to match the initial conditions, after they are modified by cycling or applying perturbations.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from wrf_ensembly import utils
from wrf_ensembly.config import Config
from wrf_ensembly.external import ExternalProcess, run


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
    It will be linked in a temp. directory so that you can run this function in parallel.

    Args:
        cfg: The configuration object.
        wrfinput: The wrfinput file to update from.
        wrfbdy: The wrfbdy file to update. Will be mutated.
        log_dir: The directory to log the stdout and stderr of the process to.
        log_filename: The filename inside `log_dir` to log the stdout and stderr

    Returns:
        The result of the external process call (see `ExternalProcessResult`).
    """

    with TemporaryDirectory(prefix="wrf_ensembly_update_bc") as work_dir:
        work_dir = Path(work_dir)

        # Link input files
        if not wrfinput.exists():
            raise FileNotFoundError(f"wrfinput file not found: {wrfinput}")
        if not wrfbdy.exists():
            raise FileNotFoundError(f"wrfbdy file not found: {wrfbdy}")

        sym_wrfinput = work_dir / "wrfinput_d01"
        sym_wrfbdy = work_dir / "wrfbdy_d01"
        sym_wrfinput.symlink_to(wrfinput.resolve())
        sym_wrfbdy.symlink_to(wrfbdy.resolve())

        # Link executable
        bin_path = (
            cfg.directories.dart_root / "models" / "wrf" / "work" / "update_wrf_bc"
        )
        if not bin_path.is_file():
            raise FileNotFoundError(f"update_wrf_bc executable not found in {bin_path}")

        command = work_dir / "update_wrf_bc"
        command.symlink_to(bin_path.resolve())

        # Copy namelist
        utils.copy(
            cfg.directories.dart_root / "models" / "wrf" / "work" / "input.nml",
            work_dir / "input.nml",
        )

        res = run(
            ExternalProcess(
                [str(command.resolve())],
                cwd=work_dir,
                log_filename=log_filename,
            )
        )

    return res
