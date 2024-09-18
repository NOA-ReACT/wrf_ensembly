"""Routines about postprocessing of `wrfout` files"""

from pathlib import Path

import xarray as xr
import xwrf  # noqa: F401

from wrf_ensembly import external
from wrf_ensembly.console import logger


def _xwrf_post(args: tuple[Path, Path]):
    """Single argument version of `xwrf_post`"""
    xwrf_post(*args)


def xwrf_post(input_file: Path, output_path: Path):
    """
    Apply the postprocessing routines from [xWRF](https://github.com/xarray-contrib/xwrf)
    to a single WRF output file, and write the result to a new file. Specifically:
    - Make the units pint friendly
    - Rename dimensions to (t, x, y, z)
    - Destagger variables
    - Compute derived variables such as air temperature and earth-relative wind speed
    - Computes X and Y arrays in the model's projection for interpolation purposes
    """

    # TODO Fix /SerializationWarning: saving variable ISEEDARRAY_SPP_LSM with floating point data as an integer dtype without any _FillValue to use for NaNs

    with xr.open_dataset(input_file) as ds:
        ds = ds.xwrf.postprocess().xwrf.destagger()

        # Since the projection object is not serialisable, we need to drop it before saving
        ds = ds.drop_vars("wrf_projection")

        ds.to_netcdf(output_path)


def apply_scripts_to_file(
    scripts: list[str], input_file: Path, workdir: Path
) -> tuple[Path, Path]:
    """
    Apply a list of scripts to a single WRF output file, and write the result to a new file.
    """

    orig_path = input_file
    filename = input_file.name
    input_path = input_file

    for i, script in enumerate(scripts):
        output_path = workdir / f"{filename}_script_{i}"

        script = script.replace("{in}", str(input_path))
        script = script.replace("{out}", str(output_path))

        res = external.runc(script.split())
        if res.returncode != 0:
            logger.error(
                f"Command {' '.join(res.command)} failed with return code {res.returncode} and output: {res.output}"
            )
            raise external.ExternalProcessFailed(res)

        input_path = output_path

    return orig_path, input_path
