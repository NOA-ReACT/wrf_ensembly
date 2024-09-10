"""Routines about postprocessing of `wrfout` files"""

from pathlib import Path

import xarray as xr
import xwrf  # noqa: F401


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

    with xr.open_dataset(input_file) as ds:
        ds = ds.xwrf.postprocess()

        # Since the projection object is not serialisable, we need to drop it before saving
        ds = ds.drop_vars("wrf_projection")

        ds.to_netcdf(output_path)
