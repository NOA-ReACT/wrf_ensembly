"""
Names of the files the model writes, for the code that is the same for WRF and ICON
(postprocessing, status checks).
"""

import datetime as dt

from wrf_ensembly import config
from wrf_ensembly.icon import namelists as icon_namelists


def forecast_glob(cfg: config.Config) -> str:
    """Pattern that matches the forecast output files in a member's scratch directory"""

    if cfg.is_icon:
        return f"{icon_namelists.FORECAST_STREAM}_*.nc"
    return "wrfout_d01_*"


def time_stamp(cfg: config.Config, time: dt.datetime) -> str:
    """How a time appears in the name of a forecast output file"""

    if cfg.is_icon:
        return icon_namelists.file_date(time)
    return time.strftime("%Y-%m-%d_%H:%M:%S")


def forecast_file_name(cfg: config.Config, time: dt.datetime) -> str:
    """Name of the forecast output file for a time"""

    if cfg.is_icon:
        return f"{icon_namelists.FORECAST_STREAM}_{time_stamp(cfg, time)}.nc"
    return f"wrfout_d01_{time_stamp(cfg, time)}"


def cycle_end_file_name(cfg: config.Config, time: dt.datetime) -> str:
    """
    Name of the file that a member must have written at the end of a cycle for the
    filter and the next cycle: the wrfout for WRF, the first guess for ICON.
    """

    if cfg.is_icon:
        return f"{icon_namelists.FIRST_GUESS_STREAM}_{time_stamp(cfg, time)}.nc"
    return forecast_file_name(cfg, time)
