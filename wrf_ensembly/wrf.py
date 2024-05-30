from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from wrf_ensembly import fortran_namelists
from wrf_ensembly.config import Config
from wrf_ensembly.console import logger
from wrf_ensembly.cycling import CycleInformation
from wrf_ensembly.experiment.paths import ExperimentPaths

ESSENTIAL_VARIABLES = set(
    [
        "XTIME",
        "XLAT",
        "XLAT_U",
        "XLAT_V",
        "XLONG",
        "XLONG_U",
        "XLONG_V",
        "Time",
        "ZNU",
        "ZNW",
        "PH",
        "PHB",
        "PC",
        "P",
        "PB",
        "FNM",
        "FNP",
        "DN",
        "HGT",
        "P_TOP",
        "T00",
        "P00",
        "VAR",
        "VAR_SSO",
    ]
)


def datetime_to_namelist_items(dt: datetime, prefix: str) -> dict[str, int]:
    """
    Converts a datetime to a set of namelist items, as required by WRF.

    Args:
        dt: The datetime to convert
        prefix: Which prefix to use for the namelist items (e.g. "start" or "end")

    Returns:
        The converted namelist items in a dictionary
    """

    return {
        f"{prefix}_year": dt.year,
        f"{prefix}_month": dt.month,
        f"{prefix}_day": dt.day,
        f"{prefix}_hour": dt.hour,
        f"{prefix}_minute": dt.minute,
        f"{prefix}_second": dt.second,
    }


def timedelta_to_namelist_items(td: timedelta, prefix: str = "run") -> dict[str, int]:
    """
    Converts a timedelta to a set of namelist items, as required by WRF
    (for example, the `run_*` items).

    Args:
        td: The timedelta to convert
        prefix: Prefix of items, defaults to "run".

    Returns:
        The converted namelist items in a dictionary
    """

    return {
        f"{prefix}_days": td.days,
        f"{prefix}_hours": td.seconds // 3600,
        f"{prefix}_minutes": (td.seconds // 60) % 60,
        f"{prefix}_seconds": td.seconds % 60,
    }


def generate_wps_namelist(cfg: Config, path: Path):
    """
    Generates the WPS namelist for the experiment, at the given path.
    """

    wps_namelist = {
        "share": {
            "wrf_core": "ARW",
            "max_dom": 1,
            "start_date": cfg.time_control.start.strftime("%Y-%m-%d_%H:%M:%S"),
            "end_date": cfg.time_control.end.strftime("%Y-%m-%d_%H:%M:%S"),
            "interval_seconds": cfg.time_control.boundary_update_interval * 60,
        },
        "geogrid": {
            "parent_id": 1,
            "parent_grid_ratio": 1,
            "i_parent_start": 1,
            "j_parent_start": 1,
            "e_we": cfg.domain_control.xy_size[0],
            "e_sn": cfg.domain_control.xy_size[1],
            "geog_data_res": "default",
            "dx": cfg.domain_control.xy_resolution[0] * 1000,
            "dy": cfg.domain_control.xy_resolution[1] * 1000,
            "map_proj": cfg.domain_control.projection,
            "ref_lat": cfg.domain_control.ref_lat,
            "ref_lon": cfg.domain_control.ref_lon,
            "truelat1": cfg.domain_control.truelat1,
            "truelat2": cfg.domain_control.truelat2,
            "stand_lon": cfg.domain_control.stand_lon,
            "geog_data_path": cfg.data.wps_geog.resolve(),
        },
        "ungrib": {
            "out_format": "WPS",
            "prefix": "FILE",
        },
        "metgrid": {
            "fg_name": "FILE",
            "io_form_metgrid": 2,
        },
    }

    if path.is_dir():
        path = path / "namelist.wps"

    fortran_namelists.write_namelist(wps_namelist, path)


def generate_wrf_namelist(
    cfg: Config,
    cycle: CycleInformation,
    chem_in_opt: bool,
    path: Path,
    member: Optional[int] = None,
    paths: Optional[ExperimentPaths] = None,
):
    """
    Generates the WRF namelist for the experiment and a specific cycle, at the given path.

    Args:
        experiment: The experiment config object
        cycle: The cycle to generate the namelist for
        chem_in_opt: If true, chem_in_opt will be set to 1, otherwise 0. Use False when running real.exe and True
                     when running wrf.exe. Ignored if cfg.data.manage_chem_in is False.
        paths: The path to write the namelist to. If it points to a directory,
               the namelist will be written inside that directory with the name
               `namelist.input`.
        member: The ensemble member to generate the namelist for. If set, the &time_control.history_outname
                variable will be set to the member's scratch directory and any overrides in the configuration
                will be applied. Omit this parameter when generating a namelist for preprocessing/real.exe.
        paths: Paths of the experiment, required if member is set.
    """

    if member is not None and paths is None:
        raise ValueError(
            "paths must be provided when generating a member-specific namelist"
        )

    # Determine start/end times
    start = cycle.start
    end = cycle.end

    # Add time and domain control
    wrf_namelist = {
        "time_control": {
            **timedelta_to_namelist_items(end - start),
            **datetime_to_namelist_items(start, "start"),
            **datetime_to_namelist_items(end, "end"),
            "interval_seconds": cfg.time_control.boundary_update_interval * 60,
            "history_interval": (
                cycle.output_interval
                if cycle.output_interval is not None
                else cfg.time_control.output_interval
            ),
            "history_outname": "wrfout_d<domain>_<date>",
        },
        "domains": {
            "e_we": cfg.domain_control.xy_size[0],
            "e_sn": cfg.domain_control.xy_size[1],
            "dx": cfg.domain_control.xy_resolution[0] * 1000,
            "dy": cfg.domain_control.xy_resolution[1] * 1000,
            "grid_id": 1,
            "parent_id": 0,
            "max_dom": 1,
        },
    }
    if member is not None and paths is not None:
        wrfout_dest = paths.scratch_forecasts_path(cycle.index, member)
        wrfout_dest.mkdir(parents=True, exist_ok=True)
        wrf_namelist["time_control"][
            "history_outname"
        ] = f"{wrfout_dest}/wrfout_d<domain>_<date>"

    # Add overrides
    for name, group in cfg.wrf_namelist.items():
        if name in wrf_namelist:
            wrf_namelist[name] |= group
        else:
            wrf_namelist[name] = group
    if str(member) in cfg.wrf_namelist_per_member:
        member_group = cfg.wrf_namelist_per_member[str(member)]
        for name, group in member_group.items():
            if name in wrf_namelist:
                wrf_namelist[name] |= group
            else:
                wrf_namelist[name] = group

    # Handle chem_in_opt
    if cfg.data.manage_chem_ic:
        if "chem" in wrf_namelist:
            wrf_namelist["chem"]["chem_in_opt"] = 1 if chem_in_opt else 0
        else:
            wrf_namelist["chem"] = {"chem_in_opt": 1 if chem_in_opt else 0}
    else:
        logger.warning("!!! manage_chem_ic is set to False !!!")

    # Write namelist(s)
    if path.is_dir():
        path = path / "namelist.input"
    fortran_namelists.write_namelist(wrf_namelist, path)
    logger.info(f"Wrote namelist to {path}")
