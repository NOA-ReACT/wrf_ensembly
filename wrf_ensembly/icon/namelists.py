"""
Namelists for an ICON run: `icon_master.namelist` and the atmosphere namelist.

Both are built from the `[icon]` section of the configuration, with the parts that
depend on the cycle (dates, where to start from, where to write) filled in here.
"""

import datetime as dt
from pathlib import Path
from typing import Any

from wrf_ensembly import config, cycling
from wrf_ensembly.fortran_namelists import convert_to_string

MASTER_NAMELIST = "icon_master.namelist"
ATMO_NAMELIST = "NAMELIST_atm"
LATBC_MAP_FILE = "map_file.latbc"

# Names of the files in a member's run directory. They are links, so that the
# namelist does not hold long paths and stays the same from cycle to cycle.
GRID_LINK = "grid.nc"
EXTPAR_LINK = "extpar.nc"
IFS_LINK = "ifs_initial.nc"
FIRST_GUESS_LINK = "first_guess.nc"
ANALYSIS_LINK = "analysis.nc"
ECRAD_LINK = "ecrad_data"
LATBC_LINK = "latbc"

# Output streams: the name is the start of the file name
FORECAST_STREAM = "icon"
FIRST_GUESS_STREAM = "fg"

INIT_MODE_FIRST_GUESS = 1  # ICON first guess, with or without an analysis
INIT_MODE_IFS = 2

Namelist = dict[str, dict[str, Any] | list[dict[str, Any]]]


def icon_date(time: dt.datetime) -> str:
    """A date the way ICON's namelists want it, e.g. 2026-09-13T06:00:00Z"""

    return time.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_date(time: dt.datetime) -> str:
    """A date the way ICON writes it in file names (`<datetime2>`), e.g. 20260913T060000Z"""

    return time.astimezone(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_file_date(name: str) -> dt.datetime:
    """The date in a file name like `icon_20260913T060000Z.nc`"""

    stamp = Path(name).stem.rsplit("_", 1)[-1]
    return dt.datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc)


def iso_duration(minutes: int) -> str:
    """ISO 8601 duration with two-digit fields, which is all ICON reads (PT06H, PT30M)"""

    days, rest = divmod(minutes, 24 * 60)
    hours, mins = divmod(rest, 60)
    date = f"{days:02d}D" if days else ""
    time = (f"{hours:02d}H" if hours else "") + (f"{mins:02d}M" if mins else "")
    if not date and not time:
        raise ValueError("A duration must be longer than zero")
    return "P" + date + ("T" + time if time else "")


def forecast_file(directory: Path, time: dt.datetime) -> Path:
    return directory / f"{FORECAST_STREAM}_{file_date(time)}.nc"


def first_guess_file(directory: Path, time: dt.datetime) -> Path:
    return directory / f"{FIRST_GUESS_STREAM}_{file_date(time)}.nc"


def master_namelist(cfg: config.Config, cycle: cycling.CycleInformation) -> Namelist:
    """Contents of `icon_master.namelist` for a run over the given cycle"""

    assert cfg.icon is not None
    namelist: Namelist = {
        "master_nml": {"lrestart": False},
        "master_time_control_nml": {
            "calendar": "proleptic gregorian",
            "experimentStartDate": icon_date(cycle.start),
            "experimentStopDate": icon_date(cycle.forecast_end),
            "restartTimeIntval": "",
            "checkpointTimeIntval": "",
        },
        "time_nml": {"is_relative_time": False},
        "master_model_nml": {
            "model_name": "atm",
            "model_namelist_filename": ATMO_NAMELIST,
            "model_type": 1,
            "model_min_rank": 0,
            "model_max_rank": 65535,
            "model_inc_rank": 1,
            "model_rank_group_size": 1,
        },
    }
    return merge(namelist, cfg.icon.master_namelist)


def atmosphere_namelist(
    cfg: config.Config,
    cycle: cycling.CycleInformation,
    member: int,
    output_dir: Path,
    from_first_guess: bool,
    with_analysis: bool,
) -> Namelist:
    """
    Contents of the atmosphere namelist for one member and cycle.

    Args:
        output_dir: Where the member writes its forecast and first guess files.
        from_first_guess: Start from the first guess of the previous cycle
            (`first_guess.nc` in the run directory) instead of from IFS data.
        with_analysis: Also read `analysis.nc`, the filter's analysis of the previous
            cycle. Only with `from_first_guess`.
    """

    icon = cfg.icon
    assert icon is not None
    output_interval = cycle.output_interval or cfg.time_control.output_interval

    namelist: Namelist = {
        "run_nml": {"output": "nml"},
        "grid_nml": {
            "dynamics_grid_filename": GRID_LINK,
            "dynamics_parent_grid_id": 0,
            "l_limited_area": True,
        },
        "extpar_nml": {"extpar_filename": EXTPAR_LINK},
        "limarea_nml": {
            "itype_latbc": 1,
            "dtime_latbc": float(cfg.time_control.boundary_update_interval * 60),
            "latbc_path": LATBC_LINK,
            "latbc_filename": icon.boundary_filename,
            "latbc_varnames_map_file": LATBC_MAP_FILE,
            # The state at the start of the run is the boundary data for that time
            "init_latbc_from_fg": True,
        },
        "radiation_nml": {"ecrad_data_path": f"./{ECRAD_LINK}"},
    }

    if from_first_guess:
        namelist["initicon_nml"] = {
            "init_mode": INIT_MODE_FIRST_GUESS,
            "dwdfg_filename": FIRST_GUESS_LINK,
            "dwdana_filename": ANALYSIS_LINK,
            "lread_ana": with_analysis,
            # ICON cannot tell the surface tiles apart in a netCDF first guess (that
            # needs GRIB metadata), so they start from the tile-averaged fields
            "ltile_coldstart": True,
            # The checks want the GRIB "type of generating process" of an analysis,
            # which a netCDF file cannot have
            "lconsistency_checks": False,
        }
    else:
        namelist["initicon_nml"] = {
            "init_mode": INIT_MODE_IFS,
            "ifs2icon_filename": IFS_LINK,
        }
        if icon.initial_noise > 0:
            namelist["initicon_nml"] |= {
                "pinit_seed": member + 1,
                "pinit_amplitude": icon.initial_noise,
            }

    if icon.perturbation_number:
        namelist["gribout_nml"] = {"perturbationNumber": member + 1}

    # The user's namelist goes on top of the above, except for the output streams,
    # which are added to ours
    user = {k: v for k, v in icon.namelist.items() if k != "output_nml"}
    namelist = merge(namelist, user)
    namelist = merge(
        namelist,
        {
            k: v
            for k, v in cfg.icon_namelist_overrides_for_member(member).items()
            if k != "output_nml"
        },
    )

    common = {
        "filetype": 5,  # netCDF4
        "mode": 1,  # time axis relative to the start of the run
        "steps_per_file": 1,
        "steps_per_file_inclfirst": False,  # or the first file gets two steps
        "filename_format": "<output_filename>_<datetime2>",
        "remap": 0,  # native grid
    }
    streams: list[dict[str, Any]] = [
        {
            "output_filename": str(output_dir / FORECAST_STREAM),
            "output_start": icon_date(cycle.start),
            "output_end": icon_date(cycle.forecast_end),
            "output_interval": iso_duration(output_interval),
            "ml_varlist": list(icon.output_variables),
            **common,
        },
        {
            "output_filename": str(output_dir / FIRST_GUESS_STREAM),
            "output_start": icon_date(cycle.end),
            "output_end": icon_date(cycle.end),
            "output_interval": iso_duration(
                int((cycle.end - cycle.start).total_seconds() // 60)
            ),
            "ml_varlist": list(icon.first_guess_variables),
            **common,
        },
    ]
    extra = icon.namelist.get("output_nml", [])
    if isinstance(extra, dict):
        extra = [extra]
    for stream in extra:
        stream = dict(stream)
        name = stream.get("output_filename", "extra")
        if Path(name).name in (FORECAST_STREAM, FIRST_GUESS_STREAM):
            raise ValueError(
                f"The output stream name '{name}' is used by wrf-ensembly, pick another"
            )
        if not Path(name).is_absolute():
            stream["output_filename"] = str(output_dir / name)
        streams.append(stream)
    namelist["output_nml"] = streams

    return namelist


def merge(base: Namelist, override: dict[str, Any]) -> Namelist:
    """Returns `base` with the variables of `override` set, group by group"""

    result: Namelist = {
        k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()
    }
    for name, group in override.items():
        current = result.get(name)
        if isinstance(current, dict) and isinstance(group, dict):
            current.update(group)
        else:
            result[name] = dict(group) if isinstance(group, dict) else group
    return result


def format_value(value: Any) -> str:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(v) for v in value)
    return convert_to_string(value)


def write(namelist: Namelist, path: Path):
    """
    Writes a namelist file. A group whose value is a list is written once per item,
    which is how ICON takes several output streams.
    """

    lines = []
    for name, groups in namelist.items():
        for group in groups if isinstance(groups, list) else [groups]:
            lines.append(f"&{name}")
            for key, value in group.items():
                if value is None:
                    continue
                lines.append(f" {key} = {format_value(value)}")
            lines.append("/")
    path.write_text("\n".join(lines) + "\n")


def write_latbc_map(variables: dict[str, str], path: Path):
    """Writes the file that tells ICON what its variables are called in boundary files"""

    width = max(len(name) for name in variables) + 2
    path.write_text(
        "".join(f"{name:<{width}}{in_file}\n" for name, in_file in variables.items())
    )
