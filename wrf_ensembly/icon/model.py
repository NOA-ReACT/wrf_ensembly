"""
Running ICON members and cycling them with the filter's analysis.

ICON has its own way of starting from a short forecast plus an analysis, the one its
operational data assimilation uses, and we go through it instead of editing restart
files:

- At the end of a cycle each member writes a first guess file on the native grid. It
  holds the prognostic state (edge-normal wind `vn`, `w`, `rho`, `theta_v`, tracers,
  soil and surface fields of all tiles) and the diagnostic `u`, `v`, `temp`, `pres`.
- The filter reads its prior from the first guess files and writes `u`, `v`, `temp`,
  `pres` and the moisture variables of the analysis.
- The next cycle starts with `init_mode = 1`: ICON reads the first guess and the
  analysis file, turns the difference in `u` and `v` into an increment of `vn` and
  rebuilds `rho` and `theta_v` from the analysed temperature and pressure.

The first cycle starts from IFS data (`init_mode = 2`).
"""

import datetime as dt
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import netCDF4
import numpy as np

from wrf_ensembly import cycling, external, utils
from wrf_ensembly.console import logger
from wrf_ensembly.icon import namelists

if TYPE_CHECKING:
    from wrf_ensembly.experiment import Experiment

ANALYSIS_FILE_VARIABLES = ("temp", "pres", "u", "v", "qv", "qc", "qi", "qr", "qs", "qg")
"""
What ICON takes from the analysis file with `init_mode = 1`. Other state variables
(e.g. aerosol tracers) are part of the first guess and get updated there.
"""

REQUIRED_ANALYSIS_VARIABLES = ("temp", "pres", "u", "v")
"""ICON needs these four together to build its prognostic variables"""

EXECUTABLE = "icon"
LOG_FILE = "icon.log"
FINISH_STATUS = "finish.status"


def for_member(path: Path, member: int) -> Path:
    """`path` with the %MEMBER% placeholder replaced by the member number"""

    return Path(str(path).replace("%MEMBER%", f"{member:02d}"))


def executable(exp: "Experiment") -> Path:
    root = exp.cfg.directories.icon_root
    assert root is not None
    return root / "bin" / "icon"


def _link(target: Path, link: Path):
    """Makes `link` point to `target`, replacing whatever was there"""

    if link.is_symlink() or link.exists():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link)
        else:
            link.unlink()
    link.symlink_to(target.resolve())


def setup_member(exp: "Experiment", member_i: int):
    """
    Prepares a member's run directory for the first cycle: links to the executable,
    the grid, the external parameters, the IFS initial conditions, the boundary data
    and ecRad's data. With `[perturbations]`, the member gets its own copy of the
    initial conditions instead of a link, for `apply-perturbations` to change.
    """

    icon = exp.cfg.icon
    assert icon is not None
    member_path = exp.paths.member_path(member_i)
    member_path.mkdir(parents=True, exist_ok=True)

    ecrad_data = icon.ecrad_data
    if ecrad_data is None:
        assert exp.cfg.directories.icon_root is not None
        ecrad_data = exp.cfg.directories.icon_root / "externals" / "ecrad" / "data"

    links = {
        EXECUTABLE: executable(exp),
        namelists.GRID_LINK: icon.grid_file,
        namelists.EXTPAR_LINK: icon.extpar_file,
        namelists.IFS_LINK: for_member(icon.initial_conditions, member_i),
        namelists.LATBC_LINK: for_member(icon.boundary_conditions, member_i),
        namelists.ECRAD_LINK: ecrad_data,
    }
    for file_cfg in icon.extra_files:
        links[file_cfg.destination_name or Path(file_cfg.source).name] = Path(
            file_cfg.source
        )

    for name, target in links.items():
        if not target.exists():
            raise FileNotFoundError(f"Member {member_i}: {target} does not exist")
        if name == namelists.IFS_LINK and exp.cfg.perturbations.variables:
            (member_path / name).unlink(missing_ok=True)
            shutil.copyfile(target, member_path / name)
        else:
            _link(target, member_path / name)

    namelists.write_latbc_map(
        icon.boundary_variables, member_path / namelists.LATBC_MAP_FILE
    )

    # Start from the IFS data, not from what an earlier experiment left here
    for name in (namelists.FIRST_GUESS_LINK, namelists.ANALYSIS_LINK):
        (member_path / name).unlink(missing_ok=True)


def missing_boundary_files(exp: "Experiment", member_i: int) -> list[Path]:
    """
    The boundary files the experiment needs that are not there. Only understands the
    date keywords of `boundary_filename` (<y>, <m>, <d>, <h>, <min>, <sec>).
    """

    icon = exp.cfg.icon
    assert icon is not None
    directory = for_member(icon.boundary_conditions, member_i)
    step = dt.timedelta(minutes=exp.cfg.time_control.boundary_update_interval)

    missing = []
    time = exp.cfg.time_control.start + step
    while time <= exp.cfg.time_control.end:
        name = icon.boundary_filename
        for keyword, value in (
            ("<y>", f"{time:%Y}"),
            ("<m>", f"{time:%m}"),
            ("<d>", f"{time:%d}"),
            ("<h>", f"{time:%H}"),
            ("<min>", f"{time:%M}"),
            ("<sec>", f"{time:%S}"),
        ):
            name = name.replace(keyword, value)
        if "<" in name:
            return []  # a keyword we do not know, leave the check to ICON
        if not (directory / name).exists():
            missing.append(directory / name)
        time += step
    return missing


def run_member(
    exp: "Experiment", member_i: int, cycle: cycling.CycleInformation, cores: int
) -> tuple[bool, dt.datetime, dt.datetime]:
    """
    Writes the namelists and runs ICON for a member from `cycle.start` to
    `cycle.forecast_end`.

    Returns:
        Whether ICON succeeded, and when it started and ended
    """

    member_path = exp.paths.member_path(member_i)
    output_dir = exp.paths.scratch_forecasts_path(cycle.index, member_i)
    output_dir.mkdir(parents=True, exist_ok=True)

    first_guess = member_path / namelists.FIRST_GUESS_LINK
    analysis = member_path / namelists.ANALYSIS_LINK
    from_first_guess = cycle.index > 0
    if from_first_guess and not first_guess.exists():
        raise FileNotFoundError(
            f"Member {member_i}: No first guess to start cycle {cycle.index} from at "
            f"{first_guess}, was the previous cycle cycled?"
        )

    namelists.write(
        namelists.master_namelist(exp.cfg, cycle),
        member_path / namelists.MASTER_NAMELIST,
    )
    namelists.write(
        namelists.atmosphere_namelist(
            exp.cfg,
            cycle,
            member_i,
            output_dir.resolve(),
            from_first_guess=from_first_guess,
            with_analysis=from_first_guess and analysis.exists(),
        ),
        member_path / namelists.ATMO_NAMELIST,
    )

    # Output of an earlier attempt would be mistaken for this run's
    for old in list(output_dir.glob(f"{namelists.FORECAST_STREAM}_*.nc")) + list(
        output_dir.glob(f"{namelists.FIRST_GUESS_STREAM}_*.nc")
    ):
        old.unlink()
    status_file = member_path / FINISH_STATUS
    status_file.unlink(missing_ok=True)

    logger.info(f"Running ICON for member {member_i}...")
    cmd = [
        *exp.cfg.slurm.mpirun_command.split(" "),
        "-n",
        str(cores),
        str((member_path / EXECUTABLE).resolve()),
    ]
    start_time = dt.datetime.now()
    res = external.runc(cmd, cwd=member_path, log_filename=LOG_FILE)
    end_time = dt.datetime.now()

    # ICON writes "OK" when it reached the end of the experiment
    status = status_file.read_text().strip() if status_file.exists() else ""
    if res.returncode != 0 or status != "OK":
        logger.error(
            f"Member {member_i}: icon failed with exit code {res.returncode}, "
            f"{FINISH_STATUS} is '{status}'"
        )
        for line in res.output.splitlines():
            if "FINISH PE" in line:
                logger.error(line.strip())
        return False, start_time, end_time

    first_guess_out = namelists.first_guess_file(output_dir, cycle.end)
    if not first_guess_out.exists():
        logger.error(
            f"Member {member_i}: ICON finished but there is no first guess file at "
            f"{first_guess_out}"
        )
        return False, start_time, end_time

    return True, start_time, end_time


def cell_dimension(ds: netCDF4.Dataset, ncells: int | None = None) -> str:
    """
    Name of the dimension that runs over the cells. ICON calls it `ncells`, or
    `ncells_2` if the file also has variables on edges.
    """

    for name in ("temp", "pres", "u", "qv", "pres_sfc", "z_ifc"):
        if name in ds.variables:
            return ds.variables[name].dimensions[-1]
    candidates = [
        name
        for name, dim in ds.dimensions.items()
        if name.startswith("ncells") and (ncells is None or len(dim) == ncells)
    ]
    if len(candidates) != 1:
        raise ValueError(f"Cannot find the cell dimension of {ds.filepath()}")
    return candidates[0]


def _posterior_field(posterior: netCDF4.Dataset, name: str, shape: tuple) -> np.ndarray:
    """A variable of the filter's output, in the shape it has in the ICON file"""

    data = posterior.variables[name][:]
    data = np.ma.filled(data, np.nan) if np.ma.isMaskedArray(data) else np.asarray(data)
    if data.size != int(np.prod(shape)):
        raise ValueError(
            f"{name} has shape {data.shape} in {posterior.filepath()}, "
            f"which does not fit {shape}"
        )
    if not np.isfinite(data).all():
        raise ValueError(f"{name} in {posterior.filepath()} has missing values")
    return data.reshape(shape)


def write_analysis_file(
    first_guess: Path, posterior: Path, target: Path, variables: list[str]
):
    """
    Writes the analysis file that ICON reads next to the first guess.

    It has `temp`, `pres`, `u`, `v` (taken from the first guess if the filter did not
    update them) and the moisture variables that are in `variables`, with the grid
    and time information of the first guess.
    """

    names = list(REQUIRED_ANALYSIS_VARIABLES)
    names += [v for v in variables if v in ANALYSIS_FILE_VARIABLES and v not in names]

    tmp = target.with_name(target.name + ".tmp")
    with (
        netCDF4.Dataset(first_guess) as fg,
        netCDF4.Dataset(posterior) as post,
        netCDF4.Dataset(tmp, "w", format="NETCDF4") as out,
    ):
        fg.set_auto_maskandscale(False)
        out.setncatts({k: fg.getncattr(k) for k in fg.ncattrs()})

        needed_dims: list[str] = []
        for name in names:
            if name not in fg.variables:
                raise ValueError(
                    f"{name} is not in the first guess file {first_guess}, add it to "
                    "icon.first_guess_variables"
                )
            for dim in fg.variables[name].dimensions:
                if dim not in needed_dims:
                    needed_dims.append(dim)

        for dim in needed_dims:
            size = None if fg.dimensions[dim].isunlimited() else len(fg.dimensions[dim])
            out.createDimension(dim, size)
        # Coordinate variables: time and the level index
        for dim in needed_dims:
            if dim in fg.variables:
                _copy_variable(fg, out, dim)
                bounds = getattr(fg.variables[dim], "bounds", None)
                if bounds is not None and bounds in fg.variables:
                    for bdim in fg.variables[bounds].dimensions:
                        if bdim not in out.dimensions:
                            out.createDimension(bdim, len(fg.dimensions[bdim]))
                    _copy_variable(fg, out, bounds)

        for name in names:
            src = fg.variables[name]
            var = _copy_variable(fg, out, name, copy_data=False)
            if name in variables:
                if name not in post.variables:
                    raise ValueError(
                        f"{name} is in assimilation.state_variables but not in the "
                        f"filter's output {posterior}"
                    )
                var[:] = _posterior_field(post, name, src.shape)
            else:
                var[:] = src[:]
    tmp.rename(target)


def _copy_variable(
    src_ds: netCDF4.Dataset, dst_ds: netCDF4.Dataset, name: str, copy_data: bool = True
):
    src = src_ds.variables[name]
    var = dst_ds.createVariable(name, src.dtype, src.dimensions)
    var.setncatts({k: src.getncattr(k) for k in src.ncattrs() if k != "_FillValue"})
    if copy_data:
        var[:] = src[:]
    return var


def overwrite_variables(
    target: Path, posterior: Path, variables: list[str]
) -> list[str]:
    """
    Writes the filter's output for `variables` into an ICON file, where they exist in
    it. Returns the names it wrote.
    """

    written = []
    with netCDF4.Dataset(target, "r+") as ds, netCDF4.Dataset(posterior) as post:
        ds.set_auto_maskandscale(False)
        for name in variables:
            if name not in ds.variables or name not in post.variables:
                continue
            ds.variables[name][:] = _posterior_field(
                post, name, ds.variables[name].shape
            )
            written.append(name)
    return written


def analysis_first_guess_file(exp: "Experiment", cycle_i: int, member_i: int) -> Path:
    """The first guess with the analysis of the variables ICON only reads from there"""

    return namelists.first_guess_file(
        exp.paths.scratch_analysis_path(cycle_i) / f"member_{member_i:02d}",
        exp.cycles[cycle_i].end,
    )


def analysis_file(exp: "Experiment", cycle_i: int, member_i: int) -> Path:
    cycle = exp.cycles[cycle_i]
    return (
        exp.paths.scratch_analysis_path(cycle_i)
        / f"member_{member_i:02d}"
        / f"ana_{namelists.file_date(cycle.end)}.nc"
    )


def make_analysis(exp: "Experiment", member_i: int):
    """
    Turns the filter's output for a member into what ICON and the postprocessing need,
    in the cycle's scratch analysis directory:

    - `ana_<date>.nc`, the analysis file for the next cycle's start
    - `fg_<date>.nc`, only if the state has variables that ICON reads from the first
      guess: a copy of it with those variables updated
    - `icon_<date>.nc`, the forecast output at the analysis time with the analysed
      variables, so that the analysis is postprocessed like a forecast
    """

    cycle_i = exp.current_cycle_i
    cycle = exp.current_cycle
    variables = exp.cfg.assimilation.state_variables

    forecast_dir = exp.paths.scratch_forecasts_path(cycle_i, member_i)
    analysis_dir = exp.paths.scratch_analysis_path(cycle_i) / f"member_{member_i:02d}"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    posterior = exp.paths.scratch_dart_path(cycle_i) / f"dart_member_{member_i:02d}.nc"
    first_guess = namelists.first_guess_file(forecast_dir, cycle.end)

    write_analysis_file(
        first_guess, posterior, analysis_file(exp, cycle_i, member_i), variables
    )

    in_first_guess = [v for v in variables if v not in ANALYSIS_FILE_VARIABLES]
    if in_first_guess:
        target = analysis_first_guess_file(exp, cycle_i, member_i)
        utils.copy(first_guess, target)
        written = overwrite_variables(target, posterior, in_first_guess)
        missing = set(in_first_guess) - set(written)
        if missing:
            raise ValueError(
                f"State variable(s) {', '.join(sorted(missing))} are not in the first "
                f"guess file {first_guess}, add them to icon.first_guess_variables"
            )

    forecast = namelists.forecast_file(forecast_dir, cycle.end)
    if forecast.exists():
        target = namelists.forecast_file(analysis_dir, cycle.end)
        utils.copy(forecast, target)
        overwrite_variables(target, posterior, variables)
        with netCDF4.Dataset(target, "r+") as ds:
            ds.experiment_name = exp.cfg.metadata.name
            ds.current_cycle = cycle_i
            ds.cycle_start = cycle.start.strftime("%Y-%m-%d_%H:%M:%S")
            ds.cycle_end = cycle.end.strftime("%Y-%m-%d_%H:%M:%S")


def cycle_member(exp: "Experiment", member_i: int, use_forecast: bool):
    """
    Points a member's run directory to what the next cycle starts from: the first
    guess of the current cycle and, unless `use_forecast`, the analysis.
    """

    cycle_i = exp.current_cycle_i
    cycle = exp.current_cycle
    member_path = exp.paths.member_path(member_i)

    first_guess = namelists.first_guess_file(
        exp.paths.scratch_forecasts_path(cycle_i, member_i), cycle.end
    )
    analysis_link = member_path / namelists.ANALYSIS_LINK
    analysis_link.unlink(missing_ok=True)

    if not use_forecast:
        analysis = analysis_file(exp, cycle_i, member_i)
        if not analysis.exists():
            raise FileNotFoundError(
                f"Member {member_i}: No analysis file at {analysis}, run the analysis first"
            )
        _link(analysis, analysis_link)
        updated = analysis_first_guess_file(exp, cycle_i, member_i)
        if updated.exists():
            first_guess = updated

    if not first_guess.exists():
        raise FileNotFoundError(
            f"Member {member_i}: No first guess file at {first_guess}"
        )
    _link(first_guess, member_path / namelists.FIRST_GUESS_LINK)
    logger.info(
        f"Member {member_i}: Next cycle starts from {first_guess.name}"
        + ("" if use_forecast else f" and {analysis_link.resolve().name}")
    )

    (member_path / FINISH_STATUS).unlink(missing_ok=True)


def prepare_filter(exp: "Experiment") -> list[Path]:
    """
    Links the files DART's ICON interface reads next to `filter` and returns the prior
    file of each member (their first guess files).
    """

    icon = exp.cfg.icon
    assert icon is not None
    cycle_i = exp.current_cycle_i
    priors = [
        namelists.first_guess_file(
            exp.paths.scratch_forecasts_path(cycle_i, member_i), exp.current_cycle.end
        )
        for member_i in range(exp.cfg.assimilation.n_members)
    ]

    model_nml = exp.cfg.dart_namelist.get("model_nml", {})
    dart_dir = exp.paths.dart_work_dir
    # The first guess has everything: the state variables and the level heights
    links = {
        model_nml.get("template_file", "icon_template.nc"): priors[0],
        model_nml.get("height_file", "icon_heights.nc"): priors[0],
        model_nml.get("grid_file", "icon_grid.nc"): icon.grid_file,
    }
    for name, target in links.items():
        _link(target, dart_dir / name)
        logger.info(f"Linked {dart_dir / name} to {target}")
    return priors


def clean_first_guess_files(exp: "Experiment"):
    """
    Deletes the first guess and analysis files of cycles before the previous one, they
    are large and nothing starts from them anymore. Honours
    `assimilation.keep_restart_files` and `keep_restart_files_for_cycles`.
    """

    if exp.cfg.assimilation.keep_restart_files:
        return
    keep = set(exp.cfg.assimilation.keep_restart_files_for_cycles)
    for cycle_i in range(exp.current_cycle_i - 1):
        if cycle_i in keep:
            continue
        directories = [exp.paths.scratch_forecasts_path(cycle_i)]
        directories.append(exp.paths.scratch_analysis_path(cycle_i))
        for directory in directories:
            for pattern in (f"{namelists.FIRST_GUESS_STREAM}_*.nc", "ana_*.nc"):
                for f in directory.glob(f"member_*/{pattern}"):
                    logger.debug(f"Removing {f}")
                    f.unlink()
