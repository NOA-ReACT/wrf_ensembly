import itertools
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cartopy.crs as ccrs
import h5py
import netCDF4
import pyproj
import xarray as xr
import xwrf  # noqa: F401

from wrf_ensembly import fortran_namelists, update_bc
from wrf_ensembly.config import Config, DomainControlConfig
from wrf_ensembly.console import logger
from wrf_ensembly.cycling import CycleInformation
from wrf_ensembly.experiment.paths import ExperimentPaths

if TYPE_CHECKING:
    from netCDF4 import CompressionLevel

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
            "geog_data_res": "30s",
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
    member: int | None = None,
    paths: ExperimentPaths | None = None,
    add_iofields: bool = True,
    restart_interval: int | None = None,
    restart: bool | None = None,
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
        add_iofields: If True, the iofields.txt file will be generated if the config has runtime_io set. Use only for wrf.exe, not real.exe.
        restart_interval: In restart mode, minutes between restart files. Defaults to the
            cycle's length, so the only restart file is the one at its end.
        restart: In restart mode, whether wrf.exe starts from a restart file. Defaults to
            every cycle but the first; a run resumed from a checkpoint always does.
    """

    if member is not None and paths is None:
        raise ValueError(
            "paths must be provided when generating a member-specific namelist"
        )

    # Determine start/end times. The forward run (and the boundary conditions
    # produced by real.exe) extend to `forecast_end`, which equals `cycle.end`
    # unless a forecast extension is configured. The assimilation boundary itself
    # stays at `cycle.end` and is handled outside namelist generation.
    start = cycle.start
    end = cycle.forecast_end

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
        wrf_namelist["time_control"]["history_outname"] = (
            f"{wrfout_dest}/wrfout_d<domain>_<date>"
        )

    # Add iofields
    if (
        add_iofields
        and cfg.time_control.runtime_io is not None
        and len(cfg.time_control.runtime_io) > 0
    ):
        wrf_namelist["time_control"]["iofields_filename"] = "iofields.txt"
        wrf_namelist["time_control"]["ignore_iofields_warning"] = False

        if path.is_dir():
            iofields_path = path / "iofields.txt"
        else:
            iofields_path = path.parent / "iofields.txt"

        with open(iofields_path, "w") as f:
            for var in cfg.time_control.runtime_io:
                f.write(f"{var}\n")
        logger.info(f"Wrote iofields to {iofields_path}")

    # Add overrides. Groups are copied so the per-member overrides and chem_in_opt below
    # don't modify `cfg.wrf_namelist` when several namelists are built in one process.
    for name, group in cfg.wrf_namelist.items():
        if name in wrf_namelist:
            wrf_namelist[name] |= group
        else:
            wrf_namelist[name] = dict(group)
    # Always resolve, even without a member, so invalid keys fail early (e.g. at `preprocess`)
    member_overrides = cfg.wrf_namelist_overrides_for_member(
        member if member is not None else -1
    )
    if member is not None:
        for name, group in member_overrides.items():
            if name in wrf_namelist:
                wrf_namelist[name] |= group
            else:
                wrf_namelist[name] = group

    # In restart mode, every cycle writes a restart file at its end, and every cycle but
    # the first starts from one (the first starts from real.exe's wrfinput). This
    # overrides any restart settings in the config, which can't work here.
    if cfg.assimilation.cycling_mode == "restart":
        time_control = wrf_namelist["time_control"]
        time_control["restart"] = (
            restart if restart is not None else member is not None and cycle.index > 0
        )
        if member is not None and paths is not None:
            rst_dest = paths.scratch_restart_path(cycle.index, member)
            rst_dest.mkdir(parents=True, exist_ok=True)
            time_control["restart_interval"] = (
                restart_interval
                if restart_interval is not None
                else int((cycle.end - cycle.start).total_seconds() // 60)
            )
            time_control["rst_outname"] = f"{rst_dest}/wrfrst_d<domain>_<date>"
            # Take history_interval etc. from the namelist, not from the restart file
            time_control["override_restart_timers"] = True

    # With sst_update, real.exe writes the lower boundary (SST, vegetation, albedo, sea ice)
    # to wrflowinp_d01 and wrf.exe reads it as auxinput4. Both refuse to run without these.
    if wrf_namelist.get("physics", {}).get("sst_update", 0) == 1:
        time_control = wrf_namelist["time_control"]
        time_control.setdefault("io_form_auxinput4", 2)
        time_control.setdefault("auxinput4_inname", "wrflowinp_d<domain>")
        time_control.setdefault(
            "auxinput4_interval", cfg.time_control.boundary_update_interval
        )

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


def wrfout_time(name: str) -> datetime:
    """The (UTC) time of a WRF output or restart file from its name, e.g. `wrfout_d01_2021-01-01_06:00:00`"""

    return datetime.strptime(name[-19:], "%Y-%m-%d_%H:%M:%S").replace(
        tzinfo=timezone.utc
    )


def extract_boundary_records(
    src: Path, dest: Path, start: datetime, end: datetime
) -> int:
    """
    Copies the records of a wrfbdy file needed to run from `start` to `end` into a new
    file. These are the records whose interval (from `thisbdytime` to `nextbdytime`)
    overlaps [start, end).

    Used in the restart cycling mode, where real.exe makes one wrfbdy for the whole
    experiment and each member gets the part its cycle needs. `update_bc` then modifies
    the member's copy. If all records are needed, the file is copied as is.

    Args:
        src: The wrfbdy to read from
        dest: The wrfbdy to create, overwritten if it exists
        start: Start of the run
        end: End of the run

    Returns:
        The number of records copied
    """

    with netCDF4.Dataset(src, "r") as ds_src:  # type: ignore
        ds_src.set_auto_mask(False)
        this_times = update_bc.parse_wrf_times(ds_src[update_bc.THIS_BDY_TIME])
        next_times = update_bc.parse_wrf_times(ds_src[update_bc.NEXT_BDY_TIME])
        records = [
            i
            for i, (this, nxt) in enumerate(zip(this_times, next_times))
            if this < end and nxt > start
        ]
        if (
            not records
            or this_times[records[0]] > start
            or next_times[records[-1]] < end
        ):
            raise ValueError(
                f"{src} does not cover {start} -> {end} "
                f"({this_times[0]} -> {next_times[-1]})"
            )
        first, last = records[0], records[-1] + 1

        dest.unlink(missing_ok=True)
        if first == 0 and last == len(this_times):
            # All records are needed (e.g. a segment covering the whole experiment), so
            # copy the file instead of decompressing and recompressing every variable
            shutil.copy(src, dest)
            return last - first

        record_vars = []
        with netCDF4.Dataset(dest, "w", format=ds_src.data_model) as ds_dest:  # type: ignore
            ds_dest.setncatts({a: ds_src.getncattr(a) for a in ds_src.ncattrs()})
            ds_dest.START_DATE = this_times[first].strftime("%Y-%m-%d_%H:%M:%S")
            for name, dim in ds_src.dimensions.items():
                ds_dest.createDimension(name, None if dim.isunlimited() else dim.size)
            for name, var in ds_src.variables.items():
                # Keep the compression and chunking of the source (WRF built with
                # netCDF4 compresses its output), or the copy ends up larger than
                # the whole experiment's file
                filters = var.filters()
                chunking = var.chunking()
                out = ds_dest.createVariable(
                    name,
                    var.datatype,
                    var.dimensions,
                    zlib=filters["zlib"],
                    complevel=cast("CompressionLevel", filters["complevel"]),
                    shuffle=filters["shuffle"],
                    chunksizes=None if chunking == "contiguous" else chunking,
                )
                out.setncatts({a: var.getncattr(a) for a in var.ncattrs()})
                if (
                    var.dimensions
                    and ds_src.dimensions[var.dimensions[0]].isunlimited()
                ):
                    record_vars.append(name)
                else:
                    out[:] = var[:]

        # Copying the compressed chunks is much faster than recompressing the records
        if not _copy_record_chunks(src, dest, record_vars, first, last):
            logger.debug(f"Can't copy the chunks of {src}, recompressing the records")
            with netCDF4.Dataset(dest, "a") as ds_dest:  # type: ignore
                for name in record_vars:
                    ds_dest[name][:] = ds_src[name][first:last]

    return last - first


def _copy_record_chunks(
    src: Path, dest: Path, names: list[str], first: int, last: int
) -> bool:
    """
    Copies records `first:last` of the variables `names` from `src` into `dest` as
    compressed chunks, without decompressing them. `dest` must have the variables
    already, without any records.

    Only possible for netCDF4 files where each chunk holds one record and the variables
    have the same chunking and filters in both files. Returns False, before changing
    `dest`, if that's not the case.
    """

    if not names:
        return True
    if not (h5py.is_hdf5(src) and h5py.is_hdf5(dest)):
        return False

    with h5py.File(src, "r") as h_src, h5py.File(dest, "r+") as h_dest:
        pairs = []
        for name in names:
            var_src, var_dest = h_src.get(name), h_dest.get(name)
            if not (
                isinstance(var_src, h5py.Dataset)
                and isinstance(var_dest, h5py.Dataset)
                and var_src.chunks is not None
                and var_src.chunks[0] == 1
                and var_src.chunks == var_dest.chunks
                and var_src.shape[1:] == var_dest.shape[1:]
                and var_src.dtype == var_dest.dtype
                and _hdf5_filters(var_src) == _hdf5_filters(var_dest)
            ):
                return False
            # Chunks that were never written read as the fill value, but can't be copied
            n_chunks = math.prod(
                math.ceil(size / chunk)
                for size, chunk in zip(var_src.shape, var_src.chunks)
            )
            if var_src.id.get_num_chunks() != n_chunks:
                return False
            pairs.append((var_src, var_dest))

        for var_src, var_dest in pairs:
            var_dest.resize(last - first, axis=0)
            offsets = itertools.product(
                range(first, last),
                *(
                    range(0, size, chunk)
                    for size, chunk in zip(var_src.shape[1:], var_src.chunks[1:])
                ),
            )
            for t, *rest in offsets:
                mask, data = var_src.id.read_direct_chunk((t, *rest))
                var_dest.id.write_direct_chunk((t - first, *rest), data, mask)

    return True


def _hdf5_filters(var: h5py.Dataset) -> list[tuple]:
    """The filter pipeline of an HDF5 dataset, as (id, flags, parameters) per filter"""

    plist = var.id.get_create_plist()
    return [plist.get_filter(i)[:3] for i in range(plist.get_nfilters())]


def _create_proj_crs(domain: DomainControlConfig):
    if domain.projection.lower() != "lambert":
        raise NotImplementedError(
            f"Projection {domain.projection} not supported yet in _create_proj_crs()"
        )

    if domain.stand_lon is None or domain.ref_lat is None:
        raise ValueError("stand_lon and ref_lat must be set in the domain config")

    return pyproj.CRS(
        {
            "x_0": 0,
            "y_0": 0,
            "a": 6370000,
            "b": 6370000,
            "proj": "lcc",
            "lat_1": domain.truelat1,
            "lat_2": domain.truelat2,
            "lat_0": domain.ref_lat,
            "lon_0": domain.stand_lon,
        }
    )


def get_wrf_proj_transformer(domain: DomainControlConfig):
    """
    Returns a pyproj transformer for the given WRF domain. Source projection is always
    WGS84 (EPSG:4326).

    You can use this transformer to convert lat/lon coordinates to the WRF domain's (x, y) projection,
    where grid points are regularly spaced. The returned transformer uses (lon, lat) ordering.

    Only works for Lambert Conformal Conic projections!
    """

    wrf_crs = _create_proj_crs(domain)
    return pyproj.Transformer.from_crs(pyproj.CRS("EPSG:4326"), wrf_crs, always_xy=True)


def get_wrf_reverse_proj_transformer(domain: DomainControlConfig):
    """
    Returns a pyproj transformer for the given WRF domain. Target projection is always
    WGS84 (EPSG:4326).

    You can use this transformer to convert (x, y) coordinates in the WRF domain's projection
    to lat/lon coordinates. The returned transformer uses (x, y) ordering.

    Only works for Lambert Conformal Conic projections!
    """

    wrf_crs = _create_proj_crs(domain)
    return pyproj.Transformer.from_crs(wrf_crs, pyproj.CRS("EPSG:4326"), always_xy=True)


def get_wrf_cartopy_crs(domain: DomainControlConfig):
    """
    Returns a cartopy CRS for the given WRF domain.

    Only works for Lambert Conformal Conic projections!
    """

    if domain.projection.lower() != "lambert":
        raise NotImplementedError(
            f"Projection {domain.projection} not supported yet in get_wrf_cartopy_crs()"
        )
    if domain.stand_lon is None or domain.ref_lat is None:
        raise ValueError("stand_lon and ref_lat must be set in the domain config")

    return ccrs.LambertConformal(
        central_longitude=domain.stand_lon,
        central_latitude=domain.ref_lat,
        standard_parallels=(domain.truelat1, domain.truelat2),
    )


def get_wrf_cartopy_crs_from_ds_attrs(ds: xr.Dataset):
    """
    Same as `get_wrf_cartopy_crs` but reads the required values from the attributes
    of a dataset. Useful if you have opened an output file with xarray and just want to
    get the right projection.
    """

    required_attrs = ["STAND_LON", "CEN_LAT", "TRUELAT1", "TRUELAT2"]
    for attr in required_attrs:
        if attr not in ds.attrs:
            raise ValueError(f"{attr} must be set in the dataset attributes")

    return ccrs.LambertConformal(
        central_longitude=ds.attrs["STAND_LON"],
        central_latitude=ds.attrs["CEN_LAT"],
        standard_parallels=(ds.attrs["TRUELAT1"], ds.attrs["TRUELAT2"]),
    )


def get_spatial_domain_bounds(wrfinput_path: Path) -> tuple[float, float, float, float]:
    """
    Returns the spatial bounds of a WRF domain from a WRF input file (wrfinput_d01 or similar).

    Args:
        wrfinput_path: Path to the WRF input file.

    Returns:
        A tuple of (x_min, x_max, y_min, y_max) in the WRF projection's units.
    """

    with xr.open_dataset(wrfinput_path) as ds:
        ds = ds.xwrf.postprocess()
        x = ds["x"]  # .isel(Time=0).values
        y = ds["y"]  # .isel(Time=0).values

        x_min = float(x.min().item())
        x_max = float(x.max().item())
        y_min = float(y.min().item())
        y_max = float(y.max().item())

    return x_min, x_max, y_min, y_max


def get_temporal_domain_bounds(
    cycles: list[CycleInformation],
) -> tuple[datetime, datetime]:
    """
    Returns the temporal bounds of a list of cycles.

    Args:
        cycles: List of CycleInformation objects.

    Returns:
        A tuple of (start, end) datetimes.
    """

    if len(cycles) == 0:
        raise ValueError("cycles list is empty")

    start = cycles[0].start
    end = cycles[-1].end

    return start, end
