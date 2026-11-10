"""
First step of the postprocessing pipeline for ICON output.

Turns a forecast file on the native triangular grid into the layout that the rest of
the pipeline (statistics, validation, plots) expects: dimensions `(t, z, y, x)` on a
regular lat/lon grid, with `x` the longitude and `y` the latitude in degrees and `z`
counting model levels from the ground up.
"""

import re

import netCDF4
import numpy as np
import xarray as xr

from wrf_ensembly.console import logger
from wrf_ensembly.icon.regrid import IconGrid, LatLonGrid, Regridder, find_cell_dim
from wrf_ensembly.processors import DataProcessor, ProcessingContext

RENAMES = {
    "temp": "air_temperature",
    "pres": "air_pressure",
    "u": "wind_east",
    "v": "wind_north",
    "z_mc": "geopotential_height",
}
"""ICON names of the variables that the validation tools look for under other names"""


class IconPostProcessor(DataProcessor):
    """
    - Interpolates all cell variables to a regular lat/lon grid (`[icon.regrid]`)
    - Averages variables on half levels (e.g. `w`) to full levels
    - Turns the vertical around, so that `z = 0` is the lowest model level
    - Adds the height of the model levels as `geopotential_height`
    - Renames the time dimension to `t`, the horizontal ones to `x` and `y`
    - Filters variables with `postprocess.variables_to_keep`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._regridders: dict[str, Regridder] = {}
        self._heights: xr.DataArray | None = None
        self._heights_checked = False

    @property
    def name(self) -> str:
        return "ICON post-processor"

    def _regridder(self, context: ProcessingContext, method: str) -> Regridder:
        if method not in self._regridders:
            icon = context.config.icon
            assert icon is not None
            cfg = icon.regrid
            target = None
            if cfg.bounds is not None or cfg.resolution is not None:
                grid = IconGrid.from_file(icon.grid_file)
                if cfg.bounds is not None:
                    west, east, south, north = cfg.bounds
                    resolution = (
                        cfg.resolution
                        or LatLonGrid.covering(grid).lon[1]
                        - (LatLonGrid.covering(grid).lon[0])
                    )
                    target = LatLonGrid.regular(west, east, south, north, resolution)
                else:
                    target = LatLonGrid.covering(grid, cfg.resolution)

            if context.cache_dir is not None:
                self._regridders[method] = Regridder.cached(
                    icon.grid_file, context.cache_dir, target, method
                )
            else:
                self._regridders[method] = Regridder.build(
                    IconGrid.from_file(icon.grid_file), target, method
                )
            logger.debug(f"Interpolation weights ready ({method})")
        return self._regridders[method]

    def _method_for(self, context: ProcessingContext, name: str) -> str:
        icon = context.config.icon
        assert icon is not None
        for pattern, method in icon.regrid.method_overrides.items():
            if re.fullmatch(pattern, name):
                return method
        return icon.regrid.method

    def _level_heights(
        self, context: ProcessingContext, cell_dim: str, ncells: int
    ) -> xr.DataArray | None:
        """
        Height of the full levels on the native grid, (z, cells), lowest level first.
        Read once from a first guess file next to the input, which always has the half
        level heights.
        """

        if self._heights_checked:
            return self._heights
        self._heights_checked = True

        candidates = sorted(context.input_file.parent.glob("fg_*.nc"))
        if not candidates:
            # The analysis directory only has a first guess if tracers are analysed
            forecasts = context.input_file.parent.parent.parent.parent / "forecasts"
            candidates = sorted(
                forecasts.glob(f"cycle_{context.cycle:03d}/member_*/fg_*.nc")
            )
        for path in candidates:
            with netCDF4.Dataset(path) as ds:
                if "z_ifc" not in ds.variables:
                    continue
                z_ifc = np.asarray(ds.variables["z_ifc"][:], dtype=np.float32)
            z_ifc = z_ifc.reshape(-1, z_ifc.shape[-1])
            if z_ifc.shape[-1] != ncells:
                continue
            z_mc = 0.5 * (z_ifc[:-1] + z_ifc[1:])[::-1]
            self._heights = xr.DataArray(
                z_mc,
                dims=("z", cell_dim),
                attrs={
                    "units": "m",
                    "long_name": "Height of the model levels above mean sea level",
                    "standard_name": "geopotential_height",
                },
            )
            break
        else:
            logger.warning(
                "No first guess file with z_ifc found, the output will have no "
                "geopotential_height"
            )
        return self._heights

    def process(self, ds: xr.Dataset, context: ProcessingContext) -> xr.Dataset:
        icon = context.config.icon
        if icon is None:
            raise ValueError("The ICON post-processor needs an [icon] configuration")

        regridder = self._regridder(context, icon.regrid.method)
        cell_dim = find_cell_dim(ds, regridder.ncells)

        ds = ds.drop_vars(
            [name for name in ds.variables if str(name).endswith("_bnds")],
            errors="ignore",
        )

        # Vertical dimensions: ICON names them height, height_2, ... in the order they
        # come up. Tell them apart by their size.
        level_dims = [
            str(dim)
            for dim in ds.dims
            if str(dim).startswith("height") or str(dim) in ("lev", "alt")
        ]
        sizes = sorted({ds.sizes[dim] for dim in level_dims if ds.sizes[dim] > 1})
        nlev = sizes[0] if sizes else None
        for dim in level_dims:
            size = ds.sizes[dim]
            if size == 1:
                # 2 m temperature, 10 m wind, ...
                ds = ds.squeeze(dim, drop=True)
            elif size == nlev:
                ds = ds.drop_vars(dim, errors="ignore").rename({dim: "z"})
            elif nlev is not None and size == nlev + 1:
                # Half levels: average to full levels
                ds = ds.drop_vars(dim, errors="ignore")
                staggered = [n for n, v in ds.data_vars.items() if dim in v.dims]
                for name in staggered:
                    var = ds[name]
                    upper = var.isel({dim: slice(0, -1)})
                    lower = var.isel({dim: slice(1, None)})
                    full = (0.5 * (upper + lower.values)).rename({dim: "z"})
                    full.attrs = var.attrs
                    ds = ds.drop_vars(name).assign({name: full.astype(var.dtype)})
                if dim in ds.dims:
                    ds = ds.drop_dims(dim)
        if "z" in ds.dims:
            # ICON counts levels from the top. An index array, not a slice with a
            # negative step: xarray gets those wrong on variables it has not loaded.
            ds = ds.isel(z=np.arange(ds.sizes["z"])[::-1])

        heights = self._level_heights(context, cell_dim, regridder.ncells)
        if (
            heights is not None
            and "z" in ds.dims
            and ds.sizes["z"] == heights.sizes["z"]
            and "z_mc" not in ds
        ):
            ds = ds.assign(z_mc=heights)

        ds = ds.rename({k: v for k, v in RENAMES.items() if k in ds and v not in ds})

        variables_to_keep = context.config.postprocess.variables_to_keep
        if variables_to_keep:
            patterns = [re.compile(v) for v in variables_to_keep]
            drop = [
                name
                for name in ds.data_vars
                if not any(p.match(str(name)) for p in patterns)
            ]
            ds = ds.drop_vars(drop)

        # Interpolate, each variable with its own method
        methods: dict[str, list[str]] = {}
        for name, var in ds.data_vars.items():
            if cell_dim in var.dims:
                methods.setdefault(self._method_for(context, str(name)), []).append(
                    str(name)
                )
        untouched = ds[[n for n, v in ds.data_vars.items() if cell_dim not in v.dims]]
        parts = [untouched.drop_dims(cell_dim, errors="ignore")]
        threads = icon.regrid.threads
        for method, names in methods.items():
            parts.append(
                self._regridder(context, method).regrid_dataset(
                    ds[names], cell_dim=cell_dim, threads=threads
                )
            )
        out = xr.merge(parts, combine_attrs="drop_conflicts")
        out.attrs = dict(ds.attrs)
        # Points outside the model domain are NaN. Give the variables a fill value, so
        # that the writers store them as missing: the lossy compression would turn a
        # NaN into zero.
        for names in methods.values():
            for name in names:
                dtype = out[name].dtype
                if dtype.kind == "f":
                    out[name].attrs["_FillValue"] = dtype.type(
                        netCDF4.default_fillvals[f"f{dtype.itemsize}"]
                    )

        # Layout of the rest of the pipeline
        out = out.rename({"lat": "y", "lon": "x"})
        if "time" in out.dims:
            out = out.rename({"time": "t"})
        elif "time" in out.coords:
            out = out.rename({"time": "t"}).expand_dims("t")
        out["t"].attrs.pop("bounds", None)
        out = out.assign_coords(
            longitude=("x", out["x"].values, out["x"].attrs),
            latitude=("y", out["y"].values, out["y"].attrs),
        )
        if "z" in out.dims:
            out = out.assign_coords(z=np.arange(out.sizes["z"]))
            out["z"].attrs = {"long_name": "Model level, counted from the ground"}
        order = [d for d in ("t", "z", "y", "x") if d in out.dims]
        out = out.transpose(*order, ...)
        out.attrs["regrid_method"] = icon.regrid.method
        return out
