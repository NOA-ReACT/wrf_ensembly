"""Doing some MSG serivi tests"""

# %%
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import satpy
import xarray as xr

xr.set_options(display_style="text", display_max_rows=100, display_expand_attrs=False)

# %%
fpath = "/mnt/wsl/observations/msg_hr_seviri/2021-08/MSG4-SEVI-MSG15-0100-NA-20210801001242.788000000Z-NA.nat"
scn = satpy.Scene(filenames=[fpath], reader="seviri_l1b_native")
scn.load(["WV_062", "WV_073", "IR_087", "IR_108", "IR_120"])

# %%
scn["IR_087"].plot()

# %%
scn["IR_087"].attrs["start_time"]

# %% Read an ensemble mean file to get some parameters
cycle = 59
wrf = (
    xr.open_dataset(
        # f"/home/thgeorgiou/experiments/aeolus_control/data/forecasts/cycle_{cycle:03d}/forecast_mean_cycle_{cycle:03d}.nc"
        "/home/thgeorgiou/code/tools/rttov_operator/forecast_mean_cycle_085.out.nc"
    )
    .isel(t=0)
    .set_coords(["latitude", "longitude"])
)
wrf
# %%
from pyresample.utils import create_area_def

target_area = create_area_def(
    "wrf",
    {
        "x_0": 0,
        "y_0": 0,
        "a": 6370000,
        "b": 6370000,
        "proj": "lcc",
        "lat_1": wrf.attrs["TRUELAT1"],
        "lat_2": wrf.attrs["TRUELAT2"],
        "lat_0": wrf.attrs["CEN_LAT"],
        "lon_0": wrf.attrs["STAND_LON"],
    },
    resolution=40e3,
    area_extent=[wrf.x.min(), wrf.y.min(), wrf.x.max(), wrf.y.max()],
)
target_area
# %%
variables_msg = ["WV_062", "WV_073", "IR_087", "IR_108", "IR_120"]
variables_wrf = ["WV62", "WV73", "IR87", "IR108", "IR120"]

# %%
resampled = scn.resample(target_area, resampler="bucket_avg")
lons_t, lats_t = target_area.get_lonlats()

bts = {
    var: xr.DataArray(
        np.array(resampled[var]),
        dims=["y", "x"],
        coords={"lat": (["y", "x"], lats_t), "lon": (["y", "x"], lons_t)},
        attrs={"units": "K", "long_name": f"{var} BT"},
    )
    for var in variables_msg
}
# %% Plot the two grid in a map/scatter plot
fig, ax = plt.subplots(
    1, 1, figsize=(10, 10), subplot_kw={"projection": target_area.to_cartopy_crs()}
)
ax.scatter(wrf.longitude, wrf.latitude, s=1, transform=ccrs.PlateCarree(), label="WRF")
ax.scatter(
    bts["IR_108"].lon, bts["IR_108"].lat, s=1, transform=ccrs.PlateCarree(), label="MSG"
)
ax.set_extent(
    [-30, -15, 10, 15], crs=ccrs.PlateCarree()
)  # Focus around Cape Verde and the coast to actually see the points
ax.coastlines()
ax.legend()


# %%
limits = {
    "WV62": (200, 260),
    "WV73": (210, 270),
    "IR87": (200, 310),
    "IR108": (200, 310),
    "IR120": (200, 310),
}

fig, axes = plt.subplots(
    len(variables_msg),
    2,
    figsize=(15, 12),
    subplot_kw={"projection": target_area.to_cartopy_crs()},
)

for i, (msg_var, wrf_var) in enumerate(zip(variables_msg, variables_wrf)):
    # vmin = bts[msg_var].min()
    # vmax = bts[msg_var].max()
    vmin, vmax = limits[wrf_var]
    bts[msg_var].plot(
        x="lon",
        y="lat",
        ax=axes[i, 0],
        # cmap="RdBu_r",
        cmap="RdYlBu_r",
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )
    wrf[wrf_var].plot(
        x="longitude",
        y="latitude",
        ax=axes[i, 1],
        cmap="RdYlBu_r",
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )

for ax in axes.flat:
    ax.coastlines()

plt.tight_layout()

# %%
for var in variables_wrf:
    print(var, wrf[var].mean().item())
