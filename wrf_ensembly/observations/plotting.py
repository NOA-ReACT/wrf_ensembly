from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs


def plot_observation_locations_on_map(
    observations: pd.DataFrame,
    proj: ccrs.Projection | None,
    domain_bounds: tuple[float, float, float, float] | None = None,
    fig_kwargs: dict = {},
    subplot_kwargs: dict = {},
    ax_kwargs: dict = {},
) -> Figure:
    """
    Plot the locations of a set of observations on a map

    It will use the columns `latitude`, `longitude` from the DataFrame. If there is
    also columns for `instrument` and `quantity`, it will use them for a legend and to
    color the points.

    Args:
        observations: DataFrame with columns `latitude`, `longitude`, optionally
            `instrument` and `quantity`.
        proj: Cartopy projection to use for the map. If None, will use PlateCarree.
        domain_bounds: If given, will set the map extent to these bounds (min_x, max_x,
            min_y, max_y). If None, the extent will be set automatically to contain the
            observations. It must be in `proj` if provided, otherwise in PlateCarree.
        fig_kwargs: Additional keyword arguments to pass to `plt.figure()`
        subplot_kwargs: Additional keyword arguments to pass to `plt.subplot()`
        ax_kwargs: Additional keyword arguments to pass to `ax.scatter()`

    Returns:
        A matplotlib Figure object with the plot.
    """

    if "latitude" not in observations.columns:
        raise ValueError("DataFrame must have a 'latitude' column")
    if "longitude" not in observations.columns:
        raise ValueError("DataFrame must have a 'longitude' column")

    fig, ax = plt.subplots(
        subplot_kw={"projection": proj or ccrs.PlateCarree(), **subplot_kwargs},
        figsize=(10, 8),
        **fig_kwargs,
    )
    ax.coastlines()

    if "instrument" in observations.columns and "quantity" in observations.columns:
        for (instrument, quantity), group in observations.groupby(
            ["instrument", "quantity"]
        ):
            label = f"{instrument} - {quantity}"
            ax.scatter(
                group["longitude"],
                group["latitude"],
                label=label,
                transform=ccrs.PlateCarree(),
                s=0.1,
                **ax_kwargs,
            )

        # Create a nice legend outside the plot
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(1, 0.5),
            title="Instrument - Quantity",
            fontsize="small",
            title_fontsize="medium",
            frameon=False,
        )

    else:
        ax.scatter(
            observations["longitude"],
            observations["latitude"],
            transform=ccrs.PlateCarree(),
            s=0.1,
            **ax_kwargs,
        )

    if domain_bounds is not None:
        ax.set_extent(domain_bounds, crs=proj or ccrs.PlateCarree())

    return fig
