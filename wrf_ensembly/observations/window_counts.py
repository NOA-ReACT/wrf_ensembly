"""How many observations an assimilation window would actually take in.

Sweeping the window length before committing to one shows the trade between
observation count and how far from the analysis time observations are drawn.
"""

from pathlib import Path
from typing import Iterable

import click
import matplotlib.pyplot as plt
import pandas as pd
from rich.progress import track
from rich.table import Table

from wrf_ensembly.console import console
from wrf_ensembly.observations import io as obs_io
from wrf_ensembly.superobs import grid_bin

DEFAULT_WINDOWS_H = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
"""Window lengths swept when none are given, in hours."""


def _parse_bin_spec(raw: tuple[str, ...]) -> dict[str, int]:
    """
    Parse repeated ``dim=size`` options into a bin specification.

    Args:
        raw: Strings such as ``along_track=27``.

    Returns:
        Mapping of native grid dimension name to bin width in grid steps.

    Raises:
        click.BadParameter: If an entry is malformed or the size is not a
            positive integer.
    """
    spec: dict[str, int] = {}
    for entry in raw:
        if "=" not in entry:
            raise click.BadParameter(
                f"Invalid bin specification '{entry}'. Expected 'dimension=size'."
            )
        dim, _, size = entry.partition("=")
        dim = dim.strip()
        try:
            value = int(size)
        except ValueError:
            raise click.BadParameter(
                f"Bin size for '{dim}' must be an integer, got '{size}'."
            )
        if value < 1:
            raise click.BadParameter(
                f"Bin size for '{dim}' must be at least 1, got {value}."
            )
        spec[dim] = value
    return spec


def superob_files(
    files: Iterable[Path],
    instrument: str,
    quantity: str,
    hoz_bins: dict[str, int],
    vert_bins: dict[str, int],
    qc_flags: Iterable[int] = (0,),
    reduce_instrument_error: bool = True,
) -> pd.DataFrame:
    """
    Read observation files and superob them the way the ingest pipeline does.

    Files are binned one at a time, matching how `add_observation_file` calls
    `grid_bin`, so the counts reflect what ingest would produce.

    Args:
        files: Observation files to read.
        instrument: Instrument to keep.
        quantity: Quantity to keep.
        hoz_bins: Horizontal bin sizes, keyed by native dimension name.
        vert_bins: Vertical bin sizes, keyed by native dimension name.
        qc_flags: Quality control flags to keep.
        reduce_instrument_error: Passed through to `grid_bin`.

    Returns:
        The superobservations from every file, concatenated. Empty if nothing
        matched.
    """
    qc_flags = list(qc_flags)
    binned = []

    for path in track(list(files), description="Superobbing files..."):
        df = obs_io.read_obs(path)
        df = df[
            (df["instrument"] == instrument)
            & (df["quantity"] == quantity)
            & (df["qc_flag"].isin(qc_flags))
        ]
        if df.empty:
            continue

        try:
            superobs = grid_bin(df, hoz_bins, vert_bins, reduce_instrument_error)
        except ValueError as e:
            raise click.ClickException(
                f"{path.name}: {e} Omit --hoz-bin/--vert-bin to count the "
                "observations as they are."
            )

        superobs["source_file"] = path.name
        binned.append(superobs)

    if not binned:
        return pd.DataFrame()
    return pd.concat(binned, ignore_index=True)


def count_by_window(
    times: pd.Series,
    centres: pd.DatetimeIndex,
    window_hours: Iterable[float],
) -> pd.DataFrame:
    """
    Count observations inside windows of varying length around each analysis time.

    Windows are half-open, ``[centre - w/2, centre + w/2)``.

    Args:
        times: Observation timestamps.
        centres: Analysis times to centre the windows on.
        window_hours: Total window lengths, in hours.

    Returns:
        Frame indexed by window length, with one column per analysis time plus
        `total` and `mean_per_cycle`.
    """
    times = pd.to_datetime(pd.Series(times))
    if times.dt.tz is None:
        times = times.dt.tz_localize("UTC")
    else:
        times = times.dt.tz_convert("UTC")

    centre_labels = [c.strftime("%Y-%m-%d %H:%M") for c in centres]
    rows = {}

    for window in window_hours:
        half = pd.Timedelta(hours=window / 2)
        rows[window] = [
            int(((times >= centre - half) & (times < centre + half)).sum())
            for centre in centres
        ]

    result = pd.DataFrame(rows, index=centre_labels).T
    result.index.name = "window_hours"
    result["total"] = result[centre_labels].sum(axis=1)
    result["mean_per_cycle"] = result[centre_labels].mean(axis=1)
    return result


@click.command(name="window-counts")
@click.argument(
    "input_files", nargs=-1, required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--instrument", required=True, help="Instrument to count.")
@click.option("--quantity", required=True, help="Quantity to count.")
@click.option(
    "--analysis-start",
    type=click.DateTime(),
    required=True,
    help="First analysis time (UTC).",
)
@click.option(
    "--analysis-end",
    type=click.DateTime(),
    required=True,
    help="Last analysis time (UTC), inclusive.",
)
@click.option(
    "--analysis-interval",
    type=int,
    required=True,
    help="Spacing between analysis times, in minutes.",
)
@click.option(
    "--window",
    "windows",
    multiple=True,
    type=float,
    help="Total assimilation window length in hours. Can be specified multiple "
    f"times. Defaults to {' '.join(str(w) for w in DEFAULT_WINDOWS_H)}.",
)
@click.option(
    "--hoz-bin",
    "hoz_bin",
    multiple=True,
    metavar="DIM=SIZE",
    help="Horizontal superob bin size in native grid steps, e.g. along_track=27. "
    "Can be specified multiple times. Omit to skip superobbing.",
)
@click.option(
    "--vert-bin",
    "vert_bin",
    multiple=True,
    metavar="DIM=SIZE",
    help="Vertical superob bin size in native grid steps, e.g. JSG_height=3. "
    "Can be specified multiple times.",
)
@click.option(
    "--qc-flag",
    "qc_flags",
    multiple=True,
    type=int,
    default=(0,),
    show_default=True,
    help="Quality control flags to keep. Can be specified multiple times.",
)
@click.option(
    "--no-reduce-instrument-error",
    is_flag=True,
    help="Do not reduce the superob instrument error by sqrt(n). Set this when "
    "in-bin errors are correlated, as for heavily smoothed retrievals.",
)
@click.option(
    "--output-csv",
    type=click.Path(path_type=Path),
    default=Path("obs_window_counts.csv"),
    show_default=True,
)
@click.option(
    "--output-plot",
    type=click.Path(path_type=Path),
    default=Path("obs_window_counts.png"),
    show_default=True,
)
def window_counts(
    input_files: tuple[Path, ...],
    instrument: str,
    quantity: str,
    analysis_start,
    analysis_end,
    analysis_interval: int,
    windows: tuple[float, ...],
    hoz_bin: tuple[str, ...],
    vert_bin: tuple[str, ...],
    qc_flags: tuple[int, ...],
    no_reduce_instrument_error: bool,
    output_csv: Path,
    output_plot: Path,
):
    """
    Count observations falling inside assimilation windows of varying length.

    Sweeps a range of window lengths centred on a series of analysis times and
    reports how many observations each would take in, to inform the choice of
    assimilation window before an experiment is set up.

    Superobbing uses the same code path as observation ingest, so the counts
    reflect what would actually reach the filter. Analysis times are given
    explicitly here since there is no experiment configuration to read them from.
    """

    window_hours = sorted(windows) if windows else list(DEFAULT_WINDOWS_H)
    hoz_bins = _parse_bin_spec(hoz_bin)
    vert_bins = _parse_bin_spec(vert_bin)

    centres = pd.date_range(
        pd.Timestamp(analysis_start, tz="UTC"),
        pd.Timestamp(analysis_end, tz="UTC"),
        freq=pd.Timedelta(minutes=analysis_interval),
    )
    if len(centres) == 0:
        raise click.BadParameter(
            "No analysis times in range; check --analysis-start/--analysis-end."
        )

    if hoz_bins or vert_bins:
        observations = superob_files(
            input_files,
            instrument,
            quantity,
            hoz_bins,
            vert_bins,
            qc_flags,
            not no_reduce_instrument_error,
        )
        label = "superobservations"
    else:
        frames = []
        for path in track(list(input_files), description="Reading files..."):
            df = obs_io.read_obs(path)
            df = df[
                (df["instrument"] == instrument)
                & (df["quantity"] == quantity)
                & (df["qc_flag"].isin(list(qc_flags)))
            ]
            if not df.empty:
                frames.append(df)
        observations = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        label = "observations"

    if observations.empty:
        print(f"No observations matched {instrument}.{quantity}")
        return

    print(f"{len(observations):,} {label} across {len(input_files)} file(s)")
    print(f"{len(centres)} analysis time(s), {len(window_hours)} window length(s)")

    counts = count_by_window(observations["time"], centres, window_hours)
    counts.to_csv(output_csv)

    table = Table(title=f"{instrument}.{quantity} per analysis time")
    table.add_column("Window (h)", justify="right")
    table.add_column(f"Mean {label} per cycle", justify="right")
    table.add_column("Total", justify="right")
    for window in window_hours:
        table.add_row(
            f"{window:g}",
            f"{counts.loc[window, 'mean_per_cycle']:,.0f}",
            f"{counts.loc[window, 'total']:,}",
        )
    console.print(table)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(window_hours, counts["mean_per_cycle"], "o-")
    ax.set_xlabel("Assimilation window length (h, total)")
    ax.set_ylabel(f"{label.capitalize()} per analysis time")
    ax.set_title(f"{instrument}.{quantity}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {output_csv} and {output_plot}")
