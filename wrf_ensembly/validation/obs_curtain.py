"""Observation-space curtains for profile instruments.

Draws what the filter saw and what it did, on the instrument's own geometry: the
observed field, the model prior and analysis interpolated to the same pixels, and
the increment between them.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from wrf_ensembly.console import logger
from wrf_ensembly.experiment.experiment import Experiment
from wrf_ensembly.observations.utils import reconstruct_curtain

PANEL_COLUMNS = ["value", "model_forecast", "model_analysis"]
"""Columns gridded for the curtain; the increment is derived from the last two."""


class ObsCurtainAnalysis:
    """
    Draws observed / prior / analysis / increment curtains for a profile instrument.

    One figure per source observation file, since `reconstruct_array` needs a single
    native grid and each file carries its own. Requires `validation
    interpolate-model` to have populated the model columns.
    """

    def __init__(
        self,
        experiment: Experiment,
        instrument: str,
        quantity: str,
        z_min: float | None = None,
        z_max: float | None = None,
    ):
        """
        Args:
            experiment: The experiment to analyse.
            instrument: The instrument name.
            quantity: The observed quantity.
            z_min: If set, drop vertical bins below this altitude.
            z_max: If set, drop vertical bins above this altitude. Useful when the
                instrument reports to the top of the atmosphere but the signal of
                interest is confined to the lower troposphere.
        """
        self.exp = experiment
        self.instrument = instrument
        self.quantity = quantity
        self.z_min = z_min
        self.z_max = z_max
        self.output_dir = (
            experiment.paths.data_validation / "obs_curtain" / instrument / quantity
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_files(self, require_analysis: bool = True) -> list[str]:
        """
        Source files that can be drawn, oldest first.

        Args:
            require_analysis: Only list files that also have an analysis, i.e. the
                cycles where the filter ran.
        """
        return self.exp.obs.get_reconstructable_filenames(
            self.instrument, self.quantity, require_model_analysis=require_analysis
        )

    def plot_file(self, orig_filename: str) -> Path | None:
        """
        Draw the curtain for one source file.

        Args:
            orig_filename: The source file to draw.

        Returns:
            Path to the saved figure, or None if the file had no usable data.

        Raises:
            ValueError: If the instrument is not curtain-shaped.
        """
        df = self.exp.obs.get_reconstructable_for_pair(
            self.instrument,
            self.quantity,
            orig_filename=orig_filename,
            require_model_forecast=True,
            require_model_analysis=True,
        )
        if df is None or df.empty:
            logger.warning(f"No prior+analysis observations in {orig_filename}")
            return None

        fields, altitude, along_dim, vertical_dim = reconstruct_curtain(
            df, value_columns=PANEL_COLUMNS
        )
        if altitude.size == 0:
            logger.warning(f"No finite altitudes in {orig_filename}, skipping")
            return None

        # Optional altitude window, applied before the NaN crop. Instruments often
        # report to the top of the atmosphere while the signal sits in the lower
        # troposphere, and the near-zero values aloft are finite, so the NaN crop
        # below cannot remove them.
        if self.z_min is not None or self.z_max is not None:
            window = np.ones(altitude.shape, dtype=bool)
            if self.z_min is not None:
                window &= altitude >= self.z_min
            if self.z_max is not None:
                window &= altitude <= self.z_max
            if not window.any():
                logger.warning(
                    f"No vertical bins within the requested altitude window for "
                    f"{orig_filename}, skipping"
                )
                return None
            altitude = altitude[window]
            fields = {k: v[:, window] for k, v in fields.items()}

        observed = fields["value"]
        prior = fields["model_forecast"]
        analysis = fields["model_analysis"]
        increment = analysis - prior

        # The track usually crosses the feature in only part of its range; crop the
        # all-NaN margins so it fills the axes. All panels get the same crop so they
        # stay registered against each other.
        along_keep = np.where(np.isfinite(observed).any(axis=1))[0]
        vert_keep = np.where(np.isfinite(observed).any(axis=0))[0]
        if along_keep.size == 0 or vert_keep.size == 0:
            logger.warning(f"No finite observations in {orig_filename}, skipping")
            return None

        along_slice = slice(along_keep.min(), along_keep.max() + 1)
        vert_slice = slice(vert_keep.min(), vert_keep.max() + 1)
        observed = observed[along_slice, vert_slice]
        prior = prior[along_slice, vert_slice]
        analysis = analysis[along_slice, vert_slice]
        increment = increment[along_slice, vert_slice]
        altitude = altitude[vert_slice]

        vmax = _upper_limit(observed, percentile=99.0)
        dlim = _symmetric_limit(increment, percentile=98.0)
        x = np.arange(observed.shape[0])

        fig, axes = plt.subplots(
            2, 2, figsize=(14, 9), sharex=True, sharey=True, layout="constrained"
        )
        panels = [
            (axes[0, 0], "observed", observed, "viridis", 0.0, vmax, self.quantity),
            (axes[0, 1], "prior (forecast)", prior, "viridis", 0.0, vmax, self.quantity),
            (axes[1, 0], "analysis", analysis, "viridis", 0.0, vmax, self.quantity),
            (
                axes[1, 1],
                "increment (analysis - prior)",
                increment,
                "RdBu_r",
                -dlim,
                dlim,
                "analysis - prior",
            ),
        ]

        for ax, title, field, cmap, vmin, vhigh, clabel in panels:
            mesh = ax.pcolormesh(
                x, altitude, field.T, cmap=cmap, vmin=vmin, vmax=vhigh, shading="nearest"
            )
            ax.set_title(title)
            fig.colorbar(mesh, ax=ax, shrink=0.9, label=clabel)

        for ax in axes[:, 0]:
            ax.set_ylabel(f"altitude ({vertical_dim})")
        for ax in axes[1, :]:
            ax.set_xlabel(along_dim)

        fig.suptitle(
            f"{self.exp.cfg.metadata.name} - {self.instrument}.{self.quantity}\n"
            f"{orig_filename}",
            fontsize=14,
            fontweight="bold",
        )

        path = self.output_dir / f"{_slugify(orig_filename)}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return path

    def run(self, filenames: list[str] | None = None) -> dict:
        """
        Draw a curtain for each requested source file.

        Args:
            filenames: Source files to draw. Defaults to every file with both a
                prior and an analysis.

        Returns:
            Dict with the output directory and the list of figures produced.
        """
        results: dict = {
            "instrument": self.instrument,
            "quantity": self.quantity,
            "output_dir": self.output_dir,
            "figures": [],
        }

        if filenames is None:
            filenames = self.list_files(require_analysis=True)
        if not filenames:
            logger.warning(
                f"No source files with both a prior and an analysis for "
                f"{self.instrument}.{self.quantity}"
            )
            logger.warning("Run `validation interpolate-model` first.")
            return results

        for orig_filename in filenames:
            try:
                path = self.plot_file(orig_filename)
            except ValueError as e:
                # Not curtain-shaped; the whole pair is unusable, so stop early
                logger.error(f"Cannot draw {self.instrument}.{self.quantity}: {e}")
                return results

            if path is not None:
                results["figures"].append(path)
                logger.info(f"Saved {path}")

        return results


def _upper_limit(values: np.ndarray, percentile: float, fallback: float = 1.0) -> float:
    """Upper colour limit from a percentile, guarding empty and all-NaN fields."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    limit = float(np.percentile(finite, percentile))
    if not np.isfinite(limit) or limit == 0.0:
        return fallback
    return limit


def _symmetric_limit(
    values: np.ndarray, percentile: float, fallback: float = 1.0
) -> float:
    """Symmetric colour limit from a percentile of |values|."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return fallback
    limit = float(np.percentile(np.abs(finite), percentile))
    if not np.isfinite(limit) or limit == 0.0:
        return fallback
    return limit


def _slugify(filename: str) -> str:
    """Filesystem-safe stem for a source filename."""
    return Path(filename).stem.replace(" ", "_")
