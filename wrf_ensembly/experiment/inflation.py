import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wrf_ensembly import config, utils
from wrf_ensembly.console import logger


@dataclass
class InflationConfig:
    """
    Resolved inflation settings for an experiment.

    Inflation is controlled by the following settings:
    - `cfg.assimilation.use_inflation` enabled the ensembly's handling of inflation files
    - `cfg.dart_namelist["filter_nml"]["inf_flavor"]` controls what kind of prior and posterior
      inflation DART will use. Zero values means disabled.
    """

    enabled: bool
    prior_enabled: bool
    posterior_enabled: bool
    dart_work_dir: Path
    data_inflation_dir: Path

    @classmethod
    def from_config(
        cls, cfg: config.Config, data_inflation_dir: Path
    ) -> "InflationConfig":
        dart_work_dir = cfg.directories.dart_root / "models" / "wrf" / "work"

        if not cfg.assimilation.use_inflation:
            return cls(
                enabled=False,
                prior_enabled=False,
                posterior_enabled=False,
                dart_work_dir=dart_work_dir,
                data_inflation_dir=data_inflation_dir,
            )

        inf_flavor = cfg.dart_namelist["filter_nml"]["inf_flavor"]
        return cls(
            enabled=True,
            prior_enabled=inf_flavor[0] > 0,
            posterior_enabled=inf_flavor[1] > 0,
            dart_work_dir=dart_work_dir,
            data_inflation_dir=data_inflation_dir,
        )

    @property
    def active_files(self) -> list[str]:
        """The inflation restart filenames DART will produce/consume."""
        files = []
        if self.prior_enabled:
            files += ["output_priorinf_mean.nc", "output_priorinf_sd.nc"]
        if self.posterior_enabled:
            files += ["output_postinf_mean.nc", "output_postinf_sd.nc"]
        return files

    def apply_namelist_overrides(
        self, dart_namelist: dict[str, dict[str, Any]], cycle_i: int
    ) -> dict[str, dict[str, Any]]:
        """
        Apply inflation-related overrides to the DART namelist.

        For cycle 0, restart files don't exist yet so from_restart is always False.
        For subsequent cycles, from_restart matches which inflation types are enabled.
        """
        dart_namelist = copy.deepcopy(dart_namelist)

        if not self.enabled:
            return dart_namelist

        if cycle_i == 0:
            logger.info("First cycle, not using inflation restart files")
            dart_namelist["filter_nml"]["inf_initial_from_restart"] = [False, False]
            dart_namelist["filter_nml"]["inf_sd_initial_from_restart"] = [False, False]
        else:
            dart_namelist["filter_nml"]["inf_initial_from_restart"] = [
                self.prior_enabled,
                self.posterior_enabled,
            ]
            dart_namelist["filter_nml"]["inf_sd_initial_from_restart"] = [
                self.prior_enabled,
                self.posterior_enabled,
            ]

        return dart_namelist

    def _stashed_path(self, cycle_i: int, filename: str) -> Path:
        """Path where an inflation file is stored between cycles."""
        return self.data_inflation_dir / f"cycle_{cycle_i:03d}_{filename}"

    def stash_restart_files(self, cycle_i: int):
        """
        Move the current cycle's inflation output files from the DART work
        directory to the data directory for use in the next cycle.
        """
        if not self.enabled:
            return

        for filename in self.active_files:
            src = self.dart_work_dir / filename
            dst = self._stashed_path(cycle_i, filename)
            if src.exists():
                src.rename(dst)
                logger.info(f"Stashed inflation file {filename} for cycle {cycle_i}")
            else:
                logger.warning(
                    f"Inflation enabled but filter didn't produce {filename}!"
                )

    def pop_restart_files(self, cycle_i: int):
        """
        Restore the previous cycle's inflation output files from the data
        directory back into the DART work directory as input for the current cycle.
        """
        if not self.enabled:
            return

        # No last cycle to use for the first cycle
        if cycle_i == 0:
            return

        last_cycle = cycle_i - 1
        for filename in self.active_files:
            src = self._stashed_path(last_cycle, filename)
            dst = self.dart_work_dir / filename.replace("output", "input")
            if not src.exists():
                msg = f"Inflation is enabled but last cycle's inflation file wasn't found at {src}"
                logger.error(msg)
                raise FileNotFoundError(msg)
            utils.copy(src, dst)
            logger.info(f"Restored inflation file {filename} from cycle {last_cycle}")
