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
        cls, cfg: config.Config, data_inflation_dir: Path, dart_work_dir: Path
    ) -> "InflationConfig":
        if not cfg.assimilation.use_inflation:
            return cls(
                enabled=False,
                prior_enabled=False,
                posterior_enabled=False,
                dart_work_dir=dart_work_dir,
                data_inflation_dir=data_inflation_dir,
            )

        inf_flavor = cfg.dart_namelist["filter_nml"]["inf_flavor"]
        if len(inf_flavor) < 2:
            raise ValueError(
                f"filter_nml.inf_flavor must have at least 2 elements, got {inf_flavor}"
            )

        prior_enabled = inf_flavor[0] > 0
        posterior_enabled = inf_flavor[1] > 0
        if not prior_enabled and not posterior_enabled:
            raise ValueError(
                "use_inflation is True but filter_nml.inf_flavor is [0, 0] — "
                "set use_inflation = false or configure a non-zero inf_flavor"
            )

        return cls(
            enabled=True,
            prior_enabled=prior_enabled,
            posterior_enabled=posterior_enabled,
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

    def prepare_cycle(
        self, dart_namelist: dict[str, dict[str, Any]], cycle_i: int
    ) -> dict[str, dict[str, Any]]:
        """
        Prepare inflation for a cycle: restore restart files from a previous cycle
        and apply the appropriate namelist overrides.

        Searches backwards through cycles to find the most recent one with inflation
        files. If none are found, configures DART to generate initial inflation values.

        Returns the modified DART namelist.
        """

        dart_namelist = copy.deepcopy(dart_namelist)

        if not self.enabled:
            return dart_namelist

        restart_cycle = self.pop_restart_files(cycle_i)
        use_restart = restart_cycle is not None

        if not use_restart:
            logger.info(
                "No inflation restart files available, DART will generate initial values"
            )
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
                # Renaming won't work across filesystems, backup plan here
                if src.stat().st_dev != dst.stat().st_dev:
                    src.copy(dst)
                    src.unlink()
                else:
                    src.rename(dst)
                logger.info(f"Stashed inflation file {filename} for cycle {cycle_i}")
            else:
                logger.warning(
                    f"Inflation enabled but filter didn't produce {filename}!"
                )

    def pop_restart_files(self, cycle_i: int) -> int | None:
        """
        Restore a previous cycle's inflation output files from the data
        directory back into the DART work directory as input for the current cycle.

        Searches backwards through cycles to find the most recent one with
        complete inflation files. Returns the cycle index used, or None if no
        files were found (including cycle 0 where no prior cycles exist).
        """

        if not self.enabled:
            return None

        if cycle_i == 0:
            return None

        if not self.active_files:
            return None

        # Search backwards to find the most recent cycle with inflation files
        for source_cycle in range(cycle_i - 1, -1, -1):
            all_exist = all(
                self._stashed_path(source_cycle, f).exists() for f in self.active_files
            )
            if all_exist:
                for filename in self.active_files:
                    src = self._stashed_path(source_cycle, filename)
                    dst = self.dart_work_dir / filename.replace("output", "input")
                    dst.unlink(missing_ok=True)
                    dst.symlink_to(src)
                    logger.info(
                        f"Linked inflation file {filename} from cycle {source_cycle}"
                    )
                return source_cycle

        return None
