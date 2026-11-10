"""
ICON behind the `Model` interface. The work is done by `wrf_ensembly.icon`, see
`icon/model.py` for how a cycle goes.
"""

import datetime as dt
from collections.abc import Callable
from pathlib import Path

from wrf_ensembly import cycling
from wrf_ensembly.console import logger
from wrf_ensembly.icon import model as icon_model
from wrf_ensembly.icon import namelists

from .base import Model, ModelRun


class IconModel(Model):
    name = "icon"

    @property
    def environment(self) -> dict[str, str]:
        return self.cfg.environment.icon

    def install(self, force: bool) -> None:
        # ICON runs from its build directory: `ensemble setup` links the executable
        # into every member's directory, there are no tables to copy
        executable = icon_model.executable(self.exp)
        if not executable.exists():
            raise FileNotFoundError(f"No ICON executable at {executable}")
        logger.info(f"Using {executable}, nothing to copy for ICON")

    def setup_member(self, member_i: int) -> None:
        icon_model.setup_member(self.exp, member_i)
        missing = icon_model.missing_boundary_files(self.exp, member_i)
        if missing:
            logger.warning(
                f"Member {member_i}: {len(missing)} boundary file(s) are missing, "
                f"the first is {missing[0]}"
            )
        logger.info(f"Member {member_i}: Linked the model and its input files")

    def preprocess_job(self, command: Callable[..., str]) -> list[str]:
        return [command("icon-icbc")]

    def run_member(
        self,
        member_i: int,
        segment: range,
        cycle: cycling.CycleInformation,
        cores: int,
    ) -> ModelRun | None:
        executable = self.paths.member_path(member_i) / icon_model.EXECUTABLE
        if not executable.exists():
            logger.error(
                f"Member {member_i}: ICON executable not found at {executable}, "
                "run `ensemble setup` first"
            )
            return None
        ok, start, end = icon_model.run_member(self.exp, member_i, cycle, cores)
        if not ok:
            return None
        return ModelRun(start, end, int((cycle.end - cycle.start).total_seconds()))

    @property
    def forecast_glob(self) -> str:
        return f"{namelists.FORECAST_STREAM}_*.nc"

    def time_stamp(self, time: dt.datetime) -> str:
        return namelists.file_date(time)

    def forecast_file_name(self, time: dt.datetime) -> str:
        return f"{namelists.FORECAST_STREAM}_{self.time_stamp(time)}.nc"

    def cycle_end_file_name(self, time: dt.datetime) -> str:
        # The first guess, not the forecast output
        return f"{namelists.FIRST_GUESS_STREAM}_{self.time_stamp(time)}.nc"

    def first_processor(self):
        # Imported here, the regridding is only needed for the postprocessing
        from wrf_ensembly.icon.processor import IconPostProcessor

        return IconPostProcessor()

    def prepare_filter(self) -> list[Path]:
        return icon_model.prepare_filter(self.exp)

    def make_analysis(self, member_i: int) -> None:
        icon_model.make_analysis(self.exp, member_i)

    def cycle_member(self, member_i: int, use_forecast: bool) -> None:
        icon_model.cycle_member(self.exp, member_i, use_forecast)

    def update_bc(self, member_i: int) -> None:
        # ICON takes the state at the start of a run as the boundary data for that
        # time (init_latbc_from_fg) and relaxes towards the next boundary file
        logger.info("Nothing to do for ICON, the boundary files are used as they are")

    def clean_old_state(self) -> None:
        icon_model.clean_first_guess_files(self.exp)
