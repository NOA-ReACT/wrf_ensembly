from pathlib import Path
from typing import Any, Optional

import tomli
import tomli_w

from wrf_ensembly import config, cycling, member_info, utils
from wrf_ensembly.console import logger
from wrf_ensembly.utils import filter_none_from_dict


class EnsembleMember:
    """Represents a member of the ensemble"""

    i: int
    """Index of the member"""

    current_cycle_i: int
    """Currentcycle of the member, i.e. which cycle should now run"""

    path: Path
    """Path to the member directory"""

    minfo_path: Path
    """Path to the member info toml file"""

    metadata: dict[str, str]
    """Metadata stored in the member info file"""

    cycles: dict[int, member_info.CycleSection]
    """History of the member, i.e. all cycles that have been run"""

    @property
    def current_cycle(self) -> member_info.CycleSection:
        """
        Returns the current cycle of the member.
        """
        return self.cycles[self.current_cycle_i]

    @current_cycle.setter
    def current_cycle(self, cycle: member_info.CycleSection):
        """Update the current cycle"""
        self.cycles[self.current_cycle_i] = cycle

    def __init__(self, experiment_path: Path, i: int, cycle_count: int):
        self.i = i
        self.path = experiment_path / "work" / "ensemble" / f"member_{self.i:02d}"
        self.minfo_path = (
            experiment_path / "work" / "ensemble" / f"member_{self.i:02d}.toml"
        )

        # Read member info, or initialize it if it does not exist
        try:
            self.read_minfo()
        except FileNotFoundError:
            self.current_cycle_i = 0
            self.metadata = {}
            self.cycles = {
                i: member_info.CycleSection(
                    runtime=None,
                    walltime_s=None,
                    advanced=False,
                    filter=False,
                    analysis=False,
                )
                for i in range(cycle_count)
            }

    def read_minfo(self):
        """
        Reads the member info file for this member.
        """
        with open(self.minfo_path, "rb") as f:
            minfo = tomli.load(f)

        minfo = member_info.MemberInfo(**minfo)

        self.current_cycle_i = minfo.member.current_cycle
        self.metadata = minfo.metadata
        self.cycles = minfo.cycle

    def write_minfo(self):
        """
        Writes the member info file for this member.
        """
        minfo = member_info.MemberInfo(
            member=member_info.MemberSection(
                i=self.i, current_cycle=self.current_cycle_i
            ),
            metadata=self.metadata,
            cycle=self.cycles,
        )
        cycle = {
            str(k): filter_none_from_dict(v.dict()) for k, v in minfo.cycle.items()
        }

        with utils.atomic_binary_open(self.minfo_path) as f:
            tomli_w.dump(minfo.dict() | {"cycle": cycle}, f)
        logger.info(f"Member {self.i}: Wrote info file to {self.minfo_path}")


class ExperimentPaths:
    """
    Paths to the different directories of an experiment
    """

    def __init__(self, experiment_path: Path, cfg: config.Config):
        self.experiment_path = experiment_path.resolve()
        self.work_path = experiment_path / "work"
        self.ensemble_path = self.work_path / "ensemble"
        self.jobfiles = experiment_path / "jobfiles"

        # Data directories
        self.data = experiment_path / cfg.directories.output_sub
        self.data_icbc = self.data / "initial_boundary"
        self.data_forecasts = self.data / "forecasts"
        self.data_dart = self.data / "dart"
        self.data_analysis = self.data / "analysis"
        self.data_diag = self.data / "diagnostics"

        self.obs = experiment_path / "obs"

        # Work directories
        self.work = experiment_path / cfg.directories.work_sub
        self.work_wrf = self.work / "WRF"
        self.work_wps = self.work / "WPS"
        self.work_ensemble = self.work / "ensemble"

        # Preprocessing
        self.work_preprocessing = self.work / "preprocessing"
        self.work_preprocessing_wrf = self.work_preprocessing / "WRF"
        self.work_preprocessing_wps = self.work_preprocessing / "WPS"

    def member_path(self, i: int) -> Path:
        """
        Get the work directory for given ensemble member
        """
        return self.ensemble_path / f"member_{i:02d}"

    def forecast_path(
        self, cycle: Optional[int] = None, member: Optional[int] = None
    ) -> Path:
        if cycle is None:
            return self.data_forecasts
        if member is None:
            return self.data_forecasts / f"cycle_{cycle:03d}"
        return self.data_forecasts / f"cycle_{cycle:03d}" / f"member_{member:02d}"

    def analysis_path(self, cycle: Optional[int] = None) -> Path:
        if cycle is None:
            return self.data_analysis
        return self.data_analysis / f"cycle_{cycle:03d}"

    def dart_path(self, cycle: Optional[int] = None) -> Path:
        if cycle is None:
            return self.data_dart
        return self.data_dart / f"cycle_{cycle:03d}"


class Experiment:
    """
    An ensemble assimilation experiment
    """

    cfg: config.Config
    cycles: list[cycling.CycleInformation]
    paths: ExperimentPaths
    members: list[EnsembleMember] = []

    def __init__(self, experiment_path: Path):
        self.cfg = config.read_config(experiment_path / "config.toml")
        self.cycles = cycling.get_cycle_information(self.cfg)

        for i in range(self.cfg.assimilation.n_members):
            self.members.append(EnsembleMember(experiment_path, i, len(self.cycles)))

        self.paths = ExperimentPaths(experiment_path, self.cfg)

    def ensure_same_cycle(self):
        """
        Ensures that all members have the same current cycle. Raises a ValueError otherwise.
        """
        for m in self.members:
            if m.current_cycle_i != self.members[0].current_cycle_i:
                raise ValueError(
                    f"Member {m.i} has cycle {m.current_cycle} but member 0 has cycle {self.members[0].current_cycle}"
                )

    def ensure_current_cycle_state(self, state: dict[str, Any]):
        """Ensures that all members have the same state for the current cycle"""

        logger.debug(f"Checking state for cycle {self.members[0].current_cycle_i}")

        self.ensure_same_cycle()
        for m in self.members:
            cycle_info = m.current_cycle.dict()
            for k, v in state.items():
                if k not in cycle_info or cycle_info[k] != v:
                    raise ValueError(
                        f"Member {m.i} has a different {k} than the expected {v}"
                    )

    def write_all_member_info(self):
        for m in self.members:
            m.write_minfo()
