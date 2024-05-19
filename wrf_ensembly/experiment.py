import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from mashumaro.mixins.toml import DataClassTOMLMixin

from wrf_ensembly import config, cycling, external, wrf
from wrf_ensembly.console import logger


@dataclass
class RuntimeStatistics:
    """Contains runtime information for one cycle"""

    cycle: int
    """Which cycle are these statistics for"""

    start: dt.datetime
    """Start of model execution"""

    end: dt.datetime
    """End of model execution"""

    duration_s: int
    """Duration of model execution in seconds"""


@dataclass
class MemberStatus:
    """Status of a single ensemble member"""

    i: int
    """The member ID"""

    advanced: bool
    """Has WRF been run for the current cycle?"""

    runtime_statistics: list[RuntimeStatistics]


@dataclass
class ExperimentStatus(DataClassTOMLMixin):
    """Status of the entire experiment"""

    current_cycle: int
    """The experiment's current cycle"""

    filter_run: bool
    """True is the filter has been run for the current cycle"""

    analysis_run: bool
    """True if the analysis has been run for the current cycle"""

    members: list[MemberStatus]
    """Status for each ensemble member individually"""


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
        self.data = experiment_path / "data"
        self.data_icbc = self.data / "initial_boundary"
        self.data_forecasts = self.data / "forecasts"
        self.data_dart = self.data / "dart"
        self.data_analysis = self.data / "analysis"
        self.data_diag = self.data / "diagnostics"

        self.obs = experiment_path / "obs"

        # Work directories
        self.work = experiment_path / "work"
        self.work_wrf = self.work / "WRF"
        self.work_wps = self.work / "WPS"
        self.work_ensemble = self.work / "ensemble"
        self.member_paths = [
            self.member_path(i) for i in range(cfg.assimilation.n_members)
        ]

        # Preprocessing
        self.work_preprocessing = self.work / "preprocessing"
        self.work_preprocessing_wrf = self.work_preprocessing / "WRF"
        self.work_preprocessing_wps = self.work_preprocessing / "WPS"

        # Logs
        self.logs = experiment_path / "logs"
        self.logs_slurm = self.logs / "slurm"

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
    current_cycle_i: int
    filter_run: bool
    analysis_run: bool
    paths: ExperimentPaths
    members: list[MemberStatus] = []

    def __init__(self, experiment_path: Path):
        self.cfg = config.read_config(experiment_path / "config.toml")
        self.cycles = cycling.get_cycle_information(self.cfg)

        # Read experiment status from status.toml
        status_file_path = experiment_path / "status.toml"
        if status_file_path.exists():
            status = ExperimentStatus.from_toml(status_file_path.read_text())
            self.current_cycle_i = status.current_cycle
            self.filter_run = status.filter_run
            self.analysis_run = status.analysis_run
            self.members = [
                MemberStatus(
                    i=member_status.i,
                    advanced=member_status.advanced,
                    runtime_statistics=member_status.runtime_statistics,
                )
                for member_status in status.members
            ]

            # Make sure list is of the correct length and sorted correctly
            if len(self.members) != self.cfg.assimilation.n_members:
                raise ValueError(
                    f"Number of members in status file ({len(self.members)}) does not match configuration ({self.cfg.assimilation.n_members})"
                )
            self.members.sort(key=lambda m: m.i)
        else:
            # If the file does not exist, maybe this is a fresh experiment. Assume we are at cycle 0
            self.current_cycle_i = 0
            self.filter_run = False
            self.analysis_run = False
            self.members = [
                MemberStatus(i=i, advanced=False, runtime_statistics=[])
                for i in range(self.cfg.assimilation.n_members)
            ]

        self.paths = ExperimentPaths(experiment_path, self.cfg)

    def write_status(self):
        """
        Write the current status of the experiment to status.toml
        """

        status = ExperimentStatus(
            self.current_cycle_i, self.filter_run, self.analysis_run, self.members
        )
        (self.paths.experiment_path / "status.toml").write_text(status.to_toml())

    def set_next_cycle(self):
        """
        Update status to the next cycle
        """

        self.current_cycle_i += 1
        if self.current_cycle_i >= len(self.cycles):
            raise ValueError("No more cycles to run")
        self.filter_run = False
        self.analysis_run = False
        for member in self.members:
            member.advanced = False

    def advance_member(self, member_idx: int, cores: int) -> bool:
        """
        Run WRF to advance a member to the next cycle.
        Initial and boundary condition files must already be present in the member directory.
        Will generate the appropriate namelist. Will move forecasts to the output directory.

        Args:
            member: Index of the member to advance
            cores: Number of cores to use

        Returns:
            True if the member was advanced successfully
        """

        member = self.members[member_idx]
        member_path = self.paths.member_path(member_idx)
        cycle = self.cycles[self.current_cycle_i]

        # Refuse to run model if already advanced
        if member.advanced:
            logger.error(f"Member {member_idx} already advanced")
            return False

        # Locate WRF executable, icbc, ensure they all exist
        wrf_exe_path = (member_path / "wrf.exe").resolve()
        if not wrf_exe_path.exists():
            logger.error(
                f"Member {member_idx}: WRF executable not found at {wrf_exe_path}"
            )
            return False

        ic_path = (member_path / "wrfinput_d01").resolve()
        bc_path = (member_path / "wrfbdy_d01").resolve()
        if not ic_path.exists() or not bc_path.exists():
            logger.error(
                f"Member {member_idx}: Initial/boundary conditions not found at {ic_path} or {bc_path}"
            )
            return False

        # Generate namelist
        wrf_namelist_path = member_path / "namelist.input"
        wrf.generate_wrf_namelist(self.cfg, cycle, True, wrf_namelist_path, member_idx)

        # Clean old log files
        for f in member_path.glob("rsl.*"):
            f.unlink()

        # Run WRF
        logger.info(f"Running WRF for member {member_idx}...")
        cmd = [self.cfg.slurm.mpirun_command, "-n", str(cores), str(wrf_exe_path)]

        start_time = dt.datetime.now()
        res = external.runc(cmd, cwd=member_path)
        end_time = dt.datetime.now()

        # Check output logs
        for f in member_path.glob("rsl.*"):
            logger.add_log_file(f)
        rsl_file = member_path / "rsl.out.0000"
        if not rsl_file.exists():
            logger.error(f"Member {member_idx}: RSL file not found at {rsl_file}")
            return False

        rsl_content = rsl_file.read_text()

        if "SUCCESS COMPLETE WRF" not in rsl_content:
            logger.error(
                f"Member {member_idx}: wrf.exe failed with exit code {res.returncode}"
            )
            return False

        # Copy wrfout files to the forecasts directory
        forecasts_dir = self.paths.forecast_path(self.current_cycle_i, member_idx)
        forecasts_dir.mkdir(parents=True, exist_ok=True)
        for f in member_path.glob("wrfout*"):
            logger.info(f"Member {member_idx}: Copying {f} to {forecasts_dir}")
            f.rename(forecasts_dir / f.name)

        # Update member status
        member.advanced = True
        member.runtime_statistics.append(
            RuntimeStatistics(
                cycle=self.current_cycle_i,
                start=start_time,
                end=end_time,
                duration_s=int((end_time - start_time).total_seconds()),
            )
        )

        return True

    @property
    def current_cycle(self) -> cycling.CycleInformation:
        """
        Get the current cycle
        """

        return self.cycles[self.current_cycle_i]

    @property
    def all_members_advanced(self) -> bool:
        """
        Check if all ensemble members have been advanced
        """

        return all(m.advanced for m in self.members)
