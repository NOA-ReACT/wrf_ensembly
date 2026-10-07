from pathlib import Path

from wrf_ensembly.config import Config


class ExperimentPaths:
    """
    Paths to the different directories of an experiment
    """

    def __init__(self, experiment_path: Path, cfg: Config):
        self.experiment_path = experiment_path.resolve()
        self.work_path = experiment_path / "work"
        self.ensemble_path = self.work_path / "ensemble"
        self.jobfiles = experiment_path / "jobfiles"

        # Data directories
        self.data = experiment_path / "data"
        self.data_icbc = self.data / "initial_boundary"
        self.data_forecasts = self.data / "forecasts"
        self.data_analysis = self.data / "analysis"
        self.data_diag = self.data / "diagnostics"
        self.data_inflation = self.data / "inflation"
        # Created lazily by the validation analyses, not by `create_directories`
        self.data_validation = self.data / "validation"

        self.obs = experiment_path / "obs"
        self.obs_temp = self.obs / "temp"  # Temporary files during processing
        self.obs_db = experiment_path / "observations.duckdb"

        # Experiment status, stored as a tree of small files (see state_store.py)
        self.status = experiment_path / "status"
        self.status_experiment_file = self.status / "experiment.json"
        self.status_cycles = self.status / "cycles"

        self.plots = experiment_path / "plots"

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

        # DART working directory
        self.dart_work_dir = cfg.directories.dart_root / "models" / "wrf" / "work"

        # Scratch
        self.scratch = cfg.directories.scratch_root
        if not self.scratch.is_absolute():
            self.scratch = experiment_path / self.scratch
        self.scratch_forecasts = self.scratch / "forecasts"
        self.scratch_analysis = self.scratch / "analysis"
        self.scratch_dart = self.scratch / "dart"
        self.scratch_restart = self.scratch / "restart"

    def create_directories(self):
        """Creates all required directories"""
        self.obs.mkdir()
        self.work.mkdir()
        self.work_preprocessing.mkdir()
        self.jobfiles.mkdir()
        self.plots.mkdir()

        self.data.mkdir()
        self.data_analysis.mkdir()
        self.data_forecasts.mkdir()
        self.data_icbc.mkdir()
        self.data_diag.mkdir()
        self.data_inflation.mkdir()

        self.scratch.mkdir()
        self.scratch_forecasts.mkdir()
        self.scratch_analysis.mkdir()
        self.scratch_dart.mkdir()
        self.scratch_restart.mkdir()

        self.logs.mkdir(exist_ok=True)
        self.logs_slurm.mkdir()

        self.status.mkdir(exist_ok=True)
        self.status_cycles.mkdir(exist_ok=True)

    def cycle_status_path(self, cycle: int) -> Path:
        """Directory holding all status files for a given cycle"""
        return self.status_cycles / f"cycle_{cycle:03d}"

    def cycle_members_path(self, cycle: int) -> Path:
        """Directory holding the per-member status files for a given cycle"""
        return self.cycle_status_path(cycle) / "members"

    def member_status_path(self, cycle: int, member: int) -> Path:
        """Status file for one member of one cycle. Written only by that member's job."""
        return self.cycle_members_path(cycle) / f"member_{member:02d}.json"

    def cycle_marker_path(self, cycle: int, name: str) -> Path:
        """
        Marker file for a completed step of a cycle (e.g. `filter_complete`). The file
        existing means the step is done.
        """
        return self.cycle_status_path(cycle) / name

    def cycle_op_path(self, cycle: int, name: str) -> Path:
        """Marker file for a completed optional operation (e.g. `apply_perturbations`)"""
        return self.cycle_status_path(cycle) / "ops" / name

    def member_path(self, i: int) -> Path:
        """
        Get the work directory for given ensemble member
        """
        return self.ensemble_path / f"member_{i:02d}"

    def forecast_path(
        self, cycle: int | None = None, member: int | None = None
    ) -> Path:
        if cycle is None:
            return self.data_forecasts
        if member is None:
            return self.data_forecasts / f"cycle_{cycle:03d}"
        return self.data_forecasts / f"cycle_{cycle:03d}" / f"member_{member:02d}"

    def analysis_path(self, cycle: int | None = None) -> Path:
        if cycle is None:
            return self.data_analysis
        return self.data_analysis / f"cycle_{cycle:03d}"

    def scratch_forecasts_path(
        self, cycle: int | None = None, member: int | None = None
    ) -> Path:
        if cycle is None:
            return self.scratch_forecasts
        if member is None:
            return self.scratch_forecasts / f"cycle_{cycle:03d}"
        return self.scratch_forecasts / f"cycle_{cycle:03d}" / f"member_{member:02d}"

    def scratch_analysis_path(self, cycle: int | None = None) -> Path:
        if cycle is None:
            return self.scratch_analysis
        return self.scratch_analysis / f"cycle_{cycle:03d}"

    def scratch_dart_path(self, cycle: int | None = None) -> Path:
        if cycle is None:
            return self.scratch_dart
        return self.scratch_dart / f"cycle_{cycle:03d}"

    def scratch_restart_path(self, cycle: int, member: int) -> Path:
        """
        Where a member writes its WRF restart files during a cycle, in the `restart`
        cycling mode. The one at the cycle end is the next cycle's initial state.
        """
        return self.scratch_restart / f"cycle_{cycle:03d}" / f"member_{member:02d}"

    def icbc_file_path(
        self, prefix: str, member: int | None, cycle: int | None
    ) -> Path:
        """
        Where real.exe's output `prefix` (e.g. `wrfbdy_d01`) is stored. `member = None` is
        the file shared by all members, `cycle = None` the file for the whole experiment,
        as made in the `restart` cycling mode.
        """
        cycle_suffix = "" if cycle is None else f"_cycle_{cycle}"
        if member is None:
            return self.data_icbc / f"{prefix}{cycle_suffix}"
        return (
            self.data_icbc
            / f"member_{member:02d}"
            / f"{prefix}_member_{member:02d}{cycle_suffix}"
        )

    def _icbc_path(self, prefix: str, member: int, cycle: int | None) -> Path:
        """
        Like `icbc_file_path`, but returns the member-specific path only if it exists,
        otherwise the shared path.
        """
        member_path = self.icbc_file_path(prefix, member, cycle)
        if member_path.exists():
            return member_path
        return self.icbc_file_path(prefix, None, cycle)

    def ic_path(self, member: int, cycle: int) -> Path:
        """
        Get the initial conditions (wrfinput) file for a given member/cycle.
        Returns the member-specific path if it exists, otherwise the shared path.
        """
        return self._icbc_path("wrfinput_d01", member, cycle)

    def bc_path(self, member: int, cycle: int | None) -> Path:
        """
        Get the boundary conditions (wrfbdy) file for a given member/cycle, or for the
        whole experiment if `cycle` is None.
        Returns the member-specific path if it exists, otherwise the shared path.
        """
        return self._icbc_path("wrfbdy_d01", member, cycle)

    def lowinp_path(self, member: int, cycle: int | None) -> Path:
        """
        Get the lower boundary conditions (wrflowinp, written by real.exe when
        `sst_update = 1`) file for a given member/cycle, or for the whole experiment if
        `cycle` is None.
        Returns the member-specific path if it exists, otherwise the shared path.
        """
        return self._icbc_path("wrflowinp_d01", member, cycle)
