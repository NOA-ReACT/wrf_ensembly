from pathlib import Path
from typing import Optional

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

        self.obs = experiment_path / "obs"
        self.obs_temp = self.obs / "temp"  # Temporary files during processing
        self.obs_db = experiment_path / "observations.duckdb"

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

        # Scratch
        self.scratch = cfg.directories.scratch_root
        if not self.scratch.is_absolute():
            self.scratch = experiment_path / self.scratch
        self.scratch_forecasts = self.scratch / "forecasts"
        self.scratch_analysis = self.scratch / "analysis"
        self.scratch_dart = self.scratch / "dart"

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

        self.scratch.mkdir()
        self.scratch_forecasts.mkdir()
        self.scratch_analysis.mkdir()
        self.scratch_dart.mkdir()

        self.logs.mkdir(exist_ok=True)
        self.logs_slurm.mkdir()

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

    def scratch_forecasts_path(
        self, cycle: Optional[int] = None, member: Optional[int] = None
    ) -> Path:
        if cycle is None:
            return self.scratch_forecasts
        if member is None:
            return self.scratch_forecasts / f"cycle_{cycle:03d}"
        return self.scratch_forecasts / f"cycle_{cycle:03d}" / f"member_{member:02d}"

    def scratch_analysis_path(self, cycle: Optional[int] = None) -> Path:
        if cycle is None:
            return self.scratch_analysis
        return self.scratch_analysis / f"cycle_{cycle:03d}"

    def scratch_dart_path(self, cycle: Optional[int] = None) -> Path:
        if cycle is None:
            return self.scratch_dart
        return self.scratch_dart / f"cycle_{cycle:03d}"
