"""
What a model has to provide to be cycled by `Experiment`.

The experiment owns the workflow: which cycle it is on, which members have advanced,
running the filter, planning segments. Everything that depends on the model (its
files, how it is run, how the analysis gets back into it) is behind `Model`, with one
implementation per model in this package.
"""

import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from wrf_ensembly import cycling

if TYPE_CHECKING:
    from wrf_ensembly.experiment import Experiment
    from wrf_ensembly.processors import DataProcessor


@dataclass
class ModelRun:
    """A successful model run of a member"""

    start: dt.datetime | None
    """When the run started and ended. None if nothing had to run."""
    end: dt.datetime | None
    simulated_s: int
    """How much model time it covered, in seconds"""


class Model(ABC):
    name: ClassVar[str]
    """Value of `metadata.model` in the configuration"""

    def __init__(self, exp: "Experiment"):
        self.exp = exp
        self.cfg = exp.cfg
        self.paths = exp.paths
        self.cycles = exp.cycles

    # Setting up

    @property
    @abstractmethod
    def environment(self) -> dict[str, str]:
        """Environment variables for running the model"""

    @abstractmethod
    def install(self, force: bool) -> None:
        """
        Puts the model into the experiment directory (`experiment copy-model`). With
        `force`, replaces what is there.
        """

    @abstractmethod
    def setup_member(self, member_i: int) -> None:
        """Prepares a member's directory for the first cycle (`ensemble setup`)"""

    @abstractmethod
    def preprocess_job(self, command: Callable[..., str]) -> list[str]:
        """
        The lines of the preprocessing job. `command("real", cycle=1)` gives the line
        that runs `wrf-ensembly <experiment> preprocess real --cycle 1`.
        """

    # Running

    @abstractmethod
    def run_member(
        self,
        member_i: int,
        segment: range,
        cycle: cycling.CycleInformation,
        cores: int,
    ) -> ModelRun | None:
        """
        Runs a member over the cycles of `segment`, which together are the period
        `cycle`, and leaves the output of each cycle in its scratch directory.

        Returns:
            None if the run failed
        """

    # Output files

    @property
    @abstractmethod
    def forecast_glob(self) -> str:
        """Pattern that matches the forecast output files in a scratch directory"""

    @abstractmethod
    def time_stamp(self, time: dt.datetime) -> str:
        """How a time appears in the name of a forecast output file"""

    @abstractmethod
    def forecast_file_name(self, time: dt.datetime) -> str:
        """Name of the forecast output file for a time"""

    def cycle_end_file_name(self, time: dt.datetime) -> str:
        """
        Name of the file a member must have written at the end of a cycle, for the
        filter and the next cycle
        """

        return self.forecast_file_name(time)

    @abstractmethod
    def first_processor(self) -> "DataProcessor":
        """
        The first step of the postprocessing pipeline: brings an output file to the
        common layout (dimensions `t, z, y, x`, CF names)
        """

    # Assimilation and cycling, all for the experiment's current cycle

    @abstractmethod
    def prepare_filter(self) -> list[Path]:
        """
        Puts what DART's model interface reads into the DART work directory.

        Returns:
            The file the filter reads the prior of each member from
        """

    @abstractmethod
    def make_analysis(self, member_i: int) -> None:
        """
        Combines the filter's output for a member (`dart_member_NN.nc` in the cycle's
        scratch DART directory) with its forecast, in the scratch analysis directory.
        """

    @abstractmethod
    def cycle_member(self, member_i: int, use_forecast: bool) -> None:
        """
        Prepares a member's directory for the next cycle, starting from the analysis
        or, with `use_forecast` (no filter in this cycle), from the forecast.
        """

    @abstractmethod
    def update_bc(self, member_i: int) -> None:
        """
        Makes a member's boundary conditions consistent with its initial state after
        that changed (analysis, perturbations)
        """

    @abstractmethod
    def clean_old_state(self) -> None:
        """
        Deletes the files in scratch that members started earlier cycles from and that
        no cycle can start from anymore. Called after moving to the next cycle.
        """

    # Optional features. `Config.check` rejects the settings that need them for a
    # model that does not have them (ICON: only the first cycle can be perturbed).

    def generate_perturbations(self, cycle_i: int) -> None:
        """Writes the `[perturbations]` fields of a cycle"""

        raise self._unsupported("[perturbations]")

    def generate_perturbations_for_cycles(
        self, cycle_indices: Iterable[int], jobs: int
    ) -> Iterator[int]:
        """Like `generate_perturbations` for several cycles in parallel, yielding each
        cycle as it is done"""

        raise self._unsupported("[perturbations]")

    def apply_perturbations(self, member_i: int) -> None:
        """Adds the perturbations of the current cycle to a member's initial state"""

        raise self._unsupported("[perturbations]")

    def member_initial_state(self, member_i: int, cycle_i: int) -> Path:
        """The file in a member's directory that it starts a cycle from"""

        raise self._unsupported("[segments]")

    def prepare_member_boundaries(self, member_i: int, cycle_i: int) -> None:
        """Puts the boundary conditions for a cycle, or for the segment that starts
        at it, in a member's directory"""

        raise self._unsupported("[segments]")

    def remove_cycle_output(self, cycle_i: int, member_i: int) -> None:
        """Deletes what a member wrote for a cycle, so that it can be run again"""

        raise self._unsupported("[segments]")

    def _unsupported(self, what: str) -> NotImplementedError:
        return NotImplementedError(f"{what} are not supported with {self.name}")
