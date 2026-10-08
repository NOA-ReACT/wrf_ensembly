import datetime as dt
import json
import os
import shutil
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import netCDF4
import numpy as np
import xarray as xr

from wrf_ensembly import (
    config,
    cycling,
    external,
    obs_sequence,
    perturbations,
    rebalance,
    restart,
    segments,
    update_bc,
    utils,
    wrf,
)
from wrf_ensembly.console import logger
from wrf_ensembly.fortran_namelists import write_namelist

from .dataclasses import MemberStatus
from .inflation import InflationConfig
from .observations import ExperimentObservations
from .paths import ExperimentPaths
from .state_machine import (
    CycleState,
    ExperimentStateError,
    ExperimentStateMachine,
    StateTransition,
)
from .state_store import ExperimentState


MEMBER_VISIBILITY_TIMEOUT_S = 60.0
"""
How long to keep re-reading the status directory when members appear to be missing.

On NFS a directory listing can be served from the attribute cache for up to `acdirmax`
(30s by default), so a member that has just finished may not be visible to the process
that runs the filter yet. Harmless on filesystems with coherent metadata.
"""


def _member_visibility_timeout() -> float:
    """
    Only wait for missing members inside a batch job, where the filter starts right after
    the member jobs and can genuinely be looking at a stale directory listing. Run by
    hand, it is much more useful to say "3/20 advanced" immediately than to hang for a
    minute first.
    """

    return MEMBER_VISIBILITY_TIMEOUT_S if "SLURM_JOB_ID" in os.environ else 0.0


# Groups that filter cannot run without. `[dart_namelist]` is the complete input.nml, so
# these are only a sanity check; DART reports any other missing group itself.
REQUIRED_DART_NAMELIST_GROUPS = ("filter_nml", "model_nml", "obs_kind_nml")


class Experiment:
    """
    An ensemble assimilation experiment
    """

    cfg: config.Config
    cycles: list[cycling.CycleInformation]
    paths: ExperimentPaths
    members: list[MemberStatus] = []

    state: ExperimentState
    obs: ExperimentObservations
    inflation: InflationConfig
    state_machine: ExperimentStateMachine

    def __init__(self, experiment_path: Path):
        self.cfg = config.read_config(experiment_path / "config.toml")
        self.cycles = cycling.get_cycle_information(self.cfg)

        self.paths = ExperimentPaths(experiment_path, self.cfg)
        self.state = ExperimentState(self.paths, self.cfg.assimilation.n_members)

        # Read the experiment status. Nothing is written here.
        self.load_status()

        self.obs = ExperimentObservations(self.cfg, self.cycles, self.paths)
        self.inflation = InflationConfig.from_config(
            self.cfg, self.paths.data_inflation, self.paths.dart_work_dir
        )

    def load_status(self):
        """Load the status of the experiment from the status files"""

        self.state_machine = ExperimentStateMachine(
            state=self.state,
            n_cycles=len(self.cycles),
            n_members=self.cfg.assimilation.n_members,
            current_cycle_idx=self.state.get_current_cycle(),
        )

        advanced = self.state.get_advanced_members(self.current_cycle_i)
        self.members = [
            MemberStatus(i=i, advanced=i in advanced)
            for i in range(self.cfg.assimilation.n_members)
        ]

    @property
    def current_cycle_i(self) -> int:
        """Which cycle the experiment is currently on"""

        return self.state_machine.current_cycle_idx

    @current_cycle_i.setter
    def current_cycle_i(self, cycle: int):
        """
        Move the experiment to a cycle, in memory and on disk. The state machine reads
        this too, so the two can never drift apart.
        """

        self.state_machine.current_cycle_idx = cycle
        self.state.set_current_cycle(cycle)

    @property
    def filter_run(self) -> bool:
        """Check if filter has been run for the current cycle (derived from state machine)."""
        state = self.state_machine.current_cycle.current_state
        return state in (
            CycleState.FILTER_COMPLETE,
            CycleState.ANALYSIS_COMPLETE,
            CycleState.CYCLE_COMPLETE,
        )

    @property
    def analysis_run(self) -> bool:
        """Check if analysis has been run for the current cycle (derived from state machine)."""
        state = self.state_machine.current_cycle.current_state
        return state in (CycleState.ANALYSIS_COMPLETE, CycleState.CYCLE_COMPLETE)

    def set_next_cycle(self):
        """
        Update status to the next cycle
        """

        next_cycle_i = self.current_cycle_i + 1
        if next_cycle_i >= len(self.cycles):
            raise ValueError("No more cycles to run")

        # No member status to reset: the next cycle has its own directory, which is
        # empty until its members start advancing.
        self.current_cycle_i = next_cycle_i
        for member in self.members:
            member.advanced = False

    def setup_dart(self):
        """Prepare DART working directory by writing namelist and linking files"""

        dart_dir = self.paths.dart_work_dir

        # input.nml is written from [dart_namelist] alone, so it must hold the whole namelist.
        # Catch the obvious cases here instead of letting filter fail on a partial file.
        missing = [
            group
            for group in REQUIRED_DART_NAMELIST_GROUPS
            if group not in self.cfg.dart_namelist
        ]
        if missing:
            raise ValueError(
                f"[dart_namelist] is missing the {', '.join(missing)} group(s). "
                "It must contain the complete DART input.nml, not only overrides: "
                "setup-dart replaces input.nml with its contents."
            )

        # Prepare inflation: restore restart files and apply namelist overrides
        filter_namelist_path = dart_dir / "input.nml"
        logger.info(f"Writing DART filter namelist to {filter_namelist_path}")
        dart_namelist = self.inflation.prepare_cycle(
            self.cfg.dart_namelist, self.current_cycle_i
        )
        dart_namelist["filter_nml"]["ens_size"] = self.cfg.assimilation.n_members
        write_namelist(dart_namelist, filter_namelist_path)

        # Copy files if defined in config
        for file_cfg in self.cfg.extra_dart_files:
            source_path = Path(file_cfg.source).resolve()
            if file_cfg.destination_name is None:
                target_path = dart_dir / source_path.name
            else:
                target_path = dart_dir / file_cfg.destination_name
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Extra DART file not found at {source_path}"
                )
            utils.copy(source_path, target_path)
            logger.info(f"Copied extra DART file from {source_path} to {target_path}")

    def _perturbation_worker_args(self) -> dict[str, Any]:
        """
        The arguments `perturbations.generate_perturbations_for_cycle` needs, pulled out
        of the experiment. Kept in one place because the same set has to be handed to
        worker processes, which cannot be given `self`.
        """

        return {
            "perturbations_cfg": self.cfg.perturbations,
            "reapplied_parameters": perturbations.reapplied_parameters(self.cfg),
            "n_members": self.cfg.assimilation.n_members,
            "experiment_name": self.cfg.metadata.name,
            "ic_path": self.paths.ic_path(0, 0),
            "diag_dir": self.paths.data_diag,
        }

    def generate_perturbations(self, cycle_i: int):
        """
        Generates perturbations for a given cycle and stores them in `data/diag/perturbations`.
        """

        perturbations.generate_perturbations_for_cycle(
            cycle_i=cycle_i, **self._perturbation_worker_args()
        )

    def generate_perturbations_for_cycles(
        self, cycle_indices: Iterable[int], jobs: int
    ) -> Iterator[int]:
        """
        Generates perturbations for several cycles in parallel, yielding each cycle index
        as it finishes so the caller can record progress one cycle at a time.

        Args:
            cycle_indices: Which cycles to generate perturbations for
            jobs: How many cycles to process in parallel

        Yields:
            Each cycle index, once its perturbations have been written
        """

        cycle_indices = list(cycle_indices)
        worker = partial(
            perturbations.generate_perturbations_for_cycle,
            **self._perturbation_worker_args(),
        )

        with ProcessPoolExecutor(max_workers=jobs, max_tasks_per_child=1) as executor:
            results = executor.map(worker, cycle_indices)
            for cycle_i in cycle_indices:
                next(results)  # Raises here if the worker failed
                yield cycle_i

    def apply_perturbations(self, member_i: int):
        """
        Apply perturbations to the initial conditions of a member.
        Must be generated with `generate_perturbations` first.
        You should call this function once for every member, after the initial conditions
        are copied in the member directory (either during `ensemble setup` or `ensemble cycle`).

        This function will mostly not check the configuration, rather it will assume that
        the contents of the perturbation files are correct. This allows you to modify them
        outside of this program if needed. The exception is variables in the config that
        are not perturbed at this cycle (see `perturbations.applied_at_cycle`), which are
        skipped even if the file has them. Files generated before `kind = "parameter"`
        existed repeat the first cycle's field in every cycle, and in restart mode
        applying it again would compound it.
        """

        if member_i >= self.cfg.assimilation.n_members:
            raise ValueError(
                f"Member index {member_i} is out of bounds for {self.cfg.assimilation.n_members} members"
            )

        pert_file = (
            self.paths.data_diag
            / "perturbations"
            / f"perts_cycle_{self.current_cycle_i}.nc"
        )
        if not pert_file.exists():
            raise FileNotFoundError(
                f"Perturbation file for cycle {self.current_cycle_i} not found at {pert_file}"
            )
        initial_state = self.member_initial_state(member_i, self.current_cycle_i)
        if not initial_state.exists():
            raise FileNotFoundError(
                f"Initial conditions file for member {member_i} not found at {initial_state}"
            )
        # In restart mode, after the first cycle, it's a restart file that needs rebalancing
        is_restart = initial_state.name.startswith("wrfrst")

        perts = xr.open_dataset(pert_file).sel(member=member_i)
        applied = perturbations.applied_at_cycle(self.cfg, self.current_cycle_i)
        pert_config = {}
        for name, var in perts.data_vars.items():
            if name in self.cfg.perturbations.variables and name not in applied:
                logger.info(
                    f"Member {member_i}: Not perturbing {name}, it is not perturbed at "
                    f"cycle {self.current_cycle_i}"
                )
                continue
            pert_config[name] = json.loads(var.attrs["cfg"])
        if not pert_config:
            logger.info(f"Member {member_i}: Nothing to perturb")
            return

        with netCDF4.Dataset(initial_state, "r+") as member_icbc:  # type: ignore
            before = rebalance.balanced_fields(member_icbc) if is_restart else None
            for name, cfg in pert_config.items():
                if "operation" not in cfg:
                    raise ValueError(
                        f"Perturbation config for {name} does not contain 'operation', parsed from netCDF attribute `cfg`"
                    )
                operation = cfg["operation"]

                logger.debug(f"Applying perturbation to {name} for member {member_i}")
                logger.debug(f"Perturbation config: {pert_config}")
                if not restart.field_names(member_icbc, name):
                    raise ValueError(f"Variable {name} not found in member IC/BC file.")

                field = perts[name].to_numpy()

                # Apply midcycle taper if configured (for non-first cycles using
                # reused perturbation fields)
                midcycle_taper_width = cfg.get("midcycle_taper_width", 0)
                if midcycle_taper_width > 0 and self.current_cycle_i > 0:
                    logger.debug(
                        f"Applying mid-cycle taper (width={midcycle_taper_width}) to {name}"
                    )
                    taper = perturbations.edge_taper(
                        field.shape[-2], field.shape[-1], midcycle_taper_width
                    )
                    if operation == "add":
                        field = field * taper
                    elif operation == "multiply":
                        field = 1 + (field - 1) * taper
                    else:
                        raise ValueError(
                            "Cannot use 'assign' operation with midcycle taper"
                        )

                # Perturb the current state, `set_field` handles restart time levels
                current = member_icbc[restart.field_names(member_icbc, name)[-1]][:]
                match operation:
                    case "add":
                        perturbed = current + field
                    case "multiply":
                        perturbed = current * field
                    case "assign":
                        perturbed = np.broadcast_to(field, current.shape)
                    case _:
                        raise ValueError(f"Unknown perturbation operation: {operation}")
                restart.set_field(member_icbc, name, perturbed)

            if before is not None:
                rebalance.rebalance(member_icbc, before)

        logger.info(f"Applied perturbations to member {member_i}")

    def update_bc(self, member_i: int) -> None:
        """
        Update the boundary conditions of a member with the changes made to its initial
        state (analysis, perturbations), see `member_reference_state`.
        """

        member_path = self.paths.member_path(member_i)
        reference = self.member_reference_state(member_i, self.current_cycle_i)
        if not reference.exists():
            raise FileNotFoundError(
                f"Member {member_i}: The unmodified initial state is needed to update the "
                f"boundary conditions, but {reference} does not exist"
            )
        changed = update_bc.update_wrf_bc(
            self.member_initial_state(member_i, self.current_cycle_i),
            reference,
            member_path / "wrfbdy_d01",
        )
        if changed:
            logger.info(f"Member {member_i}: Updated boundary conditions")
        else:
            logger.info(
                f"Member {member_i}: Initial state unchanged at the boundaries, "
                "boundary conditions left as they are"
            )

    def advance_member(self, member_idx: int, cores: int) -> bool:
        """
        Run WRF to advance a member to the next cycle, or with segments to the end of the
        current segment.
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
        segment = self.runnable_segment()
        if len(segment) == 1:
            # Keeps the cycle's own output interval and forecast extension
            cycle = self.cycles[segment[0]]
        else:
            cycle = cycling.get_segment_period(self.cycles, segment[0], segment[-1])
            logger.info(
                f"Member {member_idx}: Running cycles {segment[0]}-{segment[-1]} as one run"
            )

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

        ic_path = self.member_initial_state(member_idx, self.current_cycle_i).resolve()
        bc_path = (member_path / "wrfbdy_d01").resolve()
        if not ic_path.exists() or not bc_path.exists():
            logger.error(
                f"Member {member_idx}: Initial/boundary conditions not found at {ic_path} or {bc_path}"
            )
            return False

        # Generate namelist
        wrf_namelist_path = member_path / "namelist.input"
        wrf.generate_wrf_namelist(
            self.cfg,
            cycle,
            True,
            wrf_namelist_path,
            member_idx,
            self.paths,
        )

        # Clean old log files
        for f in member_path.glob("rsl.*"):
            f.unlink()

        # Run WRF
        logger.info(f"Running WRF for member {member_idx}...")
        cmd = [
            *self.cfg.slurm.mpirun_command.split(" "),
            "-n",
            str(cores),
            str(wrf_exe_path),
        ]

        start_time = dt.datetime.now()
        res = external.runc(cmd, cwd=member_path)
        end_time = dt.datetime.now()

        # Check output logs
        rsl_file = member_path / "rsl.out.0000"
        if not rsl_file.exists():
            logger.error(f"Member {member_idx}: RSL file not found at {rsl_file}")
            return False

        logger.add_log_file(rsl_file)
        rsl_content = rsl_file.read_text()

        if "SUCCESS COMPLETE WRF" not in rsl_content:
            logger.error(
                f"Member {member_idx}: wrf.exe failed with exit code {res.returncode}"
            )
            return False

        # Store logs in a zip file
        if logger.log_dir is not None:
            rsl_files = sorted(member_path.glob("rsl.*"))
            utils.zip_files(rsl_files, logger.log_dir / "rsl.zip")

        self.file_segment_outputs(member_idx, segment)

        # Every cycle of the segment gets its own status file. The first cycle's is what
        # decides whether the member is advanced (`can_advance_member`, the job array),
        # so it is written last: a job that dies in between counts as not advanced.
        # The runtime goes on the last cycle's file only, since it is one run.
        for cycle_i in reversed(segment):
            if cycle_i == segment[-1]:
                self.state.set_member_advanced(
                    cycle_i,
                    member_idx,
                    start=start_time,
                    end=end_time,
                    duration_s=int((end_time - start_time).total_seconds()),
                    simulated_s=int((cycle.end - cycle.start).total_seconds()),
                )
            else:
                self.state.set_member_advanced(cycle_i, member_idx)
        self.members[member_idx].advanced = True

        return True

    def file_segment_outputs(self, member_i: int, segment: range):
        """
        After a member ran a segment: WRF wrote all output and restart files into the
        first cycle's scratch directories. Move each wrfout into the cycle it belongs to
        (`start < time <= end`), delete the one at the segment start (as for a single
        cycle), and move the restart file at the segment end into the last cycle's
        restart directory, where `cycle` looks for it.

        Safe to run again on files that were already moved.
        """

        first, last = segment[0], segment[-1]
        first_dir = self.paths.scratch_forecasts_path(first, member_i)
        first_output = (
            first_dir / f"wrfout_d01_{self.cycles[first].start:%Y-%m-%d_%H:%M:%S}"
        )
        if first_output.exists():
            logger.info(f"Removing first output file {first_output}")
            first_output.unlink()

        if first == last:
            return

        for f in sorted(first_dir.glob("wrfout_d01_*")):
            time = wrf.wrfout_time(f.name)
            cycle_i = next(
                (
                    i
                    for i in segment
                    if self.cycles[i].start < time <= self.cycles[i].end
                ),
                None,
            )
            if cycle_i is None or cycle_i == first:
                continue
            target = self.paths.scratch_forecasts_path(cycle_i, member_i) / f.name
            target.parent.mkdir(parents=True, exist_ok=True)
            f.rename(target)
        logger.info(f"Member {member_i}: Sorted the output into cycles {first}-{last}")

        if self.cfg.assimilation.cycling_mode == "restart":
            end = self.cycles[last].end
            name = f"wrfrst_d01_{end:%Y-%m-%d_%H:%M:%S}"
            source = self.paths.scratch_restart_path(first, member_i) / name
            target = self.paths.scratch_restart_path(last, member_i) / name
            if source.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
                logger.info(f"Member {member_i}: Moved {name} to {target.parent}")
            elif not target.exists():
                raise FileNotFoundError(f"WRF wrote no restart file at {source}")

    def filter(self) -> bool:
        """
        Run the Kalman Filter for the current cycle.

        Checks and updates the experiment state machine. Raises ExperimentStateError if
        the experiment is not in a valid state to run the filter.

        Returns:
            True if the filter was run successfully
        """

        # Each member advanced in its own job, and on NFS the status files of the last
        # ones to finish may not be visible here yet. Let the listing settle before the
        # check below decides the ensemble is incomplete. Returns immediately unless we
        # are in a batch job and members are actually missing.
        self.check_segment_finished()
        self.wait_for_all_members()

        # Validate state
        can_run, error = self.state_machine.can_run_filter(self.current_cycle_i)
        if not can_run:
            raise ExperimentStateError(error)

        # Skip filter if no observation file and no inflation enabled
        obs_file = self.paths.obs / f"cycle_{self.current_cycle_i:03}.obs_seq"
        if not obs_file.exists():
            logger.error(f"No observation file found at {obs_file}, cannot run filter!")
            return False

        self.setup_dart()

        # Grab observations
        dart_dir = self.paths.dart_work_dir
        obs_seq = dart_dir / "obs_seq.out"
        obs_seq.unlink(missing_ok=True)
        utils.copy(obs_file, obs_seq)

        # Write lists of input and output files
        # The input list is the latest forecast for each member
        wrfout_name = "wrfout_d01_" + self.current_cycle.end.strftime(
            "%Y-%m-%d_%H:%M:%S"
        )
        priors = [
            self.paths.scratch_forecasts_path(self.current_cycle_i, member_i)
            / wrfout_name
            for member_i in range(0, self.cfg.assimilation.n_members)
        ]
        posterior = [
            self.paths.scratch_dart_path(self.current_cycle_i)
            / f"dart_{prior.parent.name}.nc"
            for prior in priors
        ]

        dart_input_txt = dart_dir / "input_list.txt"
        dart_input_txt.write_text("\n".join(str(prior.resolve()) for prior in priors))
        logger.info(f"Wrote {dart_input_txt}")
        dart_output_txt = dart_dir / "output_list.txt"
        dart_output_txt.write_text("\n".join(str(post.resolve()) for post in posterior))
        logger.info(f"Wrote {dart_output_txt}")

        self.paths.scratch_dart_path(self.current_cycle_i).mkdir(exist_ok=True)

        # Link wrfinput, required by filter to read coordinates
        wrfinput_path = dart_dir / "wrfinput_d01"
        wrfinput_path.unlink(missing_ok=True)
        # Any wrfinput of the domain works, the grid doesn't change between cycles. In
        # restart mode, only the first cycle has one.
        wrfinput_cur_cycle_path = self.paths.ic_path(0, 0)
        wrfinput_path.symlink_to(wrfinput_cur_cycle_path)
        logger.info(f"Linked {wrfinput_path} to {wrfinput_cur_cycle_path}")

        # Run filter
        if self.cfg.assimilation.filter_mpi_tasks == 1:
            logger.info("Running filter w/out MPI")
            cmd = ["./filter"]
        else:
            logger.info(
                f"Using MPI to run filter, n={self.cfg.assimilation.filter_mpi_tasks}"
            )
            cmd = [
                *self.cfg.slurm.mpirun_command.split(" "),
                "-n",
                str(self.cfg.assimilation.filter_mpi_tasks),
                "./filter",
            ]
        res = external.runc(cmd, dart_dir, log_filename="filter.log")
        if res.returncode != 0 or "Finished ... at" not in res.output:
            logger.error(f"filter failed with exit code {res.returncode}")
            return False

        # Keep obs_seq.final for diagnostics, convert to netcdf
        obs_seq_final = dart_dir / "obs_seq.final"
        utils.copy(
            obs_seq_final,
            self.paths.data_diag / f"cycle_{self.current_cycle_i}.obs_seq.final",
        )
        obs_seq_final_nc = self.paths.data_diag / f"cycle_{self.current_cycle_i}.nc"
        obs_sequence.obs_seq_to_nc(
            self.cfg.directories.dart_root, obs_seq_final, obs_seq_final_nc
        )

        # Stash inflation files so we can use them next cycle
        self.inflation.stash_restart_files(self.current_cycle_i)

        # Transition state
        self.state_machine.current_cycle.transition(StateTransition.FILTER_COMPLETE)

        return True

    def clean_restart_files(self) -> None:
        """
        Deletes the WRF restart files in scratch that no cycle can start from anymore, in
        restart mode, unless `assimilation.keep_restart_files` is set or the cycle is in
        `assimilation.keep_restart_files_for_cycles`. Call after moving to the next cycle.

        The current cycle starts from the previous cycle's restart files, which are kept
        so it can be rerun. Older cycles' are deleted, and so are any restart files of
        the previous cycle other than the one at its end (written when a forecast
        extension reaches past the next restart time).
        """

        if (
            self.cfg.assimilation.cycling_mode != "restart"
            or self.cfg.assimilation.keep_restart_files
        ):
            return

        previous_i = self.current_cycle_i - 1
        if previous_i < 0:
            return
        keep_cycles = set(self.cfg.assimilation.keep_restart_files_for_cycles)
        for cycle_dir in sorted(self.paths.scratch_restart.glob("cycle_*")):
            cycle_i = int(cycle_dir.name.removeprefix("cycle_"))
            if cycle_i < previous_i and cycle_i not in keep_cycles:
                logger.info(f"Removing restart files in {cycle_dir}")
                shutil.rmtree(cycle_dir)

        end = self.cycles[previous_i].end
        keep = f"wrfrst_d01_{end:%Y-%m-%d_%H:%M:%S}"
        for member_i in range(self.cfg.assimilation.n_members):
            member_dir = self.paths.scratch_restart_path(previous_i, member_i)
            for f in member_dir.glob("wrfrst_*"):
                if f.name != keep:
                    logger.info(f"Removing unused restart file {f}")
                    f.unlink()

    # Segments (see `segments.py`)

    def observation_cycles(self) -> set[int]:
        """Cycles with an observation file, i.e. where filter will run"""

        return {
            i for i in range(len(self.cycles)) if self.paths.obs_seq_path(i).exists()
        }

    def segment_rate(self) -> tuple[float, str]:
        """
        Wall clock seconds per simulated hour to size segments with, and where the number
        came from
        """

        rate = self.cfg.segments.expected_walltime_per_sim_hour
        if rate is None:
            raise ValueError("segments.expected_walltime_per_sim_hour is not set")
        return rate, "expected_walltime_per_sim_hour"

    def plan_segment(
        self, first: int, run_until: int | None = None
    ) -> segments.SegmentPlan:
        """
        Plans the segment starting at cycle `first` and stores the plan, replacing any
        plan starting there. Doesn't check whether that is safe, see `segment_started`.
        """

        stops = segments.find_stops(
            len(self.cycles),
            self.observation_cycles(),
            set(self.cfg.assimilation.keep_restart_files_for_cycles),
            run_until,
        )
        rate, rate_source = self.segment_rate()
        plan = segments.plan_segment(
            self.cycles,
            first,
            stops,
            rate_s_per_sim_hour=rate,
            rate_source=rate_source,
            max_walltime_s=utils.parse_slurm_time(self.cfg.segments.max_walltime),
            safety_factor=self.cfg.segments.safety_factor,
            checkpoint_interval_hours=self.cfg.segments.checkpoint_interval_hours,
            run_until=run_until,
        )
        self.state.set_segment_plan(plan)
        logger.info(
            f"Planned segment {plan}, ends because of: {plan.stop_reason}, "
            f"estimated {utils.seconds_to_pretty_hours(plan.estimated_walltime_s)} per "
            f"member, checkpoints every {plan.checkpoint_interval_min} min"
        )
        return plan

    def segment_of(self, cycle_i: int) -> segments.SegmentPlan | None:
        """
        The planned segment covering a cycle, or None if segments are disabled or the
        cycle has no plan
        """

        if not self.cfg.segments.enabled:
            return None
        covering = [p for p in self.state.get_segment_plans() if cycle_i in p]
        # Overlapping plans only exist if files were edited by hand; the latest start wins
        return max(covering, key=lambda p: p.first, default=None)

    def runnable_segment(self) -> range:
        """
        The cycles the members run as one from the current cycle: the planned segment,
        or only the current cycle without segments.

        Raises:
            ExperimentStateError: if segments are enabled but there is no plan starting
                at the current cycle
        """

        cycle_i = self.current_cycle_i
        if not self.cfg.segments.enabled:
            return range(cycle_i, cycle_i + 1)
        plan = self.segment_of(cycle_i)
        if plan is None or plan.first != cycle_i:
            raise ExperimentStateError(
                f"No segment is planned from cycle {cycle_i}, run `ensemble plan-segment`"
            )
        return plan.cycles

    def unfinished_segment(self) -> segments.SegmentPlan | None:
        """
        The segment of several cycles the experiment is at the start of, if any. Until
        `finish_segment` has moved the experiment to its last cycle, the first cycle's
        state says MEMBERS_ADVANCED once the members are done, but there is nothing to
        filter or cycle from at its end.
        """

        plan = self.segment_of(self.current_cycle_i)
        if plan is not None and plan.first == self.current_cycle_i < plan.last:
            return plan
        return None

    def check_segment_finished(self) -> None:
        """
        Raises ExperimentStateError if the experiment is at the start of a segment of
        several cycles, for the commands that act on the end of the current cycle
        """

        plan = self.unfinished_segment()
        if plan is not None:
            raise ExperimentStateError(
                f"The experiment is at the start of segment {plan}, run "
                "`ensemble finish-segment` once all members have advanced"
            )

    def finish_segment(self) -> None:
        """
        After all members ran the current segment: marks its cycles but the last as
        complete (the forecast-only transition, as when cycling without a filter) and
        moves the experiment to the last cycle, where filter/analysis/cycle run as
        usual. Does nothing without a segment of several cycles, and is safe to run
        again if it was interrupted.

        Raises:
            ExperimentStateError: if a member hasn't advanced every cycle, or an
                observation file appeared for a cycle inside the segment after planning
        """

        plan = self.unfinished_segment()
        if plan is None:
            logger.info("Not at the start of a segment of several cycles, nothing to do")
            return

        n_members = self.cfg.assimilation.n_members
        timeout = _member_visibility_timeout()
        for cycle_i in plan.cycles:
            n = self.state.count_advanced(cycle_i, expect=n_members, timeout=timeout)
            if n < n_members:
                raise ExperimentStateError(
                    f"Only {n}/{n_members} members have advanced cycle {cycle_i} of "
                    f"segment {plan}"
                )

        # The filter would only run at the end, and this cycle's observations would be
        # skipped without a word
        inside = sorted(i for i in self.observation_cycles() if plan.first <= i < plan.last)
        if inside:
            raise ExperimentStateError(
                f"Cycle(s) {', '.join(map(str, inside))} got observations after segment "
                f"{plan} was planned, so they were not assimilated. Reset it with "
                f"`ensemble reset-cycle --cycle {plan.first}` to run it again with the "
                "new stops."
            )

        for cycle_i in range(plan.first, plan.last):
            machine = self.state_machine.get_cycle(cycle_i)
            if machine.current_state != CycleState.CYCLE_COMPLETE:
                machine.transition(StateTransition.CYCLE_COMPLETE)
        self.current_cycle_i = plan.last
        self.members = [
            MemberStatus(i=i, advanced=True) for i in range(n_members)
        ]
        logger.info(f"Finished segment {plan}, now at cycle {plan.last}")

    def reset_segment(self, plan: segments.SegmentPlan, jobs: int = 1) -> None:
        """
        Returns a segment to where its members can run it again from the start: forgets
        the status of all its cycles, deletes the output and restart files (checkpoints)
        the members wrote, so a rerun can't continue from files of an earlier attempt,
        moves the experiment back to the first cycle and plans the segment again (which
        picks up observations added since).

        Only for the segment the experiment is in. The members' initial state at the
        first cycle must still be in their directories, as it is until `cycle` runs.
        """

        if not plan.first <= self.current_cycle_i <= plan.last:
            raise ExperimentStateError(
                f"The experiment is at cycle {self.current_cycle_i}, outside segment "
                f"{plan}, which can't be reset anymore"
            )

        for cycle_i in plan.cycles:
            self.state_machine.get_cycle(cycle_i).reset()
            for member_i in range(self.cfg.assimilation.n_members):
                for f in self.paths.scratch_forecasts_path(cycle_i, member_i).glob(
                    "wrfout_*"
                ):
                    f.unlink()
                for f in self.paths.scratch_restart_path(cycle_i, member_i).glob(
                    "wrfrst_*"
                ):
                    f.unlink()
        logger.info(f"Reset the status of {plan} and deleted its output and restart files")

        self.state.clear_segment_plan(plan.first)
        self.current_cycle_i = plan.first
        self.members = [
            MemberStatus(i=i, advanced=False)
            for i in range(self.cfg.assimilation.n_members)
        ]

        self.plan_segment(plan.first, plan.run_until)
        self.refresh_segment_boundaries(jobs)

    def refresh_segment_boundaries(self, jobs: int = 1) -> None:
        """
        After the current cycle's segment was (re)planned: extracts each member's wrfbdy
        for the new length and applies the boundary increment again, as `update-bc` did
        for the old one. Skipped before `ensemble setup`, when the members have nothing
        to start from yet.
        """

        cycle_i = self.current_cycle_i
        n_members = self.cfg.assimilation.n_members
        missing = [
            i
            for i in range(n_members)
            if not self.member_initial_state(i, cycle_i).exists()
        ]
        if cycle_i == 0 and len(missing) == n_members:
            logger.info("The members are not set up yet, `ensemble setup` will do it")
            return
        if missing:
            logger.warning(
                f"{len(missing)} member(s) have no initial state for cycle {cycle_i} "
                f"(e.g. {self.member_initial_state(missing[0], cycle_i)}), their "
                "boundaries were not updated for the segment. Once they are set up, run "
                "`ensemble plan-segment --replan`."
            )
            return

        with ProcessPoolExecutor(max_workers=jobs) as executor:
            for _ in executor.map(
                self._refresh_member_boundaries, range(n_members)
            ):
                pass

    def _refresh_member_boundaries(self, member_i: int) -> None:
        self.prepare_member_boundaries(member_i, self.current_cycle_i)
        self.update_bc(member_i)

    def segment_started(self, plan: segments.SegmentPlan) -> bool:
        """
        Whether any member has started running a segment: it has advanced, or has written
        output or restart files into the first cycle's scratch directories. Once that is
        the case, the plan can't change anymore without resetting the segment.
        """

        for cycle_i in plan.cycles:
            if self.state.get_advanced_members(cycle_i):
                return True
        for member_i in range(self.cfg.assimilation.n_members):
            for directory in (
                self.paths.scratch_forecasts_path(plan.first, member_i),
                self.paths.scratch_restart_path(plan.first, member_i),
            ):
                if directory.is_dir() and any(directory.iterdir()):
                    return True
        return False

    def member_initial_state(self, member_i: int, cycle_i: int) -> Path:
        """
        The file a member starts a cycle from, in its directory: the wrfinput, or in
        restart mode (after the first cycle) the WRF restart file at the cycle's start.
        """

        member_path = self.paths.member_path(member_i)
        if self.cfg.assimilation.cycling_mode == "restart" and cycle_i > 0:
            start = self.cycles[cycle_i].start
            return member_path / f"wrfrst_d01_{start:%Y-%m-%d_%H:%M:%S}"
        return member_path / "wrfinput_d01"

    def member_reference_state(self, member_i: int, cycle_i: int) -> Path:
        """
        The initial state of a member at a cycle before it was modified by the analysis or
        perturbations: real.exe's wrfinput, or in restart mode (after the first cycle) the
        restart file WRF wrote at the end of the previous cycle. `cycle` copies the latter
        into the member directory and keeps the original in scratch.
        """

        if self.cfg.assimilation.cycling_mode == "restart" and cycle_i > 0:
            start = self.cycles[cycle_i].start
            return self.paths.scratch_restart_path(cycle_i - 1, member_i) / (
                f"wrfrst_d01_{start:%Y-%m-%d_%H:%M:%S}"
            )
        return self.paths.ic_path(member_i, cycle_i)

    def prepare_member_boundaries(self, member_i: int, cycle_i: int) -> None:
        """
        Puts the boundary conditions for a cycle in the member's directory: the wrfbdy
        (in restart mode, the cycle's records of the experiment-long one, or the whole
        segment's if one starts at this cycle) and the lower boundary (wrflowinp, linked)
        if real.exe made one.
        """

        member_path = self.paths.member_path(member_i)
        cycle = self.cycles[cycle_i]
        bdy_target = member_path / "wrfbdy_d01"
        if self.cfg.assimilation.cycling_mode == "restart":
            end = cycle.forecast_end
            segment = self.segment_of(cycle_i)
            if segment is not None and segment.first == cycle_i:
                end = self.cycles[segment.last].end
            n_records = wrf.extract_boundary_records(
                self.paths.bc_path(member_i, None),
                bdy_target,
                cycle.start,
                end,
            )
            logger.info(f"Member {member_i}: Extracted {n_records} boundary record(s)")
            lowinp = self.paths.lowinp_path(member_i, None)
        else:
            utils.copy(self.paths.bc_path(member_i, cycle_i), bdy_target)
            lowinp = self.paths.lowinp_path(member_i, cycle_i)

        lowinp_target = member_path / "wrflowinp_d01"
        lowinp_target.unlink(missing_ok=True)
        if lowinp.exists():
            lowinp_target.symlink_to(lowinp.resolve())
        elif self.cfg.wrf_namelist.get("physics", {}).get("sst_update", 0) == 1:
            raise FileNotFoundError(
                f"sst_update = 1 but there's no lower boundary file at {lowinp}"
            )

    def cycle_member(self, member_i: int, use_forecast: bool):
        """
        Prepares a member for the next cycle, starting from the analysis of the current
        cycle. Must be done for all members after finishing a set of forward runs and
        running the filter. If you have no observations and filter is skipped, run this
        step with use_forecast=True to use the forecast from the current cycle as the
        analysis.

        In `wrfinput` cycling mode, the next cycle's wrfinput gets the `cycled_variables`
        from the analysis. In `restart` mode, the member continues from its restart file.

        Args:
            member_i: Index of the member to cycle
            use_forecast: Use the forecast from the previous cycle as the analysis
        """

        member_path = self.paths.member_path(member_i)
        if self.cfg.assimilation.cycling_mode == "restart":
            self._cycle_member_restart(member_i, use_forecast)
        else:
            self._cycle_member_wrfinput(member_i, use_forecast)

        # Remove forecast files, log files
        logger.info(f"Cleaning up member directory {member_path}")
        for f in member_path.glob("wrfout*"):
            logger.debug(f"Removing forecast file {f}")
            f.unlink()
        for f in member_path.glob("rsl*"):
            logger.debug(f"Removing log file {f}")
            f.unlink()

    def _cycle_member_wrfinput(self, member_i: int, use_forecast: bool) -> None:
        """`cycle_member` for the `wrfinput` cycling mode"""

        member_path = self.paths.member_path(member_i)
        next_cycle_i = self.current_cycle_i + 1

        # Find analysis/forecast file to use
        wrfout_name = "wrfout_d01_" + self.current_cycle.end.strftime(
            "%Y-%m-%d_%H:%M:%S"
        )
        if use_forecast:
            analysis_file = (
                self.paths.scratch_forecasts_path(self.current_cycle_i)
                / f"member_{member_i:02d}"
                / wrfout_name
            )
        else:
            analysis_file = (
                self.paths.scratch_analysis_path(self.current_cycle_i)
                / f"member_{member_i:02d}"
                / wrfout_name
            )

        if not analysis_file.exists():
            raise FileNotFoundError(analysis_file)
        logger.info(f"Using {analysis_file} as analysis for member {member_i}")

        # Copy the initial & boundary condition files for the next cycle, as is
        icbc_target_file = member_path / "wrfinput_d01"
        utils.copy(self.paths.ic_path(member_i, next_cycle_i), icbc_target_file)
        self.prepare_member_boundaries(member_i, next_cycle_i)

        # Copy cycled variables from the analysis file to the IC file
        with (
            netCDF4.Dataset(analysis_file, "r") as nc_analysis,  # type: ignore
            netCDF4.Dataset(icbc_target_file, "r+") as nc_icbc,  # type: ignore
        ):
            for name in self.cfg.assimilation.cycled_variables:
                if name not in nc_analysis.variables:
                    logger.warning(f"Member {member_i}: {name} not in analysis file")
                    continue
                logger.info(f"Member {member_i}: Copying {name}")
                nc_icbc[name][:] = nc_analysis[name][:]

            # Add experiment name to attributes
            nc_icbc.experiment_name = self.cfg.metadata.name

    def _cycle_member_restart(self, member_i: int, use_forecast: bool) -> None:
        """
        `cycle_member` for the `restart` cycling mode: the member's restart file at the
        end of the current cycle is copied into its directory (the original is kept in
        scratch so the next cycle can be rerun). Unless cycling the forecast, the
        `state_variables` of the analysis are written into it and it is rebalanced.
        """

        member_path = self.paths.member_path(member_i)
        next_cycle_i = self.current_cycle_i + 1
        end = self.current_cycle.end
        source = self.paths.scratch_restart_path(self.current_cycle_i, member_i) / (
            f"wrfrst_d01_{end:%Y-%m-%d_%H:%M:%S}"
        )
        if not source.exists():
            raise FileNotFoundError(source)
        logger.info(f"Member {member_i}: Continuing from {source}")

        # Only the next cycle's restart file belongs in the member directory. A stale
        # wrfinput would be ignored by wrf.exe, but could be modified by mistake.
        for f in member_path.glob("wrfrst_d01_*"):
            f.unlink()
        (member_path / "wrfinput_d01").unlink(missing_ok=True)

        initial_state = self.member_initial_state(member_i, next_cycle_i)
        utils.copy(source, initial_state)

        # Parameter perturbations are applied once and then only carried over in the
        # restart file. WRF only writes the fields its Registry marks for restart, so
        # without that a parameter would silently go back to its default.
        parameters = [
            name
            for name, var in self.cfg.perturbations.variables.items()
            if var.kind == "parameter"
        ]
        if parameters:
            with netCDF4.Dataset(initial_state, "r") as nc_restart:  # type: ignore
                missing = [
                    p for p in parameters if not restart.field_names(nc_restart, p)
                ]
            if missing:
                raise ValueError(
                    f'{", ".join(missing)} (perturbed with kind = "parameter") not in '
                    f"{source}. WRF only writes fields with the `r` flag in its Registry "
                    "to restart files, so the perturbation would be lost after the first "
                    "cycle. Add the flag and rebuild WRF."
                )

        if not use_forecast:
            analysis_file = (
                self.paths.scratch_analysis_path(self.current_cycle_i)
                / f"member_{member_i:02d}"
                / f"wrfout_d01_{end:%Y-%m-%d_%H:%M:%S}"
            )
            if not analysis_file.exists():
                raise FileNotFoundError(analysis_file)
            with (
                netCDF4.Dataset(analysis_file, "r") as nc_analysis,  # type: ignore
                netCDF4.Dataset(initial_state, "r+") as nc_restart,  # type: ignore
            ):
                before = rebalance.balanced_fields(nc_restart)
                written = restart.write_state(
                    nc_restart, nc_analysis, self.cfg.assimilation.state_variables
                )
                rebalance.rebalance(nc_restart, before)
                nc_restart.experiment_name = self.cfg.metadata.name
            logger.info(
                f"Member {member_i}: Wrote the analysis ({', '.join(written)}) into "
                f"{initial_state.name} and rebalanced it"
            )

        self.prepare_member_boundaries(member_i, next_cycle_i)

    def set_wrf_environment(self):
        """
        Adds the environment variables from config's `environment.wrf` to the current environment
        """

        for key, value in self.cfg.environment.wrf.items():
            os.environ[key] = value

    def set_dart_environment(self):
        """
        Adds the environment variables from config's `environment.dart` to the current environment
        """

        for key, value in self.cfg.environment.dart.items():
            os.environ[key] = value

    @property
    def current_cycle(self) -> cycling.CycleInformation:
        """
        Get the current cycle
        """

        return self.cycles[self.current_cycle_i]

    @property
    def all_members_advanced(self) -> bool:
        """
        Check if all ensemble members have been advanced.

        Read from disk rather than from `self.members`, since members advance in other
        processes and the copy loaded at startup goes stale as soon as they do.

        Returns immediately; use `wait_for_all_members()` before gating real work on the
        result.
        """

        return (
            self.state.count_advanced(self.current_cycle_i)
            >= self.cfg.assimilation.n_members
        )

    def wait_for_all_members(self) -> bool:
        """
        Like `all_members_advanced`, but gives a filesystem that is serving a stale
        directory listing a chance to catch up first (see MEMBER_VISIBILITY_TIMEOUT_S).

        Use this in commands that refuse to run unless the ensemble is complete.
        """

        return (
            self.state.count_advanced(
                self.current_cycle_i,
                expect=self.cfg.assimilation.n_members,
                timeout=_member_visibility_timeout(),
            )
            >= self.cfg.assimilation.n_members
        )
