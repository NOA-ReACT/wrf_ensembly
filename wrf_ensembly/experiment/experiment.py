import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


from wrf_ensembly import (
    config,
    cycling,
    external,
    models,
    obs_sequence,
    segments,
    utils,
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


SEGMENT_RATE_CYCLES = 5
"""How many of the most recent cycles with member runs the segment rate is taken from"""

SEGMENT_RATE_MIN_RUNS = 3
"""Fewer member runs than this, and segments are sized by `expected_walltime_per_sim_hour`"""


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
    model: models.Model
    members: list[MemberStatus] = []

    state: ExperimentState
    obs: ExperimentObservations
    inflation: InflationConfig
    state_machine: ExperimentStateMachine

    def __init__(self, experiment_path: Path):
        self.cfg = config.read_config(experiment_path / "config.toml")
        self.cycles = cycling.get_cycle_information(self.cfg)

        self.paths = ExperimentPaths(experiment_path, self.cfg)
        self.model = models.create(self)
        self.state = ExperimentState(self.paths, self.cfg.assimilation.n_members)

        # Read the experiment status. Nothing is written here.
        self.load_status()

        self.obs = ExperimentObservations(self.cfg, self.cycles, self.paths)
        self.inflation = InflationConfig.from_config(
            self.cfg, self.paths.data_inflation, self.paths.dart_work_dir, self.state
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

    def advance_member(self, member_idx: int, cores: int) -> bool:
        """
        Run the model to advance a member to the next cycle, or with segments to the end
        of the current segment. The member's directory must be set up (`ensemble setup`
        or `ensemble cycle`). Moves the forecasts to the output directory.

        Args:
            member: Index of the member to advance
            cores: Number of cores to use

        Returns:
            True if the member was advanced successfully
        """

        member = self.members[member_idx]
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

        run = self.model.run_member(member_idx, segment, cycle, cores)
        if run is None:
            return False

        # Every cycle of the segment gets its own status file. The first cycle's is what
        # decides whether the member is advanced (`can_advance_member`, the job array),
        # so it is written last: a job that dies in between counts as not advanced.
        # The runtime goes on the last cycle's file only, since it is one run.
        for cycle_i in reversed(segment):
            if cycle_i == segment[-1] and run.start is not None and run.end is not None:
                self.state.set_member_advanced(
                    cycle_i,
                    member_idx,
                    start=run.start,
                    end=run.end,
                    duration_s=int((run.end - run.start).total_seconds()),
                    simulated_s=run.simulated_s,
                )
            else:
                self.state.set_member_advanced(cycle_i, member_idx)
        self.members[member_idx].advanced = True

        return True

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

        # Write lists of input and output files. The model also puts what DART's
        # interface to it reads (grid, coordinates) in the DART directory.
        priors = self.model.prepare_filter()
        missing_priors = [prior for prior in priors if not prior.exists()]
        if missing_priors:
            logger.error(f"Forecast file(s) missing: {missing_priors[0]}, ...")
            return False
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

        self.paths.scratch_dart_path(self.current_cycle_i).mkdir(exist_ok=True, parents=True)

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

    # Segments (see `segments.py`)

    def observation_cycles(self) -> set[int]:
        """Cycles with an observation file, i.e. where filter will run"""

        return {
            i for i in range(len(self.cycles)) if self.paths.obs_seq_path(i).exists()
        }

    def segment_rate(self, before_cycle: int) -> tuple[float, str]:
        """
        Wall clock seconds per simulated hour to size segments with, and where the number
        came from: the slowest of the member runs of the recent cycles before
        `before_cycle`, or `expected_walltime_per_sim_hour` while there are too few. The
        slowest, since the rate also sets the job array's time limit and every member has
        to finish within it.

        The runs include WRF's startup, so short runs look slower per hour than long
        ones, which errs on the safe side for long segments.
        """

        rates = []
        for stats, _ in self.state.get_recent_runtime_statistics(
            before_cycle, SEGMENT_RATE_CYCLES
        ):
            simulated_s = stats.simulated_s
            if simulated_s is None:
                # Recorded before segments: the run was the cycle
                cycle = self.cycles[stats.cycle]
                simulated_s = (cycle.forecast_end - cycle.start).total_seconds()
            if simulated_s > 0 and stats.duration_s > 0:
                rates.append(stats.duration_s / (simulated_s / 3600))

        if len(rates) >= SEGMENT_RATE_MIN_RUNS:
            return max(rates), f"max of {len(rates)} recent runs"

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
        rate, rate_source = self.segment_rate(first)
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
        # Plans left from before the experiment was moved back by hand
        for start in self.state.get_segment_starts():
            if plan.first < start <= plan.last:
                logger.warning(f"Removing the plan of the segment from cycle {start}")
                self.state.clear_segment_plan(start)
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
        # Segments don't overlap, so only the latest one starting at or before the cycle
        # can cover it. Reading just that one keeps this cheap on shared filesystems.
        starts = [first for first in self.state.get_segment_starts() if first <= cycle_i]
        if not starts:
            return None
        plan = self.state.get_segment_plan(starts[-1])
        return plan if plan is not None and cycle_i in plan else None

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
                self.model.remove_cycle_output(cycle_i, member_i)
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
            if not self.model.member_initial_state(i, cycle_i).exists()
        ]
        if cycle_i == 0 and len(missing) == n_members:
            logger.info("The members are not set up yet, `ensemble setup` will do it")
            return
        if missing:
            logger.warning(
                f"{len(missing)} member(s) have no initial state for cycle {cycle_i} "
                f"(e.g. {self.model.member_initial_state(missing[0], cycle_i)}), their "
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
        self.model.prepare_member_boundaries(member_i, self.current_cycle_i)
        self.model.update_bc(member_i)

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

    def set_model_environment(self):
        """
        Adds the model's environment variables from the config (`environment.wrf` or
        `environment.icon`) to the current environment
        """

        for key, value in self.model.environment.items():
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
