# Troubleshooting failed members, stalled cycles, and filter errors

## Where the evidence is

For a failed `advance-member` of member M in the current cycle:

1. **SLURM output**: `logs/slurm/<arrayid>_<M>-advance_member.out` (array jobs) or
   `logs/slurm/<jobid>-advance_member_<M>.out`. It shows which node ran it, Python tracebacks, and
   SLURM kills (`DUE TO TIME LIMIT`, `oom-kill`, `CANCELLED`). The node name is in the
   SLURM accounting record too: `sacct -j <jobid> --format=JobID,State,ExitCode,NodeList,Elapsed,MaxRSS`.
2. **wrf-ensembly log**: `logs/<timestamp>-ensemble-advance-member_<M>/wrf_ensembly.log`.
   This log gets a copy of `rsl.out.0000` added. It does **not** get the other ranks.
3. **The rsl files themselves**: `work/ensemble/member_<MM>/rsl.error.*` and `rsl.out.*`.
   The crash message is often in one rank that isn't rank 0, so search all of them:
   ```bash
   cd work/ensemble/member_17
   grep -l -iE "cfl|forrtl|sigsegv|segmentation|fatal|error|nan" rsl.error.* | head
   tail -n 40 rsl.error.0000
   ```
   Successful runs get their rsl files archived into the log dir (`rsl.zip`), so routine logs
   are taken care of. A **failed** run's rsl files only exist in the member dir, and the next
   `advance-member` for that member deletes every `rsl.*` before it starts. If anything looks
   suspicious, copy the rsl files and the member's `namelist.input` aside first (for example
   `logs/manual-member_17-cycle_NNN-attempt1/`). Then after the next attempt you have two runs
   to compare: same crash point, same rank, same node?

4. **The member's inputs**: `work/ensemble/member_MM/wrfinput_d01`, `wrfbdy_d01` and
   `namelist.input`. These are what wrf.exe actually read.

## Telling causes apart

| Symptom | Likely cause | Next step |
|---|---|---|
| `RSL file not found` in the log, and no rsl files at all | wrf.exe never started: MPI launcher, environment, missing binary or libs | Read the SLURM .out. Check `slurm.pre_commands`, `environment`, `mpirun_command` in config |
| `cfl` / `w-cfl` lines piling up in rsl.error before the crash | Numerical instability | See "One member keeps crashing" below |
| `forrtl`/SIGSEGV with no CFL warnings | Bad input values, stack/memory, or a bad node | Compare nodes across attempts, check the input fields |
| Same node each time it fails | Hardware/node issue | Exclude the node (`--exclude=` in the `slurm.directives`), tell the user |
| `DUE TO TIME LIMIT` | Walltime too short | Compare with `status runtime-stats` for typical durations |
| `Initial/boundary conditions not found` | `ensemble cycle` didn't finish for the previous cycle, or IC/BC missing for this cycle | `preprocess icbc-status`, check the previous cycle's markers |

## "Member 17 keeps crashing" (only one member)

If only one member crashes while the rest run fine, the physics and namelist are probably OK.
That points to this member's **initial state**. In cycle 0 that state comes from its perturbation,
and in later cycles it comes from the DART analysis increment that `ensemble cycle` copied in (the
`assimilation.cycled_variables`). Check:

- The extremes of the cycled variables in this member's `wrfinput_d01`, compared against a healthy
  member. Look for negative or NaN moisture and chem tracers, absurd T or winds, and sharp local
  spikes near observation locations. Use xarray in the venv for this.
- Whether the previous cycle's filter for this member produced something odd, in
  `scratch/dart/cycle_NNN/dart_member_MM.nc` and `scratch/analysis/cycle_NNN/member_MM/`.
- Whether the crash location (i,j from the CFL lines) lines up with steep terrain or the domain
  boundary.

Common fixes, all of which need the user's agreement because they change the experiment:
lower `time_step` for this cycle, turn on or strengthen `w_damping`/`epssm`, or clip the offending
variable in the member's wrfinput. (There's no per-member "use forecast instead of analysis"
switch; `cycle` only falls back to forecasts for a whole cycle with no analysis.)
Present the evidence and the options. Don't silently patch the input files. A fix that gets
the run through can hide the real problem: lowering the time step or damping a blow-up that
comes from a corrupted or unphysical IC treats the symptom. Say which of the two each option is,
and first ask *why* the bad values are there (a bad analysis increment? a missing variable in
the state? a broken file?).

## Rerunning a member after the fix

- Locally: `wrf-ensembly EXP ensemble advance-member --member 17 [--cores N]`. It runs the current cycle.
- Through SLURM: `slurm run-experiment` only queues the members that haven't advanced yet. If an
  analysis job is still in the queue waiting on the failed array, it will sit there forever
  (`DependencyNeverSatisfied`). Check `squeue` and ask before you `scancel` it and requeue.

## Status disagrees with reality

- Members finished (their wrfout for the cycle end time exists in scratch) but status says they
  didn't. This happens when a job is killed between WRF finishing and the status being written.
  Run `status reconcile --dry-run` first, then without `--dry-run`.
- To force a single member's state: `status set-member <M> <true|false>`.
- The filter complains that not all members advanced, right after the array finished: on NFS the
  status files can show up late. `filter` already waits for them. If it still fails, look at
  `ls status/cycles/cycle_NNN/members/` yourself before you change anything.

## Filter / analysis failures

- `No observation file found at obs/cycle_NNN.obs_seq`: the observations weren't prepared for this
  cycle. With SLURM, `run-experiment` skips the analysis when there are no obs, but a manual
  `filter` call fails.
- The DART work dir `<dart_root>/models/wrf/work` is **shared by every experiment that uses that
  DART build**. If two experiments run the filter at the same time, they overwrite each other's
  `input.nml`, `obs_seq.out` and file lists. Check `squeue` for other experiments' analysis jobs.
- The filter's stdout is in `logs/<ts>-ensemble-filter/filter.log`. DART errors show up as
  `ERROR FROM:` blocks.
