# SLURM: submitting and watching jobs

## How the chain works

`slurm run-experiment` (check `--help` for flags) generates jobfiles in `jobfiles/` and submits:

1. `advance_members_array.job.sh`, an array with one task per **not-yet-advanced** member
   (task id = member index; `--max-parallel` caps how many run at once).
2. The analysis job (`filter` → `analysis` → `cycle` → `update-bc`), submitted with
   `--dependency=afterok:<array>`.
3. Optionally postprocessing (`--run-postprocess`), which depends on the analysis job.
   `--clean-scratch` makes it delete the raw wrfout for that cycle **every cycle**. Ask first.

With `--all-cycles`, the analysis job ends by calling `slurm run-experiment` again, so the
experiment advances itself one cycle at a time until the end or `--run-until N`.

Consequences:
- If any array task fails, `afterok` is never satisfied. The analysis job stays pending as
  `DependencyNeverSatisfied` and the self-resubmitting chain stops. That's the usual
  signature of "the experiment stopped overnight".
- Resubmitting `run-experiment` after fixing a failed member only queues the members still
  pending. Cancel the orphaned analysis job first (ask before `scancel`).
- Job names start with `metadata.name`, which is how to tell experiments apart in `squeue`.

Resource directives come from `[slurm.directives.*]` (`default` merged with
`advance_model` / `make_analysis` / `preprocess` / `postprocess`). The user almost always
keeps these in `env_config.toml` (a symlink to a per-machine file), which overrides
`config.toml`. Read and change them there. Editing `config.toml` alone usually does nothing. `slurm.pre_commands` and
`env_modules` go at the top of every jobfile. Read the generated `jobfiles/*.sh` when a job
fails before wrf-ensembly even starts.

## Keeping an eye on a running job

The user may ask you to watch a job and report back. Run `scripts/orient.py EXP` for the
whole picture, plus targeted checks:

```bash
squeue -u $USER -o "%.12i %.40j %.8T %.10M %.10l %R"     # state, elapsed, limit, reason/node
sacct -j <jobid> --format=JobID,JobName%40,State,ExitCode,Elapsed,NodeList   # finished tasks
ls status/cycles/cycle_NNN/members/ | wc -l               # members recorded as advanced
tail -n 5 logs/slurm/<arrayid>_*-advance_member.out       # progress / errors
```

wrf.exe progress inside a running member: `tail -n 2 work/ensemble/member_MM/rsl.out.0000`
shows the current model time ("Timing for main: time YYYY-MM-DD_HH:MM:SS"). Compare it with
the cycle end time to estimate how long is left. `status runtime-stats` gives typical
durations from earlier cycles.

When polling repeatedly, use the session's scheduling/monitoring tools at a sensible interval.
A member advance typically takes minutes to hours, so check every few minutes, not every few
seconds. Avoid `wrf-ensembly status show` in a loop, because each call creates a log dir.
Report back when: a task fails (go straight into `troubleshooting.md`), the analysis job
starts or finishes, the cycle number moves forward, or something is pending for an unexpected
reason (`DependencyNeverSatisfied`, `QOSMaxJobsPerUserLimit`, ...).

On the dev VM (host `Iridium`), SLURM is a single node with 32 hyperthreads, of which the user
uses 24. Big runs go to the team HPC.
