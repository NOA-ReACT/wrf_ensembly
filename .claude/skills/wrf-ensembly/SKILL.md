---
name: wrf-ensembly
description: Operate WRF-Ensembly data assimilation experiments (WRF/WRF-Chem + DART) — inspecting experiment state, diagnosing crashed members or failed filter/analysis jobs, deriving a new experiment from an existing one, running WPS/real preprocessing on new input data, postprocessing (and watching it for OOM), converting/adding observations and querying or editing the observations DuckDB, and submitting or watching SLURM jobs. Use this whenever the user mentions an experiment directory, `wrf-ensembly` commands, a cycle or member number ("member 17 keeps crashing", "why did cycle 40 stop"), rsl files, wrfinput/wrfbdy, obs_seq, observations.duckdb, filter, postprocess, `wrf-ensembly-obs`, or asks to babysit a running ensemble job — even if they don't name the tool.
---

# Working with WRF-Ensembly experiments

The CLI and its source are the ground truth. Written docs (`docs/usage.md`, parts of
`CLAUDE.md`) have drifted from the code — command names in particular. Before running a
command you haven't run this session, check `wrf-ensembly EXP <group> --help`. If something
here disagrees with the code, trust the code.

Activate the repo's `.venv` first (`source <repo>/.venv/bin/activate`) unless
`wrf-ensembly` is already on PATH.

## Mental model

An **experiment** is a directory: `config.toml` + data + model copies + state.
If an `env_config.toml` sits next to `config.toml` (usually a symlink to a per-machine
file like `env_iridium.toml`), it is deep-merged on top and **wins**. It typically holds
SBATCH directives, paths and environment. Read both before reasoning about settings,
and put machine-specific edits in the env file, not `config.toml`. Every
command is `wrf-ensembly <EXP> <group> <command>`. One **cycle** is:

1. `ensemble advance-member` — each member runs wrf.exe (separate jobs, often separate nodes)
2. `ensemble filter` — DART assimilates `obs/cycle_NNN.obs_seq` into the forecasts
3. `ensemble analysis` — collect DART output into analysis files
4. `ensemble cycle` — copy next cycle's IC/BC into each member dir, overwrite the
   `cycled_variables` from the analysis, bump the current cycle

Where things live (paths that matter for debugging):

| What | Where |
|---|---|
| Current cycle, per-member advancement | `status/experiment.json`, `status/cycles/cycle_NNN/members/member_MM.json`, marker files `filter_complete` etc. |
| Member run dir (wrf.exe, namelist, **rsl files of the last run**) | `work/ensemble/member_MM/` |
| IC/BC from real.exe (expensive) | `data/initial_boundary/` |
| Raw wrfout (expensive) | `scratch/forecasts/cycle_NNN/member_MM/` — `scratch_root` may point elsewhere, check config |
| Postprocessed output (expensive) | `data/forecasts/`, `data/analysis/` |
| One log dir per command invocation | `logs/YYYY-MM-DD_HHMMSS-<command>/wrf_ensembly.log` (+ `rsl.zip` on success) |
| Observations database | `observations.duckdb` (experiment root), per-cycle `obs/cycle_NNN.{parquet,obs_seq}` |
| SLURM stdout | `logs/slurm/<jobid>-<name>.out`, array jobs `<arrayid>_<member>-advance_member.out` |
| DART work dir | `<dart_root>/models/wrf/work` — **shared by every experiment using that DART build** |

Member and cycle numbers are zero-based and zero-padded on disk (`member_17`, `cycle_042`).

## Start every task by orienting

Run `python <skill-dir>/scripts/orient.py <EXP>`. It reads state without side
effects and prints: current cycle + state markers, which members are advanced, key config
fields, newest log dirs with error lines, recent SLURM output, and the user's queue.
Then read `config.toml` sections relevant to the task.

Note: every `wrf-ensembly` command — even `status show` — creates a new `logs/` dir. Fine
occasionally, but when polling a running job, use `orient.py`/`squeue`/`ls status/`
instead so the logs dir doesn't fill with noise.

## Science stays visible

The user wants a second pair of scientific eyes, not just an operator. Bring your own opinions:
whether the domain covers the relevant sources, whether an observation site is representative
(mountain stations, volcanic events, coastal pixels), whether the DA state and the observations
fit together, whether a result looks physically plausible. Mention these even when nobody asked.
The user often misses things and wants them pointed out.

What must never happen is a science problem getting **quietly resolved on the way to some
other goal**. Watch for these moves in particular:
- turning a feature off (chemistry IC, inflation, an instrument, a processor) because its data
  or file is missing
- adding or removing a variable (state, cycled, `variables_to_keep`) or loosening a check
  because that makes a crash or error go away
- clipping or patching model fields, raising errors, or widening windows just to get a run through
- choosing a default for something the user didn't specify

Any of these may be the right call, but they're the user's call. Stop and frame it as a
science question: what's happening, why it matters for the results, the options and your
recommendation. If you're working non-interactively, finish the parts that don't depend on
the answer and list the open science questions at the top of your report, not buried at the end.

Visible doesn't mean stalled. When the user needs a result (a deadline, "just get it done") and
a reversible path exists, take it and make the choice impossible to miss: back up what you
change, prefer the option that keeps results consistent with what came before (e.g. the same
operator/table version as earlier cycles), put stop-gap outputs where they can't be confused with
the real ones (or label them clearly), and lead the report with what you chose and why. Stop and
wait only when every path forward would bake in a choice that's hard to undo, or when the inputs
themselves look wrong.

### Design decisions: surface them, don't inherit them silently

A dozen or so settings are science decisions that are expensive to undo once cycling starts:
inflation, where ensemble spread comes from (per-member meteorology vs perturbations, seed),
chemistry IC source, state/cycled variables, cycling cadence, which observations are used and how,
and what postprocessed outputs exist. Many of them depend on each other (e.g. `use_inflation`
needs `filter_nml.inf_flavor`). Whenever you create, derive or edit an experiment config, run
`python <skill-dir>/scripts/check_config.py EXP` (plus `--against PARENT` when deriving), fix or
report its ERRORs/WARNs, and ask the user about any decision they didn't state. Don't adopt a
template or parent's choice, or invent a default, without saying so. Do add your own view: if a
choice looks scientifically wrong for what they're trying to do (e.g. AOD assimilation with no
aerosol tracers in the DA state), say so and recommend something, but let them decide.
`references/experiment-decisions.md` has the list and a format for asking.

## Guardrails

**Ask before anything that deletes or overwrites expensive outputs**: `data/forecasts`,
`data/analysis`, `data/initial_boundary`, and raw wrfout in `scratch/forecasts`. They can be
regenerated but it takes hours to days. Watch for indirect deletion too:

- `postprocess clean` removes raw wrfout **by default**. Pass `--keep-wrfout` unless the user
  has said the raw files can go (sometimes they're too big to keep, which is the user's call).
- `--clean-scratch` on `slurm run-experiment` / `slurm postprocess` deletes them too, every cycle.
- `postprocess run` rewrites `data/forecasts`/`data/analysis` for that cycle. Those files may
  have been copied in from another HPC with **no raw wrfout behind them**. Check that the
  scratch wrfout exist before rerunning, or you replace good output with nothing.
- `preprocess real --cycle N` rewrites that cycle's IC/BC. If `data/initial_boundary` is a
  symlink to another experiment, you are overwriting *that* experiment's files.
- `experiment copy-model --force` wipes `work/ensemble/member_*` (including rsl evidence).
- `status reset` / `ensemble reset-cycle` don't delete data but rewind state; confirm intent.

Also confirm before submitting SLURM jobs (`sbatch`, `slurm run-experiment`, ...) on the
team HPC — they consume a shared allocation. Re-running `advance-member` for one member
locally on the dev VM is fine to just do. On the dev VM (`Iridium`), use at most 24 cores
(`--cores`, `--jobs`). It has 32 hardware threads, but they're hyperthreads. The user runs the very large experiments by hand
on a separate big HPC; if a path or hostname suggests that machine, only inspect, don't act.

Status files are plain JSON/marker files, written atomically, one writer each. Prefer the
`status` commands (`set-member`, `set-experiment`, `reconcile`) to fix them, but hand
inspection with `ls`/`cat` is expected and safe.

## Common tasks

Read the matching reference before acting — each has the non-obvious parts:

- **A member keeps crashing / a cycle stalled / filter failed** → `references/troubleshooting.md`
- **New experiment from a template or an existing one** → `references/experiment-decisions.md`, then `references/derive-experiment.md`
- **Preprocessing with new meteorology or chem data** → `references/preprocessing.md`
- **Submitting jobs or keeping an eye on a running job** → `references/slurm.md`
- **Postprocessing, including memory/OOM** → `references/postprocessing.md`
- **Observations: converting, adding, editing or querying the DB, validation columns** → `references/observations.md`

For anything else (config options, processors, plots), the repo docs
in `docs/` and the source under `wrf_ensembly/` are good; cross-check commands with `--help`.

## Reporting back

Lead with what you found and the evidence (file path + the relevant log lines), then what
you propose to do. If you changed state or files, list exactly which.
