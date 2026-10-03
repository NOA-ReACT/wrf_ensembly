# Postprocessing, and keeping it from running out of memory

`postprocess run --cycle N` reads the raw wrfout in `scratch/forecasts/cycle_NNN/member_*/`
(and `scratch/analysis/...`), runs the processor pipeline (xwrf, variable filtering, custom
processors), and writes the final files to `data/forecasts/cycle_NNN/` and
`data/analysis/cycle_NNN/`:
- `*_mean_cycle_NNN.nc` if `compute_ensemble_mean`
- `*_sd_cycle_NNN.nc` if `compute_ensemble_sd`
- `*_ensemble_cycle_NNN.nc` (every member along a `member` dimension) if `keep_per_member`

Each cycle is independent, so different cycles can run in parallel. `postprocess clean` is a
separate step. See the guardrails in SKILL.md: default to `--keep-wrfout`.

Before running:
- Check that the raw wrfout exist for the cycle and for every member. `data/forecasts` may
  hold files copied from another HPC with nothing in scratch behind them, and then a rerun
  can only make things worse.
- Run `postprocess print-variables-to-keep` to see what will actually be written.

## Where the memory goes

The pipeline streams: one timestep at a time, one member at a time, with Welford
accumulators for the mean and SD. Peak memory is roughly:

> (one member's dataset for one timestep, after the processors) + (mean + M2 accumulators for
> every kept variable, float64) + (per-member buffer, if `keep_per_member`)

× however many cycles run at the same time. The number of members mainly affects run time,
not memory, unless `keep_per_member` is on.

Levers, cheapest first. Propose them, don't apply them silently, because they change what
gets written:

| Lever | Effect |
|---|---|
| Fewer cycles at once. `slurm queue-all-postprocessing` defaults to `--max-parallel 30`! | Memory scales linearly with this. Usually the culprit |
| More memory per job (`mem`/`mem-per-cpu` in `[slurm.directives.postprocess]`, usually in `env_config.toml`) | No change to outputs |
| `variables_to_keep` / `variables_to_keep_ensemble` | Fewer variables means smaller accumulators and buffers |
| `compute_ensemble_sd = false` | Roughly halves the statistics memory, but loses the spread files that plots and validation spread columns use |
| `keep_per_member = false` | Drops the biggest buffer, but loses per-member validation |
| `--only-last-timestep` | Only for quick checks. Not a real postprocess |
| Custom processors (`[[postprocess.processors]]`) | Check whether one of them `.load()`s or makes big intermediate arrays |

## Watching a postprocess run for OOM

First get a baseline. If an earlier cycle finished, its peak is the best estimate:
`sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem` (look at the `.batch` step).
Compare that with the memory the job asks for.

While it runs:
- **Under SLURM**: `sstat -j <jobid>.batch --format=JobID,MaxRSS,AveRSS` for live peak RSS. Check
  every minute or two, since postprocessing is minutes to tens of minutes per cycle. A job that
  climbs steadily from timestep to timestep (rather than levelling off after the first one) is
  leaking, typically through a processor, and will OOM on long cycles.
- **Locally on the dev VM**: find the PID (`pgrep -f "postprocess run"`) and sample
  `ps -o rss=,etime= -p <pid>` alongside `free -g`. To record peak memory for a single run, use
  `/usr/bin/time -v wrf-ensembly ... postprocess run --cycle N`.
- **Progress**: the newest `logs/<ts>-postprocess-run/wrf_ensembly.log` shows which timestep and
  source (forecast/analysis) it's on. Use this to estimate how much is left against the memory trend.

Signs that it ran out of memory: SLURM state `OUT_OF_MEMORY`, `oom-kill`/`oom_kill event` in
`logs/slurm/<jobid>-postprocess.out`, or on the VM a process that just disappeared, with
`dmesg | grep -i oom` (may need sudo) or `journalctl -k | grep -i oom` saying why.

If it gets close to the limit (above ~85% of requested memory, still climbing, with many
timesteps left), tell the user right away with the numbers. Don't kill the job yourself
unless they said you could. An OOM-killed run leaves partial output files in `data/...` for
that cycle. They get rewritten on the next run, but don't treat them as results.
