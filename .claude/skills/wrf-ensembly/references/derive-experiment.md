# Deriving a new experiment from an existing one

Typical request: "make a copy of EXP_A but with X changed". No single command does this. The
work is (1) building the new config and (2) deciding which expensive inputs can be **reused**
and which must be **regenerated**. Getting (2) right saves hours. Getting it wrong either wastes
compute or quietly corrupts the parent experiment.

## 1. Understand the parent

- Read its `config.toml` and run `scripts/orient.py PARENT`.
- Check what exists: `preprocess icbc-status`, `ls obs/`, and whether `data/initial_boundary` is
  already a symlink to some other experiment.
- Write down the changes the user asked for, mapped to config keys. If a change is ambiguous
  (e.g. "more members", "stronger perturbations"), confirm the exact values.

## 2. Create the new experiment

```bash
wrf-ensembly NEW experiment create default      # makes the dir tree + empty status
cp PARENT/config.toml NEW/config.toml           # then edit NEW/config.toml
cp -P PARENT/env_config.toml NEW/ 2>/dev/null   # keep it a symlink if it is one
```

If the parent's `env_config.toml` is a relative symlink (for example `-> env_iridium.toml`),
copy the target file too, or point the link at the parent's file. Without it, the new
experiment silently loses the user's SBATCH directives and machine paths.

Edits that are always needed:
- `metadata.name` and `metadata.description`. SLURM job names come from `name`, so give it a
  distinct one. Otherwise you can't tell the two experiments apart in `squeue`.
- `directories.scratch_root`: if the parent uses an **absolute** path, the new experiment writes
  into the same scratch tree and overwrites the parent's raw forecasts. Point it somewhere new.
  A relative path (the default `./scratch`) is fine.

Then apply the requested changes. Some changes only work together with others. Most notably,
`assimilation.use_inflation = true` makes every command fail unless
`[dart_namelist.filter_nml] inf_flavor` (and the other `inf_*` values) are also set. Ask the user
for the inflation settings rather than copying an example. If you have to proceed without
answers, mark the values as placeholders in the config and in your report.

Run `scripts/check_config.py NEW --against PARENT`. It catches broken combinations and lists the
decision-level differences. Every decision that's the same as the parent was *inherited*, not
chosen, so list the important ones for the user to confirm (see experiment-decisions.md).

Show the user the diff (`diff PARENT/config.toml NEW/config.toml`, and the env file if it changed)
before going further.

`experiment setup-dart` (and every `filter`) overwrites `<dart_root>/models/wrf/work/input.nml`
with only the groups in `[dart_namelist]`. Nothing is merged, so the config has to hold the full
namelist.

Things to know about the shared DART build: `directories.dart_root` is used as the filter's
working directory (`<dart_root>/models/wrf/work`). If both experiments will be cycling at the same
time, their filter steps can collide. Point that out to the user. A separate DART copy is the
clean fix.

## 3. Decide what to reuse

| If the changes touch... | IC/BC (`data/initial_boundary`) | Observations (`obs/`) |
|---|---|---|
| Only DA settings (`assimilation.*` except `n_members`, inflation, localisation, cycled vars), perturbation settings, postprocess | **Reuse** | **Reuse** |
| `n_members` and `data.per_member_meteorology = false` | Reuse | Reuse |
| `observations.*` (windows, superobs, thinning, error inflation) or which instruments | Reuse | **Regenerate**: `observations prepare-cycles` |
| `time_control` (cycle length, start/end), `domain_control`, `geogrid`, `data.meteorology`, chem IC source, `wrf_namelist` options that real.exe uses (levels, `chem_opt`, ...) | **Regenerate** (full preprocessing) | Regenerate (different cycle windows) |
| Only runtime physics/dynamics options in `wrf_namelist` that real.exe doesn't read | Usually reuse. Ask if unsure | Reuse |

**Reusing IC/BC**: replace the new experiment's empty `data/initial_boundary` with a symlink to
the parent's. That's what `ensemble setup-from-other-experiment` does too. It's safe because
member setup copies the files into `work/ensemble/member_*` before changing them. The danger is
running `preprocess real` in the child later, because that overwrites the parent's files through
the symlink. If preprocessing is needed later, remove the symlink and make a real directory first.

**Reusing observations**: copy `obs/` (the per-cycle `cycle_NNN.obs_seq` files) and
`observations.duckdb`. These are small, so copy rather than symlink. That way a later
`prepare-cycles` in the child can't modify the parent.

**Starting from the middle of the parent's run** (spin-up reuse): `ensemble setup-from-other-experiment PARENT --cycle N`
requires identical `domain_control` and `time_control`, and needs the parent's raw wrfout for cycle N
to still be in its scratch directory. Check that before promising it.

Both experiments must have the same `assimilation.cycling_mode`. In restart mode the parent's
restart files at the end of cycle N are copied too, from `scratch/restart/cycle_NNN/`, and `cycle`
deletes old ones as the parent moves on. If a branch point is planned, the parent needs
`assimilation.keep_restart_files_for_cycles = [N]` (or `keep_restart_files = true`) from the start.
The command lists `[wrf_namelist]` differences as a warning: tuning parameters are fine, but
switching physics or chemistry schemes against the parent's restart state may not work.

## 4. Bring it up

```bash
wrf-ensembly NEW experiment copy-model
wrf-ensembly NEW experiment setup-dart
# preprocessing only if not reused (see preprocessing.md)
wrf-ensembly NEW ensemble setup
wrf-ensembly NEW ensemble generate-perturbations --jobs 8
wrf-ensembly NEW ensemble apply-perturbations --jobs 8
wrf-ensembly NEW ensemble update-bc
```

Confirm each with `--help` if unsure. Then `scripts/orient.py NEW` to check that cycle 0 is
ready, with no members advanced. Ask before launching (`slurm run-experiment`).

Finish by telling the user: what was reused (and through symlink or copy), what was regenerated,
the config diff, and anything that is still shared (DART dir, meteorology input).
