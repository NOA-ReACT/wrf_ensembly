# Experiment design decisions

A config has hundreds of settings, but only a dozen or so are *design decisions*: choices that
change the science of the experiment and are expensive to undo once cycling starts. When you
create an experiment, derive one, or edit its config, these are the ones to bring up with the user.

**The rule**: if the user didn't state a decision explicitly, don't settle it silently, whether
by inheriting it from a parent or template or by picking a "reasonable default". List what the
config currently says and ask. One compact list of questions at the start is much cheaper than
discovering after 40 cycles that inflation was off, or that every member started from the
same meteorology.

Run `python <skill-dir>/scripts/check_config.py EXP` (add `--against PARENT` when deriving). It
prints the ERRORs/WARNs that come from settings conflicting with each other, and the DECIDE lines
that summarise each decision below. Use those DECIDE lines as the basis for your questions.

## The decisions

| Decision | Settings | Couples to / easy to get wrong |
|---|---|---|
| **Inflation**: do we inflate, and how? | `assimilation.use_inflation`, `dart_namelist.filter_nml.inf_flavor` (+ `inf_initial`, `inf_sd_initial`, `inf_damping`, bounds) | Needs **both**: `use_inflation` without `inf_flavor` makes every command fail; `inf_flavor` without `use_inflation` inflates but never carries the restart files between cycles. Inflation restart files go through the **shared** DART work dir. Ask for the flavor and values. Never invent them |
| **Ensemble size** | `assimilation.n_members` | Cores × members per cycle, one meteorology dir per member if per-member, `wrf_namelist_per_member` keys, postprocess time |
| **Where ensemble spread comes from** | `data.per_member_meteorology` (+ `%MEMBER%` path), `[perturbations.variables.*]`, `data.chemistry.hoz_shift` | Per-member meteorology multiplies preprocessing cost by N, and needs N member dirs of GRIB. Without it, perturbations are the only spread: if there are none, all members are identical. `hoz_shift` only adds spread together with per-member meteorology |
| **Perturbations** | operation, `sd`, `gaussian_sigma`, `boundary`, `perturb_every_cycle`, `perturbations.seed` | `seed = None` means sibling experiments get *different* initial ensembles, which matters for a fair A/B comparison. Ask whether to fix it |
| **Chemistry IC/BC** | `data.manage_chem_ic`, `[data.chemistry]` (`model_name`, `path`, `multipliers`), `species_map.toml` | `manage_chem_ic` without `[data.chemistry]`: the `interpolate-chem` step fails and chem comes from whatever real.exe left. The source model has to cover every cycle time |
| **What DART updates and what carries over** | `assimilation.cycling_mode`, `assimilation.state_variables`, `assimilation.cycled_variables`, DART `model_nml` state definition | `wrfinput` mode: state variables that aren't cycled have their analysis thrown away at `cycle` (deliberate for things like W), and WRF's physics state (cloud droplets, TKE, u*) restarts every cycle. Cycle `THM`, not `T`. `restart` mode: everything carries over, `cycled_variables` is unused, `sst_update = 1` keeps the lower boundary current, and restart files cost ~0.8 GB per member per cycle in scratch. Chem tracers: are they in the state? |
| **Cycling cadence** | `time_control.analysis_interval`, `output_interval`, `boundary_update_interval`, `forecast_extension`, `[time_control.cycles]` overrides | `analysis_interval` must be a multiple of `output_interval` (otherwise filter has no prior) and should be a multiple of the boundary interval (otherwise some cycles start without meteorology). The boundary interval has to match the meteorology data frequency |
| **Which observations, and how** | `observations.instruments_to_assimilate`, `assimilation.half_window_length_minutes`, `superobs` / `temporal_binning` / `thinning`, `error_inflation_factor` | Superobs/binning/thinning are fixed **at `add` time** (see observations.md). A window longer than half the interval means each observation gets used twice. Thinning holds observations out for validation (`qc_flag = -1`) |
| **What outputs exist for analysis** | `postprocess.compute_ensemble_mean/sd`, `keep_per_member`, `variables_to_keep(_ensemble)`, processors | Validation needs the mean files, spread columns need the sd files, per-member validation needs `keep_per_member`. All of these drive memory too (postprocessing.md) |
| **DART namelist** | `[dart_namelist.*]` | `setup-dart` writes `input.nml` from this section **alone**. Nothing is merged with the existing file, so the config has to hold the complete namelist |
| **What's shared with other experiments** | `data/initial_boundary` symlink, absolute `scratch_root`, `dart_root` | See derive-experiment.md |
| **Machine-specific settings** | `env_config.toml` → `env_<machine>.toml` | SBATCH directives, paths, processor files. Changes for one machine belong here |

## How to ask

Group the questions and show the current value for each, so the user can just say "yes, yes, change 3":

> Before I set this up, a few design choices the config currently makes. Confirm or change:
> 1. Inflation: **off** (parent has `use_inflation = false`). Keep it off?
> 2. Spread: shared ERA5 + U/V multiplicative perturbations (sd 0.4), **random seed**. Fix the seed so it's comparable with the parent?
> 3. Chemistry IC: `manage_chem_ic = true` but no `[data.chemistry]` source. Which model/path?
> 4. Observations: all instruments in the DB, ±30 min window, no superobs.

Next to the questions, give your scientific read of the setup. It's a separate thing from
the questions and just as important. Things like: does the domain contain the sources that
matter for the target species? Are the observation sites representative of the model grid
(elevation, coastlines, local events such as eruptions or fires in the period)? Is the ensemble
spread mechanism enough to produce spread in the assimilated quantity? Does the observation
operator match a model variable that is actually in the state? Offer opinions and
recommendations freely. Only the *changes* need the user's go-ahead.

Leave out questions the user has already answered in their request, and decisions that can't
matter for this task (e.g. postprocess outputs when the user only asked to preprocess).
If the experiment has no `env_config.toml` (always the case for one just created from a
template), point out the user's existing per-machine env files (e.g. `env_iridium.toml` in a
sibling experiment) and offer to link one. When
you're working non-interactively, or the user said to proceed, make the change they asked for,
leave every unstated decision as inherited, and list those decisions in your report as
"inherited, please confirm".
