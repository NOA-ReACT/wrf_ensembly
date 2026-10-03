# Preprocessing (WPS → real.exe → chem IC) with new input data

Preprocessing produces `data/initial_boundary/wrfinput_d01_cycle_N` and `wrfbdy_d01_cycle_N`
(per-member subdirs if `data.per_member_meteorology`). It's slow, and its outputs are among the
expensive files. **If IC/BC already exist for any cycle, rerunning overwrites them.** Confirm
first, and check that `data/initial_boundary` isn't a symlink into another experiment.

## Pipeline

```
preprocess setup        # copies WRF/WPS into work/preprocessing/
preprocess geogrid      # once per domain
preprocess ungrib       # links GRIB files from data.meteorology, runs ungrib.exe
preprocess metgrid
preprocess real --cycle N      # once per cycle; parallel-safe (own work dir per cycle)
preprocess interpolate-chem    # WRF-Chem only, edits the IC files in place
```

With per-member meteorology, run `ungrib`/`metgrid`/`real`/`interpolate-chem` once per member
with `--member`, in that order. Ungrib and metgrid share one WPS dir, so members must be
processed one at a time through those steps.

`slurm preprocessing` writes `jobfiles/preprocess.sh`, which chains all of these. It's the
normal way to run on the HPC.

## Checklist for new input data

Do these checks *before* starting. Each one is cheap compared to a failed real.exe:

1. **Location and glob**: `data.meteorology` points at the directory, and `data.meteorology_glob`
   (default `*.grib`) matches the new files. Run `ls <dir>/<glob> | wc -l` and compare against
   what the cycles need. For per-member data, the path has `%MEMBER%` and expands to two-digit
   member numbers.
2. **Vtable matches the source**: `data.meteorology_vtable` (relative names resolve in
   `work/preprocessing/WPS/ungrib/Variable_Tables/`). Look at a GRIB file
   (`grib_ls`/`cdo sinfo`) to see whether the data is ERA5 pressure-level, model-level, GFS, etc.
   A wrong Vtable often still "succeeds" but produces missing fields, which then break metgrid
   or real.
3. **Time coverage**: run `experiment cycle-info` to list the cycle start/end times. The GRIB
   files must cover the first cycle start through the last cycle end, at the
   `interval_seconds` the namelist expects.
4. **Chemistry** (if `data.chemistry` is set): `species_map.toml` must exist in the experiment
   root. `interpolate-chem` warns about cycle times missing from the global model and
   interpolates in time. Report those warnings to the user rather than letting them pass.
5. **Domain**: `geo_em` only needs regenerating when `domain_control`/`geogrid` changed.

## Verifying and debugging

- `preprocess icbc-status` shows which cycles have IC/BC.
- ungrib/metgrid/geogrid output is saved as `ungrib.log`/`metgrid.log`/`geogrid.log` in that
  command's `logs/<ts>-preprocess-*/` dir. Ungrib/metgrid only count as successful if the
  "Successful completion" line appears.
- real.exe runs in `work/preprocessing/real_cycle_N/`. That dir is deleted on success. Pass
  `--no-auto-clean-up` to keep it, and look at its `rsl.error.0000` on failure. Common failures:
  `num_metgrid_levels` mismatch (the namelist value must match the met_em files; see
  `ncdump -h met_em*.nc | grep num_metgrid_levels`) and missing soil fields (wrong Vtable).
- After `interpolate-chem`, check a few chem tracers in a `wrfinput` for zeros/NaNs.

`preprocess clean` deletes `work/preprocessing/` (the WPS/WRF copies, intermediate files and
met_em). That's cheap to regenerate. It does not touch `data/initial_boundary`.
