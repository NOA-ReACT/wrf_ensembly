# Observations: converting, adding, editing, querying

`docs/observations.md` in the repo is current and detailed. It covers the file format, QC
flags, converters, operators, and how to add a new instrument. Read the relevant section
there. This file covers what the docs don't spell out: **at which stage each setting gets
applied**, which decides what has to be redone after a change.

## The lifecycle

```
raw files ──wrf-ensembly-obs convert <instrument>──▶ WRF-Ensembly .parquet (standalone, no experiment)
          ──wrf-ensembly EXP observations add──▶ observations.duckdb       (per experiment)
          ──wrf-ensembly EXP observations prepare-cycles──▶ obs/cycle_NNN.parquet + .obs_seq
          ──ensemble filter──▶ data/diagnostics/cycle_N.obs_seq.final
          ──validation interpolate-model──▶ model_* columns in the DuckDB
```

| Stage | Applied here | Consequence |
|---|---|---|
| `convert` | Instrument QC → `qc_flag > 0`, the native `value_uncertainty`, `metadata` JSON | Changing a converter means reconverting and re-adding |
| `add` | Trimming to the experiment period and domain (needs `data/initial_boundary/wrfinput_d01_cycle_0`, so **preprocessing must be done first**). Projection to WRF `x`,`y`. `superobs`, `temporal_binning`, `thinning` (`qc_flag = -1` for held-out obs) | These are baked into the DB. Changing them, or the domain, means `observations delete` + `add` again for the affected files |
| `prepare-cycles` | Assimilation window (`assimilation.half_window_length_minutes`), `instruments_to_assimilate`, `error_inflation_factor`, only `qc_flag == 0` goes into obs_seq, zero/NaN uncertainty → excluded (99, not stored back in the DB) | Cheap. Rerun after any change here **or after any manual DB edit**. `filter` only reads the `obs_seq` files |
| `interpolate-model` | `model_forecast`, `model_analysis` (from the postprocessed **mean** files), `model_*_spread` (from the **sd** files; skipped for operator-based quantities like HLOS wind) | Needs postprocessing done. Each run recomputes the columns from scratch |

Re-adding a file (`add` with the same filename) **replaces** its rows. That also wipes the
`model_*` columns for those rows, so rerun `interpolate-model` afterwards.

Converters: `wrf-ensembly-obs convert --help` lists them. Each one has its own options.
`wrf-ensembly-obs operations dump-info FILE` is the quick way to check a converted file
before adding it. Add with `--jobs`. On the dev VM keep that ≤ 24.

## Derived vs. source columns

When analysing data, keep track of which columns came from the instrument and which ones
WRF-Ensembly computed:

- **From the source file**: `instrument`, `quantity`, `time`, `longitude`, `latitude`, `z`,
  `z_type`, `value`, `value_uncertainty`, `qc_flag` (positive values), `orig_coords`,
  `orig_filename`, `metadata`.
- **Changed at add time**: `x`/`y` (projection). For superobbed or binned pairs, `value` and
  `value_uncertainty` are **bin aggregates**, not raw measurements: with
  `reduce_instrument_error` the uncertainty has already been divided by √n. `qc_flag = -1`
  comes from thinning.
- **Filled in later**: `model_forecast`, `model_analysis`, `model_forecast_spread`,
  `model_analysis_spread`. NULL means not interpolated yet, outside model coverage, or (for
  spread) an operator-based quantity. Per-member equivalents are in
  `data/validation/model_member_{forecast,analysis}.parquet` (from
  `interpolate-model-per-member`, which needs `keep_per_member`).
- Validation outputs (first departures, lead-time skill, curtains, spread) are files under
  `data/validation/`, not DB columns.

## Querying the DB

```python
import duckdb
con = duckdb.connect("EXP/observations.duckdb", read_only=True)
con.execute("SET TimeZone='UTC'")
df = con.execute("""
    SELECT instrument, quantity, count(*) n,
           avg(value - model_forecast) AS bias_omb,
           count(model_forecast) AS n_interp
    FROM observations WHERE qc_flag = 0
    GROUP BY ALL ORDER BY n DESC
""").fetchdf()
```

- Always open with `read_only=True` for analysis. DuckDB allows only **one writing process**,
  and a read-write connection locks the file against everyone else. So if a `wrf-ensembly`
  command (`add`, `interpolate-model`) is running, your writable connection fails, or theirs does.
- `time` is `TIMESTAMPTZ`. Set the session timezone to UTC as above, or the output shifts.
- `metadata` is JSON: `metadata->>'site_name'`, `json_extract(metadata, '$.azimuth')`.
- To turn rows back into the original swath/profile arrays, use `orig_coords` and the helpers in
  `wrf_ensembly.observations.utils` (`reconstruct_array`). See docs "Array Reconstruction".
- `observations show` and `observations cycle-summary` give quick overviews without writing SQL.
  Remember that each call creates a log dir.

## Editing the DB by hand (e.g. observation error)

Typical case: "set the error of instrument X / quantity Y to 0.05" or "scale it by 1.5".

1. **Can config do it instead?** A *multiplicative* change for a whole instrument.quantity
   pair belongs in `observations.error_inflation_factor` (applied at `prepare-cycles`). It's
   reproducible, reversible, and needs no DB edit. Suggest that first.
2. Otherwise: make sure nothing else is writing (check `squeue` and `pgrep -f wrf-ensembly`), then
   **back up the DB** (`cp observations.duckdb observations.duckdb.bak-YYYYMMDD`). It's the
   only copy of the add-time processing, so reconverting everything would take a long time.
3. Run a `SELECT count(*)` with the exact `WHERE` you'll use, show the user the count, then
   run the `UPDATE` in a transaction:
   ```sql
   BEGIN;
   UPDATE observations SET value_uncertainty = 0.05
   WHERE instrument = 'X' AND quantity = 'Y';
   -- check the count again here
   COMMIT;
   ```
   For superobbed pairs, consider whether an absolute value makes sense, since the stored
   uncertainty is already a bin aggregate.
4. Rerun `observations prepare-cycles` (all cycles, or the affected ones with `--cycle`).
   Until then, the filter keeps using the old obs_seq files.
5. Tell the user exactly what changed (WHERE clause, row count, old → new) and where the backup is.

Same procedure for other edits (flipping `qc_flag` to exclude a bad station, and so on).
