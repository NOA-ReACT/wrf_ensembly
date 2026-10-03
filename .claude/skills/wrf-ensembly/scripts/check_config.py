#!/usr/bin/env python3
"""
Lint a WRF-Ensembly experiment config for settings that disagree with each other, and
list the experiment-design decisions it encodes so they can be confirmed with the user.

Usage: python check_config.py EXPERIMENT_PATH [--against OTHER_EXPERIMENT]

Reads config.toml merged with env_config.toml (env wins, like the CLI). Read-only: no
log dirs, no state changes. With --against, also prints which decision-level settings
differ from another experiment (useful after deriving one experiment from another).

Output levels:
  ERROR    the experiment will fail or silently do the wrong thing
  WARN     probably unintended, confirm with the user
  DECIDE   a design choice the config makes; surface it if the user didn't state it
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

try:
    import tomllib  # type: ignore
except ImportError:
    import tomli as tomllib  # type: ignore


def deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(exp: Path) -> dict:
    cfg = tomllib.loads((exp / "config.toml").read_text())
    env = exp / "env_config.toml"
    if env.exists():
        cfg = deep_merge(cfg, tomllib.loads(env.read_text()))
    return cfg


def g(cfg, path, default=None):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def minutes(cfg, key, default):
    return int(g(cfg, f"time_control.{key}", default))


def as_dt(v):
    if isinstance(v, datetime):
        return v
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


class Report:
    def __init__(self):
        self.items = []

    def add(self, level, msg):
        self.items.append((level, msg))

    def print(self):
        order = {"ERROR": 0, "WARN": 1, "DECIDE": 2}
        for level, msg in sorted(self.items, key=lambda x: order[x[0]]):
            print(f"{level:7s} {msg}")
        n_err = sum(1 for lvl, _ in self.items if lvl == "ERROR")
        n_warn = sum(1 for lvl, _ in self.items if lvl == "WARN")
        print(f"\n{n_err} error(s), {n_warn} warning(s)")
        return n_err


def check(exp: Path, cfg: dict, r: Report):
    n = int(g(cfg, "assimilation.n_members", 0))

    # --- Inflation -------------------------------------------------------------
    use_inf = bool(g(cfg, "assimilation.use_inflation", False))
    flavor = g(cfg, "dart_namelist.filter_nml.inf_flavor")
    if use_inf and flavor is None:
        r.add("ERROR", "assimilation.use_inflation = true but dart_namelist.filter_nml.inf_flavor is not set: "
                       "every wrf-ensembly command will fail to load the experiment")
    elif use_inf and isinstance(flavor, list) and not any(f > 0 for f in flavor[:2]):
        r.add("ERROR", f"use_inflation = true but inf_flavor = {flavor} (all zero): commands will refuse to run")
    elif not use_inf and isinstance(flavor, list) and any(f > 0 for f in flavor[:2]):
        r.add("WARN", f"inf_flavor = {flavor} enables DART inflation but use_inflation = false: wrf-ensembly will not "
                      "carry inflation restart files between cycles, so each cycle restarts from inf_initial")
    r.add("DECIDE", f"inflation: use_inflation={use_inf}, inf_flavor={flavor}"
                    + (f", inf_initial={g(cfg, 'dart_namelist.filter_nml.inf_initial')}" if use_inf else ""))

    # --- DART namelist completeness -----------------------------------------------
    dn = g(cfg, "dart_namelist", {}) or {}
    if not dn:
        r.add("WARN", "no [dart_namelist]: `setup-dart`/`filter` write input.nml from this section ONLY (no merge "
                      "with the existing file). Setup-dart will fail, or filter will run with an empty namelist")
    else:
        groups = set(dn)
        expected = {"filter_nml", "model_nml", "obs_kind_nml", "assim_tools_nml"}
        missing = sorted(expected - groups)
        if missing:
            r.add("WARN", f"[dart_namelist] lacks {missing}: input.nml is written from this section only, so those "
                          "groups will be absent for filter")

    # --- Meteorology per member -----------------------------------------------------
    pmm = bool(g(cfg, "data.per_member_meteorology", False))
    met = str(g(cfg, "data.meteorology", ""))
    if pmm and "%MEMBER%" not in met:
        r.add("ERROR", f"per_member_meteorology = true but data.meteorology ({met}) has no %MEMBER% placeholder")
    if pmm and "%MEMBER%" in met:
        missing = [i for i in range(n) if not Path(met.replace("%MEMBER%", f"{i:02d}")).is_dir()]
        if missing and len(missing) < n:
            r.add("WARN", f"per-member meteorology dirs missing for members {missing}")
        elif missing:
            r.add("WARN", "no per-member meteorology dirs exist on this machine (maybe configured for another host)")
    if not pmm and "%MEMBER%" in met:
        r.add("WARN", "data.meteorology contains %MEMBER% but per_member_meteorology = false")
    hoz = bool(g(cfg, "data.chemistry.hoz_shift.enabled", False))
    if hoz and not pmm:
        r.add("WARN", "chemistry hoz_shift is enabled but per_member_meteorology = false: chem IC is interpolated "
                      "once, so the per-member shift produces no ensemble spread")
    r.add("DECIDE", f"meteorology: per_member={pmm} ({'preprocessing runs once per member' if pmm else 'one shared IC/BC + perturbations'}), "
                    f"vtable={g(cfg, 'data.meteorology_vtable', 'Vtable.ERA-interim.pl')}, glob={g(cfg, 'data.meteorology_glob', '*.grib')}")

    # --- Chemistry ----------------------------------------------------------------
    chem = g(cfg, "data.chemistry")
    mci = bool(g(cfg, "data.manage_chem_ic", False))
    if mci and not chem:
        r.add("WARN", "manage_chem_ic = true but no [data.chemistry]: `preprocess interpolate-chem` (last step of the "
                      "`slurm preprocessing` jobfile) will fail, and wrf.exe runs with chem_in_opt=1 on whatever chem is in wrfinput")
    if chem and not (exp / "species_map.toml").is_file():
        r.add("ERROR", "[data.chemistry] is set but species_map.toml is missing from the experiment root")
    r.add("DECIDE", f"chemistry IC: manage_chem_ic={mci}, source={g(cfg, 'data.chemistry.model_name')}, "
                    f"hoz_shift={hoz}, multipliers={g(cfg, 'data.chemistry.multipliers') or {}}")

    # --- State / cycled variables --------------------------------------------------
    state = list(g(cfg, "assimilation.state_variables", []) or [])
    cycled = list(g(cfg, "assimilation.cycled_variables", []) or [])
    not_cycled = [v for v in state if v not in cycled]
    r.add("DECIDE", f"DA state ({len(state)} vars): {state}"
                    + (f"; not cycled (analysis discarded at `cycle`, fine for diagnostic vars like W): {not_cycled}"
                       if not_cycled else ""))

    # --- Perturbations -----------------------------------------------------------
    pv = g(cfg, "perturbations.variables", {}) or {}
    desc = []
    for name, p in pv.items():
        every = p.get("perturb_every_cycle", False)
        desc.append(f"{name}:{p.get('operation')} sd={p.get('sd', 1.0)}{' every-cycle' if every else ''}")
        if every and p.get("operation") == "assign" and p.get("midcycle_taper_width", 0):
            r.add("WARN", f"perturbation {name}: midcycle_taper_width has no effect with operation=assign")
    if not pv:
        r.add("WARN", "no [perturbations.variables]: with shared meteorology all members start identical")
    seed = g(cfg, "perturbations.seed")
    r.add("DECIDE", f"perturbations: {', '.join(desc) or 'none'}; seed={seed}"
                    + (" (random: sibling experiments get different ensembles)" if seed is None else ""))

    # --- Timing ------------------------------------------------------------------
    ai = minutes(cfg, "analysis_interval", 360)
    oi = minutes(cfg, "output_interval", 60)
    bi = minutes(cfg, "boundary_update_interval", 180)
    fe = minutes(cfg, "forecast_extension", 0)
    if ai % oi:
        r.add("ERROR", f"analysis_interval ({ai} min) is not a multiple of output_interval ({oi} min): "
                       "no wrfout at the cycle end, so filter has no prior")
    if ai % bi:
        r.add("WARN", f"analysis_interval ({ai}) is not a multiple of boundary_update_interval ({bi}): some cycles "
                      "start at times with no meteorology for real.exe")
    try:
        span = (as_dt(g(cfg, "time_control.end")) - as_dt(g(cfg, "time_control.start"))).total_seconds() / 60
        if span % ai:
            r.add("WARN", f"experiment length ({span:.0f} min) is not a multiple of analysis_interval: "
                          "the last cycle is shorter")
        n_cycles = -(-int(span) // ai)
    except Exception:  # noqa: BLE001
        n_cycles = "?"
    if g(cfg, "time_control.cycles"):
        r.add("WARN", f"per-cycle overrides in [time_control.cycles] for cycles {sorted(g(cfg, 'time_control.cycles'))}")
    r.add("DECIDE", f"cycling: {n_cycles} cycles of {ai} min, output every {oi} min, boundaries every {bi} min, "
                    f"forecast_extension={fe} min")

    # --- Observations --------------------------------------------------------------
    hw = int(g(cfg, "assimilation.half_window_length_minutes", 30))
    if 2 * hw > ai:
        r.add("WARN", f"assimilation window (±{hw} min) is longer than analysis_interval ({ai}): "
                      "observations are assimilated in two consecutive cycles")
    so = set(g(cfg, "observations.superobs", {}) or {})
    tb = set(g(cfg, "observations.temporal_binning", {}) or {})
    if so & tb:
        r.add("ERROR", f"{sorted(so & tb)} in both superobs and temporal_binning: `observations add` will refuse")
    inst = g(cfg, "observations.instruments_to_assimilate")
    r.add("DECIDE", f"observations: window ±{hw} min, instruments={inst or 'ALL in the DB'}, "
                    f"superobs={sorted(so) or 'none'}, temporal_binning={sorted(tb) or 'none'}, "
                    f"thinning={sorted(g(cfg, 'observations.thinning', {}) or {}) or 'none'}, "
                    f"error_inflation={g(cfg, 'observations.error_inflation_factor', {}) or 'none'}")

    # --- Postprocess -----------------------------------------------------------------
    mean = g(cfg, "postprocess.compute_ensemble_mean", True)
    sd = g(cfg, "postprocess.compute_ensemble_sd", True)
    ppm = g(cfg, "postprocess.keep_per_member", False)
    if not mean:
        r.add("WARN", "compute_ensemble_mean = false: plots and `validation interpolate-model` need the mean files")
    for p in g(cfg, "postprocess.processors", []) or []:
        for k, v in (p.get("params") or {}).items():
            if isinstance(v, str) and v.startswith("/") and not Path(v).exists():
                r.add("WARN", f"processor {p.get('processor')} param {k} = {v} does not exist on this machine")
        proc = str(p.get("processor", ""))
        if ":" in proc and proc.split(":")[0].startswith("/") and not Path(proc.split(":")[0]).exists():
            r.add("WARN", f"processor file {proc.split(':')[0]} does not exist on this machine")
    r.add("DECIDE", f"postprocess: mean={mean} sd={sd} per_member={ppm}, "
                    f"variables_to_keep={'all' if not g(cfg, 'postprocess.variables_to_keep') else len(g(cfg, 'postprocess.variables_to_keep'))}")

    # --- Per-member WRF namelist -------------------------------------------------------
    for key in g(cfg, "wrf_namelist_per_member", {}) or {}:
        if not str(key).isdigit():
            r.add("ERROR", f"wrf_namelist_per_member key '{key}' is ignored: the code looks up the plain member "
                           "index as a string (e.g. \"17\"), not 'member_017'")
        elif int(key) >= n:
            r.add("WARN", f"wrf_namelist_per_member has key {key} but there are only {n} members")

    # --- Paths / machine ---------------------------------------------------------------
    for key in ("wrf_root", "wps_root", "dart_root"):
        v = g(cfg, f"directories.{key}")
        if v and not Path(v).exists():
            r.add("WARN", f"directories.{key} = {v} does not exist on this machine")
    sr = str(g(cfg, "directories.scratch_root", "./scratch"))
    if Path(sr).is_absolute():
        r.add("WARN", f"scratch_root is absolute ({sr}); make sure no other experiment uses the same one")
    if (exp / "data/initial_boundary").is_symlink():
        r.add("DECIDE", f"IC/BC are shared: data/initial_boundary -> {(exp / 'data/initial_boundary').resolve()}")
    if not (exp / "env_config.toml").exists():
        r.add("WARN", "no env_config.toml: machine-specific settings (SBATCH directives, paths) live in config.toml")


DIFF_KEYS = [
    "assimilation.n_members", "assimilation.use_inflation", "dart_namelist.filter_nml.inf_flavor",
    "assimilation.state_variables", "assimilation.cycled_variables", "assimilation.half_window_length_minutes",
    "data.per_member_meteorology", "data.meteorology", "data.manage_chem_ic", "data.chemistry",
    "perturbations", "time_control", "domain_control", "observations", "postprocess.keep_per_member",
    "postprocess.compute_ensemble_sd", "postprocess.variables_to_keep", "directories.scratch_root",
    "directories.dart_root",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--against", type=Path, help="another experiment to diff decision-level settings against")
    args = ap.parse_args()

    exp = args.experiment.resolve()
    try:
        cfg = load(exp)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR   could not read config: {e}")
        return 1
    r = Report()
    check(exp, cfg, r)
    n_err = r.print()

    if args.against:
        other = load(args.against.resolve())
        print(f"\nDecision-level differences vs {args.against}:")
        any_diff = False
        for k in DIFF_KEYS:
            a, b = g(other, k), g(cfg, k)
            if a != b:
                any_diff = True
                print(f"  {k}:\n    {args.against.name}: {a}\n    {exp.name}: {b}")
        if not any_diff:
            print("  (none)")
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
