#!/usr/bin/env python3
"""
Read-only snapshot of a WRF-Ensembly experiment, for getting oriented quickly.

Usage: python orient.py EXPERIMENT_PATH [--logs N] [--slurm N]

Reads files directly instead of calling `wrf-ensembly`, so it creates no log directories
and is safe to run repeatedly while polling a job. Works with any Python >= 3.8; config
parsing needs tomllib (3.11+) or tomli, and is skipped otherwise.
"""

import argparse
import getpass
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ERROR_RE = re.compile(
    r"(?<!rsl\.)error|failed|traceback|exception|fatal|cancelled|time limit|oom|sigsegv|forrtl",
    re.IGNORECASE,
)


def section(title):
    print(f"\n=== {title} ===")


def load_config(exp):
    path = exp / "config.toml"
    if not path.is_file():
        print(f"!! no config.toml at {path} - is this an experiment directory?")
        return None
    try:
        import tomllib  # type: ignore
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            print("(tomllib/tomli unavailable, skipping config summary - use the repo's .venv)")
            return None
    try:
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(f"!! config.toml does not parse: {e} - wrf-ensembly will refuse to run too")
        return None

    # Same precedence as wrf_ensembly.config.read_config: env_config.toml wins
    env = exp / "env_config.toml"
    if env.exists():
        target = f" -> {env.resolve().name}" if env.is_symlink() else ""
        print(f"env_config.toml{target} present, merged over config.toml")
        try:
            with open(env, "rb") as f:
                cfg = _deep_merge(cfg, tomllib.load(f))
        except tomllib.TOMLDecodeError as e:
            print(f"!! env_config.toml does not parse: {e}")
    return cfg


def _deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def config_summary(cfg, exp):
    section("Config")
    g = lambda *keys: _get(cfg, keys)  # noqa: E731
    print(f"name:            {g('metadata', 'name')}")
    print(f"members:         {g('assimilation', 'n_members')}")
    tc = cfg.get("time_control", {})
    print(
        f"time:            {tc.get('start')} -> {tc.get('end')}, "
        f"analysis every {tc.get('analysis_interval')} min"
    )
    dc = cfg.get("domain_control", {})
    print(f"domain:          {dc.get('xy_size')} pts @ {dc.get('xy_resolution')} km")
    print(f"meteorology:     {g('data', 'meteorology')}  (per-member: {g('data', 'per_member_meteorology') or False})")
    chem = g("data", "chemistry")
    print(f"chem IC:         {'yes' if chem else 'no'}  manage_chem_ic={g('data', 'manage_chem_ic') or False}")
    print(f"dart_root:       {g('directories', 'dart_root')}  (work dir is shared by all experiments using it)")
    scratch = g("directories", "scratch_root") or "./scratch"
    note = "ABSOLUTE - may be shared with other experiments" if Path(str(scratch)).is_absolute() else "relative to experiment"
    print(f"scratch_root:    {scratch}  ({note})")
    pp = cfg.get("postprocess", {})
    print(
        f"postprocess:     mean={pp.get('compute_ensemble_mean', True)} sd={pp.get('compute_ensemble_sd', True)} "
        f"per_member={pp.get('keep_per_member', False)} vars_kept={len(pp.get('variables_to_keep') or []) or 'all'}"
    )
    directives = _get(cfg, ("slurm", "directives")) or {}
    for name, d in directives.items():
        if isinstance(d, dict) and d:
            print(f"sbatch[{name}]:".ljust(17) + " " + ", ".join(f"{k}={v}" for k, v in d.items()))


def _get(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def state_summary(exp, n_members):
    section("State (status/)")
    exp_file = exp / "status" / "experiment.json"
    if not exp_file.is_file():
        legacy = [f.name for f in exp.iterdir() if f.name.startswith("status.") or f.suffix in (".db", ".sqlite")]
        if legacy:
            print(f"!! legacy status layout ({', '.join(legacy)}); the current CLI does not read it")
            print("   and there is no migration command - ask the user how to proceed")
        else:
            print("!! status/experiment.json missing - experiment not created yet?")
        return None
    try:
        current = json.loads(exp_file.read_text()).get("current_cycle")
    except (ValueError, OSError) as e:
        print(f"!! could not read {exp_file}: {e}")
        return None
    print(f"current cycle:   {current}")

    cycle_dir = exp / "status" / "cycles" / f"cycle_{current:03d}"
    markers = [m for m in ("filter_complete", "analysis_complete", "cycle_complete") if (cycle_dir / m).exists()]
    print(f"markers:         {', '.join(markers) or 'none'}")

    advanced, records = set(), {}
    members_dir = cycle_dir / "members"
    if members_dir.is_dir():
        for f in sorted(members_dir.glob("member_*.json")):
            try:
                rec = json.loads(f.read_text())
            except (ValueError, OSError):
                print(f"!! unreadable status file {f}")
                continue
            i = int(f.stem.split("_")[1])
            records[i] = rec
            if rec.get("advanced"):
                advanced.add(i)
    if n_members:
        missing = [i for i in range(n_members) if i not in advanced]
        print(f"advanced:        {len(advanced)}/{n_members}")
        if missing:
            print(f"not advanced:    {_ranges(missing)}")
    else:
        print(f"advanced:        {sorted(advanced)}")

    hosts = {}
    for i, rec in records.items():
        if rec.get("host"):
            hosts.setdefault(rec["host"], []).append(i)
    if hosts:
        print("hosts:           " + "; ".join(f"{h}: {_ranges(sorted(v))}" for h, v in hosts.items()))

    if (exp / "data" / "initial_boundary").is_symlink():
        print(f"NOTE: data/initial_boundary -> {(exp / 'data' / 'initial_boundary').resolve()} (shared IC/BC!)")
    return current


def _ranges(nums):
    out, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(out)


def error_lines(path, limit=4):
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    hits = [l.strip() for l in lines if ERROR_RE.search(l)]
    return hits[-limit:]


def recent_logs(exp, n):
    section(f"Newest {n} command logs (logs/)")
    logs = exp / "logs"
    if not logs.is_dir():
        print("(none)")
        return
    dirs = sorted((d for d in logs.iterdir() if d.is_dir() and d.name != "slurm"), key=lambda d: d.name)[-n:]
    for d in dirs:
        hits = error_lines(d / "wrf_ensembly.log")
        flag = "  <-- errors" if hits else ""
        print(f"{d.name}{flag}")
        for h in hits:
            print(f"    {h[:200]}")


def recent_slurm(exp, n):
    section(f"Newest {n} SLURM outputs (logs/slurm/)")
    d = exp / "logs" / "slurm"
    if not d.is_dir():
        print("(none)")
        return
    files = sorted(d.glob("*.out"), key=lambda p: p.stat().st_mtime)[-n:]
    for f in files:
        hits = error_lines(f, limit=2)
        print(f"{f.name}{'  <-- errors' if hits else ''}")
        for h in hits:
            print(f"    {h[:200]}")


def member_rsl(exp, n_members, current):
    """Members with leftover rsl files that don't end in success = evidence of a crash."""
    section("Member work dirs whose last wrf.exe run has not succeeded")
    ens = exp / "work" / "ensemble"
    found = False
    for m in sorted(ens.glob("member_*")) if ens.is_dir() else []:
        rsl0 = m / "rsl.out.0000"
        if not rsl0.is_file():
            continue
        try:
            text = rsl0.read_text(errors="replace")
        except OSError:
            continue
        if "SUCCESS COMPLETE WRF" in text:
            continue
        last = [l for l in text.splitlines() if l.strip()][-1:] or ["(empty)"]
        print(f"{m.name}: no SUCCESS line (crashed, or still running); last: {last[0][:150]}")
        found = True
    if not found:
        print("(none - either all ok, still running, or rsl files already cleaned)")


def queue(exp_name):
    section("SLURM queue (this user)")
    if not shutil.which("squeue"):
        print("(squeue not available)")
        return
    try:
        out = subprocess.run(
            ["squeue", "-u", getpass.getuser(), "-o", "%.12i %.45j %.9T %.10M %.10l %R"],
            capture_output=True, text=True, timeout=20,
        ).stdout.rstrip()
    except (subprocess.SubprocessError, OSError) as e:
        print(f"(squeue failed: {e})")
        return
    lines = out.splitlines()
    if len(lines) <= 1:
        print("(no jobs)")
        return
    print(lines[0])
    for l in lines[1:]:
        mark = "  <-- this experiment" if exp_name and exp_name in l else ""
        print(l + mark)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--logs", type=int, default=8, help="how many command log dirs to show")
    ap.add_argument("--slurm", type=int, default=6, help="how many SLURM .out files to show")
    args = ap.parse_args()

    exp = args.experiment.resolve()
    print(f"Experiment: {exp}")
    cfg = load_config(exp)
    n_members = _get(cfg, ("assimilation", "n_members")) if cfg else None
    if cfg:
        config_summary(cfg, exp)
    current = state_summary(exp, n_members)
    recent_logs(exp, args.logs)
    recent_slurm(exp, args.slurm)
    member_rsl(exp, n_members, current)
    queue(_get(cfg, ("metadata", "name")) if cfg else None)


if __name__ == "__main__":
    sys.exit(main())
