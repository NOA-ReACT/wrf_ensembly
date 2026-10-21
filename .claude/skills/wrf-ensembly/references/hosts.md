# Hosts: what you may run where, and how

Run `hostname` and read the matching section. If none matches, use **Unknown host**.
The user's current limits (e.g. fewer cores while they run something of their own) override
what is written here.

## Iridium (dev VM)

Single workstation: i9-14900K under WSL2, 117 GiB RAM. The user's own machine for small and
medium experiments.

**Permissions.** Running `advance-member`, `filter`, `postprocess`, observation conversion etc.
locally is fine to just do. Submitting to its SLURM is fine too (single node, nobody else's
allocation). Still ask before anything that deletes outputs (see the guardrails in SKILL.md).

**Cores.** Use at most 24 (`--cores`, `--jobs`, ranks summed over everything you run at once).
The CPU has 8 P-cores with hyperthreading and 16 E-cores: 24 cores, 32 threads. The extra 8
threads don't help WRF. Members slow down 35–50% once more than 8 cores are busy (the P-cores
fill up), but packing several members still wins over one wide member.

**MPI.** WSL presents the CPU as 16 cores x 2 threads, so Open MPI offers only 16 slots:
`--cores 24` fails with "not enough slots" unless `slurm.mpirun_command` (in the env file) has
`--use-hwthread-cpus`. Use `--bind-to none`; binding made no measurable difference.

**Memory.** One member of the 27 km 280x150x60 WRF-Chem (`chem_opt=300`) domain needs 20–23 GB
whatever its rank count. Memory, not cores, limits how many members run at once: 3 is
comfortable, 4 fits, 5 is at the limit.

**Recommended layout** (benchmark 2026-10-16, fixed dt=144 s; model hours per wall-clock hour):

| layout (members x ranks) | s/step per member | throughput | peak memory |
|---|---|---|---|
| 1 x 16 | 7.4 | 19.4 | ~23 GB |
| 1 x 24 (`--use-hwthread-cpus`) | 6.3 | 23.0 | ~23 GB |
| 2 x 12 | 11.3 | 25.6 | 50 GB |
| **3 x 8** | 15.8 | **27.4** | 72 GB |
| 4 x 6 | 20.9 | 27.5 | 91 GB |

- One member (e.g. re-running a crashed one): `--cores 16`, or 24 with the flag above.
- Several members: 3 at once with `--cores 8` each. Separate `advance-member` processes are
  safe to run side by side (each member has its own status file).
- SLURM: `advance_model` with 8 tasks and 32G gives 3 x 8, since only 3 x 32G fit in the
  node's `RealMemory=120000`. That relies on SLURM treating memory as a consumable resource;
  check `scontrol show config | grep SelectTypeParameters` before relying on it.
- Rough cost: 30 members take ~26 h per simulated day at 3 x 8 (less with adaptive dt).

**Running outside SLURM** needs what the jobfiles normally set up, in the same shell command
as `wrf-ensembly`: `ulimit -s unlimited` (otherwise wrf.exe segfaults right after reading
wrfbdy) and the config's `slurm.pre_commands` (spack env) for MPI and netCDF. Don't activate
the repo `.venv` *after* the spack env: venv's `deactivate` restores the pre-spack PATH and
`mpirun` disappears. Call `<repo>/.venv/bin/wrf-ensembly` directly instead.

## The big external HPC

The user runs the very large experiments there by hand. If a path or hostname suggests that
machine, only inspect, don't act, and don't record how to reach it.

## Unknown host

Assume it is shared: ask before running anything heavier than a quick inspection, and ask
how many cores and how much memory you may use. On a cluster, never run WRF, DART or
postprocessing on a login node, and confirm before submitting SLURM jobs, since they consume a
shared allocation. Read the experiment's `env_config.toml` for the scheduler directives and
environment it expects. Per member, plan for roughly the memory listed above for a similar
domain size.
