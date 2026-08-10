"""
Refreshes the used_in_da / da_cycle flags in an experiment's observations.duckdb
using explicit ? parameter binding (the same approach as get_observations_for_cycle).

Uses a CTAS (CREATE TABLE AS SELECT) strategy to avoid expensive row-by-row UPDATEs
on the large observations table.

Usage:
    python scripts/refresh_da_flags.py /path/to/experiment [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd


def _build_cycle_windows_table(con, cycles, half_window):
    """
    Populates a temp table via ? parameters so DuckDB types the columns as
    TIMESTAMPTZ — the same binding path used by get_observations_for_cycle.
    """
    con.execute(
        """
        CREATE TEMP TABLE cycle_windows (
            cycle_index  INT,
            window_start TIMESTAMPTZ,
            window_end   TIMESTAMPTZ
        )
        """
    )
    for cycle in cycles:
        con.execute(
            "INSERT INTO cycle_windows VALUES (?, ?, ?)",
            [cycle.index, cycle.end - half_window, cycle.end + half_window],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_path", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be updated without writing anything",
    )
    args = parser.parse_args()

    experiment_path = args.experiment_path.resolve()
    config_path = experiment_path / "config.toml"
    db_path = experiment_path / "observations.duckdb"

    if not config_path.exists():
        print(f"ERROR: config.toml not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    if not db_path.exists():
        print(f"ERROR: observations.duckdb not found at {db_path}", file=sys.stderr)
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from wrf_ensembly.config import read_config
    from wrf_ensembly.cycling import get_cycle_information

    cfg = read_config(config_path)
    cycles = get_cycle_information(cfg)
    half_window = pd.Timedelta(minutes=cfg.assimilation.half_window_length_minutes)
    da_instruments = cfg.observations.instruments_to_assimilate

    print(f"Experiment : {experiment_path}")
    print(f"DB         : {db_path}")
    print(f"Cycles     : {len(cycles)}")
    print(f"Half-window: {half_window}")
    print(f"Instruments: {da_instruments if da_instruments else '(all)'}")
    print()

    con = duckdb.connect(database=str(db_path), read_only=args.dry_run)
    con.execute("SET TimeZone='UTC';")
    _build_cycle_windows_table(con, cycles, half_window)

    # Instrument filter clause (instrument names come from trusted config)
    if da_instruments is not None:
        instrument_filter = (
            "AND o.instrument IN ("
            + ", ".join(f"'{i}'" for i in da_instruments)
            + ")"
        )
    else:
        instrument_filter = ""

    if args.dry_run:
        print("Current state (used_in_da=TRUE rows per cycle):")
        rows = con.execute(
            """
            SELECT da_cycle, COUNT(*) AS n
            FROM observations
            WHERE used_in_da = TRUE
            GROUP BY da_cycle
            ORDER BY da_cycle
            """
        ).fetchall()
        for r in rows:
            print(f"  cycle {r[0]}: {r[1]} observations")

        print()
        print("Predicted new state after refresh:")
        rows = con.execute(
            f"""
            SELECT cw.cycle_index, cw.window_start, cw.window_end, COUNT(o.rowid) AS n
            FROM cycle_windows cw
            LEFT JOIN observations o
              ON o.time >= cw.window_start
             AND o.time <= cw.window_end
             {instrument_filter}
            GROUP BY cw.cycle_index, cw.window_start, cw.window_end
            ORDER BY cw.cycle_index
            """
        ).fetchall()
        for r in rows:
            print(f"  cycle {r[0]}: {r[3]} observations  (window {r[1]} – {r[2]})")
        con.close()
        return

    # CTAS: single scan of observations with a LATERAL correlated lookup into the
    # small cycle_windows table.  Avoids a rowid-based double-scan (rowid stability
    # across two independent scans of the same table is not guaranteed by DuckDB).
    print("Building replacement table (CTAS)...")
    con.execute(
        f"""
        CREATE TABLE observations_new AS
        SELECT
            o.* EXCLUDE (used_in_da, da_cycle),
            cw.cycle_index IS NOT NULL AS used_in_da,
            cw.cycle_index             AS da_cycle
        FROM observations o
        LEFT JOIN LATERAL (
            SELECT MIN(cycle_index) AS cycle_index
            FROM cycle_windows
            WHERE o.time >= window_start
              AND o.time <= window_end
              {instrument_filter}
        ) cw ON TRUE
        """
    )

    print("Swapping tables...")
    con.execute("DROP TABLE observations")
    con.execute("ALTER TABLE observations_new RENAME TO observations")

    print("Recreating indexes...")
    con.execute("CREATE INDEX idx_obs_time ON observations (time)")
    con.execute("CREATE INDEX idx_obs_filename ON observations (orig_filename)")
    con.execute("CREATE INDEX idx_obs_instrument_time ON observations (instrument, time)")
    con.execute("CREATE INDEX idx_obs_da ON observations (da_cycle, used_in_da)")

    total_marked = con.execute(
        "SELECT COUNT(*) FROM observations WHERE used_in_da = TRUE"
    ).fetchone()[0]

    per_cycle = con.execute(
        """
        SELECT da_cycle, COUNT(*) AS n
        FROM observations
        WHERE used_in_da = TRUE
        GROUP BY da_cycle
        ORDER BY da_cycle
        """
    ).fetchall()
    for r in per_cycle:
        print(f"  cycle {r[0]}: {r[1]} observations")

    con.close()
    print(f"\nDone. Total observations marked used_in_da=TRUE: {total_marked}")


if __name__ == "__main__":
    main()
