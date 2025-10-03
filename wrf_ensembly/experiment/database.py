"""
Database operations for experiment status tracking. This module provides SQLite-based storage for experiment status.
"""

import datetime as dt
import sqlite3
import time
from pathlib import Path
from typing import List

from wrf_ensembly.console import logger

from .dataclasses import RuntimeStatistics


class DatabaseConnection:
    """
    A database connection that provides batched operations.
    This class is returned by ExperimentDatabase's context manager.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.cursor = conn.cursor()

    def get_experiment_state(self) -> tuple[int, bool, bool]:
        """Get the current experiment state."""

        self.cursor.execute("""
            SELECT current_cycle, filter_run, analysis_run
            FROM ExperimentState WHERE id = 1
        """)
        result = self.cursor.fetchone()
        if result is None:
            return (0, False, False)
        return result

    def set_experiment_state(
        self, current_cycle: int, filter_run: bool, analysis_run: bool
    ):
        """Update the experiment state."""

        self.cursor.execute(
            """
            UPDATE ExperimentState
            SET current_cycle = ?, filter_run = ?, analysis_run = ?
            WHERE id = 1
        """,
            (current_cycle, filter_run, analysis_run),
        )

    def get_all_members_status(self) -> List[tuple[int, bool]]:
        """Get the status of all members."""

        self.cursor.execute("""
            SELECT member_i, advanced FROM MemberStatus ORDER BY member_i
        """)
        return self.cursor.fetchall()

    def get_member_status(self, member_i: int) -> bool:
        """Get the advanced status for a specific member."""

        self.cursor.execute(
            """SELECT advanced FROM MemberStatus WHERE member_i = ?""",
            (member_i,),
        )
        result = self.cursor.fetchone()
        return result[0] if result else False

    def set_member_advanced(self, member_i: int, advanced: bool):
        """Set the advanced status for a specific member."""

        self.cursor.execute(
            """
            INSERT OR REPLACE INTO MemberStatus (member_i, advanced)
            VALUES (?, ?)
        """,
            (member_i, advanced),
        )

    def set_members_advanced_batch(self, member_statuses: List[tuple[int, bool]]):
        """Set the advanced status for multiple members in a single transaction."""

        self.cursor.executemany(
            """
            INSERT OR REPLACE INTO MemberStatus (member_i, advanced)
            VALUES (?, ?)
        """,
            member_statuses,
        )

    def reset_members_advanced(self):
        """Reset all members' advanced status to False."""

        self.cursor.execute("UPDATE MemberStatus SET advanced = FALSE")

    def add_runtime_statistics(
        self,
        member_i: int,
        cycle: int,
        start: dt.datetime,
        end: dt.datetime,
        duration_seconds: int,
    ):
        """Add runtime statistics for a member and cycle."""

        self.cursor.execute(
            """
            INSERT INTO RuntimeStatistics
            (member_i, cycle, start_time, end_time, duration_seconds)
            VALUES (?, ?, ?, ?, ?)
        """,
            (member_i, cycle, start.isoformat(), end.isoformat(), duration_seconds),
        )

    def get_member_runtime_statistics(self, member_i: int) -> List[RuntimeStatistics]:
        """Get all runtime statistics for a specific member."""

        self.cursor.execute(
            """
            SELECT cycle, start_time, end_time, duration_seconds
            FROM RuntimeStatistics
            WHERE member_i = ?
            ORDER BY cycle
        """,
            (member_i,),
        )

        stats = []
        for row in self.cursor.fetchall():
            cycle, start_str, end_str, duration = row
            start_time = dt.datetime.fromisoformat(start_str)
            end_time = dt.datetime.fromisoformat(end_str)

            stats.append(
                RuntimeStatistics(
                    cycle=cycle, start=start_time, end=end_time, duration_s=duration
                )
            )

        return stats

    def clear_runtime_statistics(self):
        """Clear all runtime statistics from the database."""

        self.cursor.execute("DELETE FROM RuntimeStatistics")

    def initialize_members(self, n_members: int):
        """Initialize member status for the given number of members."""

        # First, get existing members and add any missing ones
        self.cursor.execute("SELECT member_i FROM MemberStatus")
        existing_members = {row[0] for row in self.cursor.fetchall()}

        for i in range(n_members):
            if i not in existing_members:
                self.cursor.execute(
                    """
                    INSERT INTO MemberStatus (member_i, advanced)
                    VALUES (?, FALSE)
                """,
                    (i,),
                )

        # And remove any extras
        if existing_members:
            max_existing = max(existing_members)
            if max_existing >= n_members:
                self.cursor.execute(
                    """
                    DELETE FROM MemberStatus WHERE member_i >= ?
                """,
                    (n_members,),
                )
                self.cursor.execute(
                    """
                    DELETE FROM RuntimeStatistics WHERE member_i >= ?
                """,
                    (n_members,),
                )

    def reset_experiment(self):
        """Reset the experiment to its initial state."""

        self.cursor.execute("""
            UPDATE ExperimentState
            SET current_cycle = 0, filter_run = FALSE, analysis_run = FALSE
            WHERE id = 1
        """)
        self.cursor.execute("UPDATE MemberStatus SET advanced = FALSE")
        self.cursor.execute("DELETE FROM RuntimeStatistics")

    def commit(self):
        """Commit the current transaction."""

        self.conn.commit()


class ExperimentDatabase:
    """
    SQLite database for storing experiment status and runtime statistics.

    This class provides thread-safe access to the experiment database through
    context managers and proper locking.
    """

    def __init__(self, db_path: Path):
        """
        Initialize the database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path

    def __enter__(self) -> DatabaseConnection:
        """Enter context manager and return a database connection."""
        self._conn = self._get_connection_raw()
        self._init_database()
        return DatabaseConnection(self._conn)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close connection."""
        if hasattr(self, "_conn"):
            if exc_type is None:
                try:
                    self._conn.commit()
                except Exception as e:
                    logger.error(f"Failed to commit transaction: {e}")
                    self._conn.rollback()
                    raise
            else:
                self._conn.rollback()

            try:
                self._conn.close()
            except Exception:
                pass
            delattr(self, "_conn")

    def _init_database(self):
        """Initialize the database tables if they don't exist."""

        cursor = self._conn.cursor()

        # Only one row allowed in ExperimentState
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ExperimentState (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                current_cycle INTEGER NOT NULL DEFAULT 0,
                filter_run BOOLEAN NOT NULL DEFAULT FALSE,
                analysis_run BOOLEAN NOT NULL DEFAULT FALSE
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS MemberStatus (
                member_i INTEGER PRIMARY KEY,
                advanced BOOLEAN NOT NULL DEFAULT FALSE
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS RuntimeStatistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                member_i INTEGER NOT NULL,
                cycle INTEGER NOT NULL,
                start_time TEXT NOT NULL,
                end_time TEXT NOT NULL,
                duration_seconds INTEGER NOT NULL,
                FOREIGN KEY (member_i) REFERENCES MemberStatus (member_i)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runtime_member_cycle
            ON RuntimeStatistics (member_i, cycle)
        """)

        # Default initial state (cycle 0, no runs)
        cursor.execute("""
            INSERT OR IGNORE INTO ExperimentState (id, current_cycle, filter_run, analysis_run)
            VALUES (1, 0, FALSE, FALSE)
        """)

        self._conn.commit()

    def _get_connection_raw(self, max_retries=10, base_wait=0.5):
        """
        Get a database connection, includes a retry mechanism for transient filesystem errors.
        """

        conn = None
        last_error = None

        for attempt in range(max_retries):
            try:
                # For any extra attempts, add a jitter to avoid thundering herd
                if attempt > 0:
                    import random

                    jitter = random.uniform(0, 0.1 * attempt)
                    wait_time = min(base_wait * (2**attempt) + jitter, 30.0)
                    logger.debug(
                        f"DB retry {attempt}/{max_retries}, waiting {wait_time:.2f}s"
                    )
                    time.sleep(wait_time)

                conn = sqlite3.connect(
                    self.db_path,
                    timeout=60.0,  # Increased timeout
                    isolation_level="DEFERRED",  # Less aggressive locking
                )

                # Trying our best to make the database easier on the FS
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=60000")
                conn.execute("PRAGMA temp_store=MEMORY")  # Avoid temp file I/O
                conn.execute("PRAGMA foreign_keys = ON")

                # Verify connection actually works
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("SELECT 1")
                conn.commit()

                if attempt > 0:
                    logger.info(f"DB connection succeeded after {attempt} retries")
                else:
                    logger.debug("DB connection succeeded with no retries")

                return conn

            # Retry circus
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                last_error = e
                error_str = str(e).lower()

                # Retryable errors
                if any(
                    err in error_str
                    for err in [
                        "disk i/o error",
                        "database is locked",
                        "unable to open",
                        "locking protocol",
                    ]
                ):
                    if conn:
                        try:
                            conn.close()
                        except Exception:
                            pass
                        conn = None

                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Retryable DB error (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        continue

                # Non-retryable or exhausted retries
                if conn:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                logger.error(f"Database error after {attempt + 1} attempts: {e}")
                raise

            except Exception as e:
                if conn:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                logger.error(f"Unexpected database error: {e}")
                raise

        # If we get here, all retries failed
        if last_error:
            raise last_error
        else:
            raise RuntimeError("Failed to connect to database after all retries")
