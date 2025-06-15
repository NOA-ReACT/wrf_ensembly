"""
Database operations for experiment status tracking. This module provides SQLite-based storage for experiment status.
"""

import datetime as dt
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from wrf_ensembly.console import logger

from .dataclasses import RuntimeStatistics


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
        self._init_database()

    def _init_database(self):
        """Initialize the database tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency

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

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get a database connection"""

        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def get_experiment_state(self) -> tuple[int, bool, bool]:
        """
        Get the current experiment state.

        Returns:
            A tuple containing:
                - current_cycle: The current cycle number
                - filter_run: Whether the filter has been run for the current cycle
                - analysis_run: Whether the analysis has been run for the current cycle
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT current_cycle, filter_run, analysis_run
                FROM ExperimentState WHERE id = 1
            """)
            result = cursor.fetchone()
            if result is None:
                # This shouldn't happen if _init_database worked correctly
                return (0, False, False)
            return result

    def set_experiment_state(
        self, current_cycle: int, filter_run: bool, analysis_run: bool
    ):
        """
        Update the experiment state.

        Args:
            current_cycle: The current cycle number
            filter_run: Whether the filter has been run for the current cycle
            analysis_run: Whether the analysis has been run for the current cycle
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE ExperimentState
                SET current_cycle = ?, filter_run = ?, analysis_run = ?
                WHERE id = 1
            """,
                (current_cycle, filter_run, analysis_run),
            )
            conn.commit()

    def get_member_status(self, member_i: int) -> bool:
        """
        Get the advanced status for a specific member, given the index.

        Returns:
            True if the member has been advanced for the current cycle
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT advanced FROM MemberStatus WHERE member_i = ?
            """,
                (member_i,),
            )
            result = cursor.fetchone()
            return result[0] if result else False

    def set_member_advanced(self, member_i: int, advanced: bool):
        """
        Set the advanced status for a specific member.
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO MemberStatus (member_i, advanced)
                VALUES (?, ?)
            """,
                (member_i, advanced),
            )
            conn.commit()

    def get_all_members_status(self) -> List[tuple[int, bool]]:
        """
        Get the status of all members.

        Returns:
            List of tuples (member_i, advanced)
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT member_i, advanced FROM MemberStatus ORDER BY member_i
            """)
            return cursor.fetchall()

    def initialize_members(self, n_members: int):
        """
        Initialize member status for the given number of members.

        Args:
            n_members: Number of ensemble members
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # First, get existing members and add any missing ones
            cursor.execute("SELECT member_i FROM MemberStatus")
            existing_members = {row[0] for row in cursor.fetchall()}

            for i in range(n_members):
                if i not in existing_members:
                    cursor.execute(
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
                    cursor.execute(
                        """
                        DELETE FROM MemberStatus WHERE member_i >= ?
                    """,
                        (n_members,),
                    )
                    cursor.execute(
                        """
                        DELETE FROM RuntimeStatistics WHERE member_i >= ?
                    """,
                        (n_members,),
                    )

            conn.commit()

    def reset_members_advanced(self):
        """Reset all members' advanced status to False."""
        with self._get_connection() as conn:
            conn.execute("UPDATE MemberStatus SET advanced = FALSE")
            conn.commit()

    def add_runtime_statistics(
        self,
        member_i: int,
        cycle: int,
        start: dt.datetime,
        end: dt.datetime,
        duration_seconds: int,
    ):
        """
        Add runtime statistics for a member and cycle.

        Args:
            member_i: Member index
            cycle: Cycle number
            start: Start time
            end: End time
            duration_seconds: Duration in seconds
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO RuntimeStatistics
                (member_i, cycle, start_time, end_time, duration_seconds)
                VALUES (?, ?, ?, ?, ?)
            """,
                (member_i, cycle, start.isoformat(), end.isoformat(), duration_seconds),
            )
            conn.commit()

    def get_runtime_statistics(
        self, member_i: Optional[int] = None, cycle: Optional[int] = None
    ) -> List[RuntimeStatistics]:
        """
        Get runtime statistics, optionally filtered by member and/or cycle.

        Args:
            member_i: Optional member index filter
            cycle: Optional cycle filter

        Returns:
            List of RuntimeStatistics objects
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT member_i, cycle, start_time, end_time, duration_seconds
                FROM RuntimeStatistics
            """
            params = []

            conditions = []
            if member_i is not None:
                conditions.append("member_i = ?")
                params.append(member_i)
            if cycle is not None:
                conditions.append("cycle = ?")
                params.append(cycle)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY member_i, cycle"

            cursor.execute(query, params)
            results = []

            for row in cursor.fetchall():
                results.append(
                    RuntimeStatistics(
                        cycle=row[1],
                        start=dt.datetime.fromisoformat(row[2]),
                        end=dt.datetime.fromisoformat(row[3]),
                        duration_s=row[4],
                    )
                )

            return results

    def clear_runtime_statistics(
        self, member_i: Optional[int] = None, cycle: Optional[int] = None
    ):
        """
        Clear runtime statistics, optionally filtered by member and/or cycle.

        Args:
            member_i: Optional member filter
            cycle: Optional cycle filter
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "DELETE FROM RuntimeStatistics"
            params = []

            conditions = []
            if member_i is not None:
                conditions.append("member_i = ?")
                params.append(member_i)
            if cycle is not None:
                conditions.append("cycle = ?")
                params.append(cycle)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            cursor.execute(query, params)
            conn.commit()

    def get_member_runtime_statistics(self, member_i: int) -> List[RuntimeStatistics]:
        """
        Get all runtime statistics for a specific member by index.

        Returns:
            List of RuntimeStatistics for the member
        """

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT cycle, start_time, end_time, duration_seconds
                FROM RuntimeStatistics
                WHERE member_i = ?
                ORDER BY cycle
            """,
                (member_i,),
            )

            stats = []
            for row in cursor.fetchall():
                cycle, start_str, end_str, duration = row
                start_time = dt.datetime.fromisoformat(start_str)
                end_time = dt.datetime.fromisoformat(end_str)

                stats.append(
                    RuntimeStatistics(
                        cycle=cycle, start=start_time, end=end_time, duration_s=duration
                    )
                )

            return stats

    def reset_experiment(self):
        """Reset the experiment to its initial state."""

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE ExperimentState
                SET current_cycle = 0, filter_run = FALSE, analysis_run = FALSE
                WHERE id = 1
            """)
            cursor.execute("UPDATE MemberStatus SET advanced = FALSE")
            cursor.execute("DELETE FROM RuntimeStatistics")

            conn.commit()
