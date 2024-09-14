import json
import logging
import os
import sqlite3

from datetime import datetime
from sqlite3 import Error
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _execute_query(self, query: str, params: tuple = ()) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                conn.commit()
        except Error as e:
            logger.error(f"Database error: {e}")
            raise

    def _fetch_all(self, query: str, params: tuple = ()) -> List[tuple]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()
        except Error as e:
            logger.error(f"Database error: {e}")
            raise

    def _fetch_one(self, query: str, params: tuple = ()) -> Optional[tuple]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchone()
        except Error as e:
            logger.error(f"Database error: {e}")
            raise

    def initialize_database(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        queries = [
            """
            CREATE TABLE IF NOT EXISTS flatpack_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_id TEXT NOT NULL,
                selected_text TEXT NOT NULL,
                comment TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS flatpack_hooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hook_name TEXT NOT NULL,
                hook_script TEXT NOT NULL,
                hook_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS flatpack_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS flatpack_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                pattern TEXT,
                datetimes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_run TIMESTAMP
            )
            """
        ]

        for query in queries:
            cursor.execute(query)

        conn.commit()
        conn.close()

    # Comment operations
    def add_comment(self, block_id: str, selected_text: str, comment: str) -> int:
        query = """
        INSERT INTO flatpack_comments (block_id, selected_text, comment)
        VALUES (?, ?, ?)
        """
        self._execute_query(query, (block_id, selected_text, comment))
        return self._fetch_one("SELECT last_insert_rowid()")[0]

    def get_all_comments(self) -> List[Dict[str, Any]]:
        query = """
        SELECT id, block_id, selected_text, comment, created_at
        FROM flatpack_comments
        ORDER BY created_at DESC
        """
        results = self._fetch_all(query)
        return [
            {
                "id": row[0],
                "block_id": row[1],
                "selected_text": row[2],
                "comment": row[3],
                "created_at": row[4]
            }
            for row in results
        ]

    def delete_comment(self, comment_id: int) -> bool:
        query = "DELETE FROM flatpack_comments WHERE id = ?"
        self._execute_query(query, (comment_id,))
        return True

    # Hook operations
    def add_hook(self, hook_name: str, hook_script: str, hook_type: str) -> int:
        query = """
        INSERT INTO flatpack_hooks (hook_name, hook_script, hook_type)
        VALUES (?, ?, ?)
        """
        self._execute_query(query, (hook_name, hook_script, hook_type))
        return self._fetch_one("SELECT last_insert_rowid()")[0]

    def get_all_hooks(self) -> List[Dict[str, Any]]:
        query = """
        SELECT id, hook_name, hook_script, hook_type, created_at
        FROM flatpack_hooks
        ORDER BY created_at DESC
        """
        results = self._fetch_all(query)
        return [
            {
                "id": row[0],
                "hook_name": row[1],
                "hook_script": row[2],
                "hook_type": row[3],
                "created_at": row[4]
            }
            for row in results
        ]

    def delete_hook(self, hook_id: int) -> bool:
        query = "DELETE FROM flatpack_hooks WHERE id = ?"
        self._execute_query(query, (hook_id,))
        return True

    # Schedule operations
    def add_schedule(self, schedule_type: str, pattern: Optional[str], datetimes: Optional[List[str]]) -> int:
        query = """
        INSERT INTO flatpack_schedule (type, pattern, datetimes)
        VALUES (?, ?, ?)
        """
        datetimes_json = json.dumps(datetimes) if datetimes else None
        self._execute_query(query, (schedule_type, pattern, datetimes_json))
        return self._fetch_one("SELECT last_insert_rowid()")[0]

    def get_all_schedules(self) -> List[Dict[str, Any]]:
        query = """
        SELECT id, type, pattern, datetimes, created_at, last_run
        FROM flatpack_schedule
        ORDER BY created_at DESC
        """
        results = self._fetch_all(query)
        return [
            {
                "id": row[0],
                "type": row[1],
                "pattern": row[2],
                "datetimes": json.loads(row[3]) if row[3] else None,
                "created_at": row[4],
                "last_run": row[5]
            }
            for row in results
        ]

    def update_schedule_last_run(self, schedule_id: int, last_run: datetime) -> bool:
        query = "UPDATE flatpack_schedule SET last_run = ? WHERE id = ?"
        self._execute_query(query, (last_run.isoformat(), schedule_id))
        return True

    def delete_schedule(self, schedule_id: int) -> bool:
        query = "DELETE FROM flatpack_schedule WHERE id = ?"
        self._execute_query(query, (schedule_id,))
        return True

    def delete_schedule_datetime(self, schedule_id: int, datetime_index: int) -> bool:
        schedule = self._fetch_one("SELECT type, datetimes FROM flatpack_schedule WHERE id = ?", (schedule_id,))
        if not schedule or schedule[0] != 'manual':
            return False

        datetimes = json.loads(schedule[1])
        if 0 <= datetime_index < len(datetimes):
            del datetimes[datetime_index]
            query = "UPDATE flatpack_schedule SET datetimes = ? WHERE id = ?"
            self._execute_query(query, (json.dumps(datetimes), schedule_id))
            return True
        return False

    # Metadata operations
    def set_metadata(self, key: str, value: str) -> bool:
        query = """
        INSERT OR REPLACE INTO flatpack_metadata (key, value)
        VALUES (?, ?)
        """
        self._execute_query(query, (key, value))
        return True

    def get_metadata(self, key: str) -> Optional[str]:
        query = "SELECT value FROM flatpack_metadata WHERE key = ?"
        result = self._fetch_one(query, (key,))
        return result[0] if result else None

    def delete_metadata(self, key: str) -> bool:
        query = "DELETE FROM flatpack_metadata WHERE key = ?"
        self._execute_query(query, (key,))
        return True
