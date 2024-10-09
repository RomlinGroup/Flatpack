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
        logger.info("Executing query: %s", query)
        logger.info("Query parameters: %s", params)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                affected_rows = cursor.rowcount
                conn.commit()
                logger.info("Query executed successfully. Affected rows: %s", affected_rows)
        except Error as e:
            logger.error("Database error in _execute_query: %s", e)
            logger.error("Failed query: %s", query)
            logger.error("Failed query parameters: %s", params)
            raise

    def _fetch_all(self, query: str, params: tuple = ()) -> List[tuple]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchall()
        except Error as e:
            logger.error("Database error: %s", e)
            raise

    def _fetch_one(self, query: str, params: tuple = ()) -> Optional[tuple]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.fetchone()
        except Error as e:
            logger.error("Database error: %s", e)
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
                hook_placement TEXT NOT NULL,
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

    def delete_comment(self, comment_id: int) -> bool:
        logger.info("Attempting to delete comment with ID: %s", comment_id)
        logger.info("Type of comment_id: %s", type(comment_id))

        query = "DELETE FROM flatpack_comments WHERE id = ?"
        try:
            self._execute_query(query, (comment_id,))  # Note the comma to make it a tuple
            logger.info("Comment with ID %s deleted successfully", comment_id)
            return True
        except Exception as e:
            logger.error("Error deleting comment with ID %s: %s", comment_id, str(e))
            raise

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

    # Hook operations
    def add_hook(self, hook_name: str, hook_placement: str, hook_script: str, hook_type: str) -> int:
        query = """
        INSERT INTO flatpack_hooks (hook_name, hook_placement, hook_script, hook_type)
        VALUES (?, ?, ?, ?)
        """
        self._execute_query(query, (hook_name, hook_placement, hook_script, hook_type))
        return self._fetch_one("SELECT last_insert_rowid()")[0]

    def delete_hook(self, hook_id: int) -> bool:
        logger.info("Attempting to delete hook with ID: %s", hook_id)
        logger.info("Type of hook_id: %s", type(hook_id))

        query = "DELETE FROM flatpack_hooks WHERE id = ?"
        try:
            self._execute_query(query, (hook_id,))
            logger.info("Hook with ID %s deleted successfully", hook_id)
            return True
        except Exception as e:
            logger.error("Error deleting hook with ID %s: %s", hook_id, str(e))
            raise

    def get_all_hooks(self) -> List[Dict[str, Any]]:
        query = """
        SELECT id, hook_name, hook_placement, hook_script, hook_type, created_at
        FROM flatpack_hooks
        ORDER BY created_at DESC
        """
        results = self._fetch_all(query)
        return [
            {
                "id": row[0],
                "hook_name": row[1],
                "hook_placement": row[2],
                "hook_script": row[3],
                "hook_type": row[4],
                "created_at": row[5]
            }
            for row in results
        ]

    def get_hook_by_name(self, hook_name: str) -> Optional[Dict[str, Any]]:
        query = """
        SELECT id, hook_name, hook_placement, hook_script, hook_type, created_at
        FROM flatpack_hooks
        WHERE hook_name = ?
        """
        result = self._fetch_one(query, (hook_name,))
        if result:
            return {
                "id": result[0],
                "hook_name": result[1],
                "hook_placement": result[2],
                "hook_script": result[3],
                "hook_type": result[4],
                "created_at": result[5]
            }
        return None

    def hook_exists(self, hook_name: str) -> bool:
        query = "SELECT COUNT(*) FROM flatpack_hooks WHERE hook_name = ?"
        result = self._fetch_one(query, (hook_name,))
        return result[0] > 0 if result else False

    def update_hook(self, hook_id: int, hook_name: str, hook_placement: str, hook_script: str, hook_type: str) -> bool:
        query = """
        UPDATE flatpack_hooks 
        SET hook_name = ?, hook_placement = ?, hook_script = ?, hook_type = ?
        WHERE id = ?
        """
        try:
            self._execute_query(query, (hook_name, hook_placement, hook_script, hook_type, hook_id))
            return True
        except Exception as e:
            logger.error("Error updating hook with ID %s: %s", hook_id, str(e))
            return False

    # Schedule operations
    def add_schedule(self, schedule_type: str, pattern: Optional[str], datetimes: Optional[List[str]]) -> int:
        query = """
        INSERT INTO flatpack_schedule (type, pattern, datetimes)
        VALUES (?, ?, ?)
        """
        datetimes_json = json.dumps(datetimes) if datetimes else None
        self._execute_query(query, (schedule_type, pattern, datetimes_json))
        return self._fetch_one("SELECT last_insert_rowid()")[0]

    def delete_schedule(self, schedule_id: int) -> bool:
        logger.info("Attempting to delete schedule with ID: %s", schedule_id)
        logger.info("Type of schedule_id: %s", type(schedule_id))

        query = "DELETE FROM flatpack_schedule WHERE id = ?"
        try:
            self._execute_query(query, (schedule_id,))
            logger.info("Schedule with ID %s deleted successfully", schedule_id)
            return True
        except Exception as e:
            logger.error("Error deleting schedule with ID %s: %s", schedule_id, str(e))
            raise

    def delete_schedule_datetime(self, schedule_id: int, datetime_index: int) -> bool:
        logger.info("Attempting to delete datetime from schedule. Schedule ID: %s, Datetime index: %s", schedule_id,
                    datetime_index)
        logger.info("Type of schedule_id: %s, Type of datetime_index: %s", type(schedule_id), type(datetime_index))

        try:
            schedule = self._fetch_one("SELECT type, datetimes FROM flatpack_schedule WHERE id = ?", (schedule_id,))
            if not schedule or schedule[0] != 'manual':
                logger.warning("Schedule not found or not of type 'manual'. Schedule ID: %s", schedule_id)
                return False

            datetimes = json.loads(schedule[1])
            if 0 <= datetime_index < len(datetimes):
                del datetimes[datetime_index]
                query = "UPDATE flatpack_schedule SET datetimes = ? WHERE id = ?"
                self._execute_query(query, (json.dumps(datetimes), schedule_id))
                logger.info("Datetime at index %s deleted from schedule %s", datetime_index, schedule_id)
                return True
            logger.warning("Invalid datetime index %s for schedule %s", datetime_index, schedule_id)
            return False
        except Exception as e:
            logger.error("Error deleting datetime from schedule. Schedule ID: %s, Datetime index: %s. Error: %s",
                         schedule_id, datetime_index, str(e))
            raise

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

    # Metadata operations
    def delete_metadata(self, key: str) -> bool:
        query = "DELETE FROM flatpack_metadata WHERE key = ?"
        self._execute_query(query, (key))
        return True

    def get_metadata(self, key: str) -> Optional[str]:
        query = "SELECT value FROM flatpack_metadata WHERE key = ?"
        result = self._fetch_one(query, (key))
        return result[0] if result else None

    def set_metadata(self, key: str, value: str) -> bool:
        query = """
        INSERT OR REPLACE INTO flatpack_metadata (key, value)
        VALUES (?, ?)
        """
        self._execute_query(query, (key, value))
        return True
