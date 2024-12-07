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
                show_on_frontpage BOOLEAN DEFAULT 0,
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
            """,
            """
            CREATE TABLE IF NOT EXISTS flatpack_source_hook_mappings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                target_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_id, target_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS flatpack_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    def add_hook(self, hook_name: str, hook_placement: str, hook_script: str, hook_type: str,
                 show_on_frontpage: bool = False) -> int:
        query = """
        INSERT INTO flatpack_hooks (hook_name, hook_placement, hook_script, hook_type, show_on_frontpage)
        VALUES (?, ?, ?, ?, ?)
        """
        self._execute_query(query, (hook_name, hook_placement, hook_script, hook_type, int(show_on_frontpage)))
        return self._fetch_one("SELECT last_insert_rowid()")[0]

    def delete_hook(self, hook_id: int) -> bool:
        """Delete a hook and its related connections.

        Args:
            hook_id (int): The ID of the hook to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        logger.info("Attempting to delete hook with ID: %s", hook_id)
        logger.info("Type of hook_id: %s", type(hook_id))

        try:
            hook_name = self.get_hook_name_by_id(hook_id)

            if not hook_name:
                logger.error("Hook with ID %s not found", hook_id)
                return False

            base_hook_name = hook_name.split('-')[0]
            self.delete_mappings_by_target(base_hook_name)

            query = "DELETE FROM flatpack_hooks WHERE id = ?"
            self._execute_query(query, (hook_id,))

            remaining_mappings = self.get_all_source_hook_mappings()
            connections_data = {
                "connections": [
                    {
                        "source_id": mapping["source_id"],
                        "target_id": mapping["target_id"],
                        "source_type": mapping["source_type"],
                        "target_type": mapping["target_type"]
                    }
                    for mapping in remaining_mappings
                ]
            }

            connections_file = os.path.join(os.path.dirname(self.db_path), 'connections.json')
            os.makedirs(os.path.dirname(connections_file), exist_ok=True)

            with open(connections_file, 'w') as f:
                json.dump(connections_data, f, indent=4)

            remaining_hooks = self.get_all_hooks()

            hooks_data = {
                "hooks": [
                    {
                        "hook_id": hook["id"],
                        "hook_name": hook["hook_name"],
                        "hook_placement": hook["hook_placement"],
                        "hook_script": hook["hook_script"],
                        "hook_type": hook["hook_type"],
                        "show_on_frontpage": hook["show_on_frontpage"]
                    }
                    for hook in remaining_hooks
                ]
            }

            hooks_file = os.path.join(os.path.dirname(self.db_path), 'hooks.json')
            os.makedirs(os.path.dirname(hooks_file), exist_ok=True)

            with open(hooks_file, 'w') as f:
                json.dump(hooks_data, f, indent=4)

            logger.info("Hook with ID %s, related connections, and hooks.json updated successfully", hook_id)
            return True

        except Exception as e:
            logger.error("Error deleting hook with ID %s: %s", hook_id, str(e))
            raise

    def delete_mappings_by_target(self, target_name: str) -> bool:
        logger.info("Attempting to delete source-hook mappings with target name: %s", target_name)
        logger.info("Type of target_name: %s", type(target_name))

        query = "DELETE FROM flatpack_source_hook_mappings WHERE target_id LIKE ?"
        try:
            self._execute_query(query, (f"{target_name}-%",))
            logger.info("Source-hook mappings with target name %s deleted successfully", target_name)
            return True
        except Exception as e:
            logger.error("Error deleting source-hook mappings with target name %s: %s", target_name, str(e))
            raise

    def get_all_hooks(self) -> List[Dict[str, Any]]:
        query = """
        SELECT id, hook_name, hook_placement, hook_script, hook_type, show_on_frontpage, created_at
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
                "show_on_frontpage": bool(row[5]),
                "created_at": row[6]
            }
            for row in results
        ]

    def get_hook_by_name(self, hook_name: str) -> Optional[Dict[str, Any]]:
        query = """
        SELECT id, hook_name, hook_placement, hook_script, hook_type, show_on_frontpage, created_at
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
                "show_on_frontpage": bool(result[5]),
                "created_at": result[6]
            }
        return None

    def get_hook_name_by_id(self, hook_id: int) -> str:
        query = "SELECT hook_name FROM flatpack_hooks WHERE id = ?"
        result = self._fetch_one(query, (hook_id,))
        if result:
            return result[0]
        return ""

    def hook_exists(self, hook_name: str) -> bool:
        query = "SELECT COUNT(*) FROM flatpack_hooks WHERE hook_name = ?"
        result = self._fetch_one(query, (hook_name,))
        return result[0] > 0 if result else False

    def update_hook(self, hook_id: int, hook_name: str, hook_placement: str, hook_script: str, hook_type: str,
                    show_on_frontpage: bool = False) -> bool:
        query = """
        UPDATE flatpack_hooks 
        SET hook_name = ?, hook_placement = ?, hook_script = ?, hook_type = ?, show_on_frontpage = ?
        WHERE id = ?
        """
        try:
            self._execute_query(query,
                                (hook_name, hook_placement, hook_script, hook_type, int(show_on_frontpage), hook_id))
            return True
        except Exception as e:
            logger.error("Error updating hook with ID %s: %s", hook_id, str(e))
            return False

    # Metadata operations
    def delete_metadata(self, key: str) -> bool:
        query = "DELETE FROM flatpack_metadata WHERE key = ?"
        self._execute_query(query, (key,))
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

    # Source-hook mapping operations
    def add_source_hook_mapping(self, source_id: str, target_id: str, source_type: str, target_type: str) -> int:
        logger.info(
            "Adding source-hook mapping: source_id=%s, target_id=%s, source_type=%s, target_type=%s",
            source_id, target_id, source_type, target_type
        )

        query = """
        INSERT OR REPLACE INTO flatpack_source_hook_mappings 
        (source_id, target_id, source_type, target_type)
        VALUES (?, ?, ?, ?)
        """

        try:
            self._execute_query(query, (source_id, target_id, source_type, target_type))
            mapping_id = self._fetch_one("SELECT last_insert_rowid()")[0]
            logger.info("Successfully added/updated source-hook mapping with ID: %s", mapping_id)
            return mapping_id
        except Exception as e:
            logger.error("Error adding source-hook mapping: %s", e)
            raise

    def delete_all_source_hook_mappings(self) -> bool:
        logger.info("Attempting to delete all source-hook mappings")
        query = "DELETE FROM flatpack_source_hook_mappings"
        try:
            self._execute_query(query)
            logger.info("Successfully deleted all source-hook mappings")
            return True
        except Exception as e:
            logger.error("Error deleting all source-hook mappings: %s", e)
            return False

    def get_all_source_hook_mappings(self) -> List[Dict[str, Any]]:
        logger.info("Retrieving all source-hook mappings")
        query = """
        SELECT id, source_id, target_id, source_type, target_type, created_at
        FROM flatpack_source_hook_mappings
        ORDER BY created_at DESC
        """
        try:
            results = self._fetch_all(query)
            mappings = [
                {
                    "id": row[0],
                    "source_id": row[1],
                    "target_id": row[2],
                    "source_type": row[3],
                    "target_type": row[4],
                    "created_at": row[5]
                }
                for row in results
            ]
            logger.info("Successfully retrieved %d source-hook mappings", len(mappings))
            return mappings
        except Exception as e:
            logger.error("Error retrieving source-hook mappings: %s", e)
            raise

    def delete_source_hook_mapping(self, mapping_id: int) -> bool:
        """Delete a specific source-hook mapping by its ID.

        Args:
            mapping_id (int): The ID of the mapping to delete.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        logger.info("Attempting to delete source-hook mapping with ID: %s", mapping_id)

        query = "DELETE FROM flatpack_source_hook_mappings WHERE id = ?"
        try:
            self._execute_query(query, (mapping_id,))
            logger.info("Successfully deleted source-hook mapping with ID: %s", mapping_id)
            return True
        except Exception as e:
            logger.error("Error deleting source-hook mapping with ID %s: %s", mapping_id, e)
            return False

    def get_source_hook_mapping(self, mapping_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific source-hook mapping by its ID.

        Args:
            mapping_id (int): The ID of the mapping to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The mapping data if found, None otherwise.
        """
        query = """
        SELECT id, source_id, target_id, source_type, target_type, created_at
        FROM flatpack_source_hook_mappings
        WHERE id = ?
        """
        try:
            result = self._fetch_one(query, (mapping_id,))
            if result:
                return {
                    "id": result[0],
                    "source_id": result[1],
                    "target_id": result[2],
                    "source_type": result[3],
                    "target_type": result[4],
                    "created_at": result[5]
                }
            return None
        except Exception as e:
            logger.error("Error retrieving source-hook mapping with ID %s: %s", mapping_id, e)
            raise

    # Sources
    def add_source(self, source_name: str, source_type: str, source_details: Optional[Dict[str, Any]] = None) -> int:
        query = """
        INSERT INTO flatpack_sources (source_name, source_type, source_details)
        VALUES (?, ?, ?)
        """
        source_details_json = json.dumps(source_details) if source_details else None
        self._execute_query(query, (source_name, source_type, source_details_json))
        source_id = self._fetch_one("SELECT last_insert_rowid()")[0]

        self._sync_sources_to_file()

        return source_id

    def get_all_sources(self) -> List[Dict[str, Any]]:
        query = """
        SELECT id, source_name, source_type, source_details, created_at
        FROM flatpack_sources
        ORDER BY created_at DESC
        """
        results = self._fetch_all(query)
        return [
            {
                "id": row[0],
                "source_name": row[1],
                "source_type": row[2],
                "source_details": json.loads(row[3]) if row[3] else None,
                "created_at": row[4]
            }
            for row in results
        ]

    def get_source_by_id(self, source_id: int) -> Optional[Dict[str, Any]]:
        query = """
        SELECT id, source_name, source_type, source_details, created_at
        FROM flatpack_sources
        WHERE id = ?
        """
        result = self._fetch_one(query, (source_id,))
        if result:
            return {
                "id": result[0],
                "source_name": result[1],
                "source_type": result[2],
                "source_details": json.loads(result[3]) if result[3] else None,
                "created_at": result[4]
            }
        return None

    def delete_source(self, source_id: int) -> bool:
        logger.info("Attempting to delete source with ID: %s", source_id)

        try:
            source = self.get_source_by_id(source_id)

            if not source:
                logger.error("Source with ID %s not found", source_id)
                return False

            source_name = source["source_name"].lower()
            query_delete_mappings = "DELETE FROM flatpack_source_hook_mappings WHERE source_id = ?"
            self._execute_query(query_delete_mappings, (source_name,))

            query_delete_source = "DELETE FROM flatpack_sources WHERE id = ?"
            self._execute_query(query_delete_source, (source_id,))

            remaining_mappings = self.get_all_source_hook_mappings()
            connections_data = {
                "connections": [
                    {
                        "source_id": mapping["source_id"],
                        "target_id": mapping["target_id"],
                        "source_type": mapping["source_type"],
                        "target_type": mapping["target_type"]
                    }
                    for mapping in remaining_mappings
                ]
            }

            connections_file = os.path.join(os.path.dirname(self.db_path), 'connections.json')
            os.makedirs(os.path.dirname(connections_file), exist_ok=True)

            with open(connections_file, 'w') as f:
                json.dump(connections_data, f, indent=4)

            self._sync_sources_to_file()

            logger.info("Source with ID %s, its mappings, and connections.json updated successfully", source_id)
            return True
        except Exception as e:
            logger.error("Error deleting source with ID %s: %s", source_id, str(e))
            return False

    def update_source(self, source_id: int, source_name: str, source_type: str,
                      source_details: Optional[Dict[str, Any]]) -> bool:
        query = """
        UPDATE flatpack_sources
        SET source_name = ?, source_type = ?, source_details = ?
        WHERE id = ?
        """
        source_details_json = json.dumps(source_details) if source_details else None
        try:
            self._execute_query(query, (source_name, source_type, source_details_json, source_id))
            logger.info("Source with ID %s updated successfully", source_id)

            self._sync_sources_to_file()

            return True
        except Exception as e:
            logger.error("Error updating source with ID %s: %s", source_id, str(e))
            return False

    def _sync_sources_to_file(self):
        """Sync current sources to sources.json file."""
        try:
            sources = self.get_all_sources()
            sources_data = {
                "sources": [
                    {
                        "source_name": source["source_name"],
                        "source_type": source["source_type"],
                        "source_details": source["source_details"]
                    }
                    for source in sources
                ]
            }

            sources_file = os.path.join(os.path.dirname(self.db_path), 'sources.json')
            os.makedirs(os.path.dirname(sources_file), exist_ok=True)

            with open(sources_file, 'w') as f:
                json.dump(sources_data, f, indent=4)

            logger.info("sources.json updated successfully")
        except Exception as e:
            logger.error("Error updating sources.json: %s", str(e))
            raise
