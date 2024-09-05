import argparse
import ast
import asyncio
import atexit
import base64
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import sqlite3
import stat
import string
import subprocess
import sys
import tarfile
import tempfile
import time
import warnings

from datetime import datetime, timedelta, timezone
from functools import partial
from importlib.metadata import version
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from sqlite3 import Error
from typing import List, Optional, Union
from zipfile import ZipFile

import httpx
import ngrok
import requests
import toml
import uvicorn
import zstandard as zstd

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from croniter import croniter
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from fastapi import BackgroundTasks, Depends, FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from prettytable import PrettyTable
from pydantic import BaseModel

from .agent_manager import AgentManager
from .parsers import parse_toml_to_venv_script
from .session_manager import SessionManager
from .vector_manager import VectorManager

HOME_DIR = Path.home() / ".fpk"
HOME_DIR.mkdir(exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/RomlinGroup/Flatpack/main/warehouse"
CONFIG_FILE_PATH = HOME_DIR / ".fpk_config.toml"
GITHUB_CACHE = HOME_DIR / ".fpk_github.cache"
GITHUB_CACHE_EXPIRY = timedelta(hours=1)
GITHUB_REPO_URL = "https://api.github.com/repos/RomlinGroup/Flatpack"
TEMPLATE_REPO_URL = "https://api.github.com/repos/RomlinGroup/template"
VERSION = version("flatpack")

MAX_ATTEMPTS = 5
VALIDATION_ATTEMPTS = 0

SERVER_START_TIME = None
COOLDOWN_PERIOD = timedelta(minutes=1)

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


class Comment(BaseModel):
    block_id: str
    selected_text: str
    comment: str


class EndpointFilter(logging.Filter):
    def filter(self, record):
        return all(
            endpoint not in record.getMessage()
            for endpoint in ['GET /api/heartbeat', 'GET /api/build-status']
        )


class Hook(BaseModel):
    hook_name: str
    hook_script: str
    hook_type: str


def setup_logging(log_path: Path):
    """Set up logging configuration."""
    new_logger = logging.getLogger(__name__)
    new_logger.setLevel(logging.WARNING)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    new_logger.addHandler(console_handler)
    new_logger.addHandler(file_handler)

    return new_logger


# Initialize the logger
global_log_file_path = HOME_DIR / ".fpk_logger.log"
logger = setup_logging(global_log_file_path)
os.chmod(global_log_file_path, 0o600)

logger = logging.getLogger("schedule_logger")
schedule_lock = asyncio.Lock()

uvicorn_server = None


def handle_termination_signal(signal_number, frame):
    """Handle termination signals for graceful shutdown."""
    logger.info("Received termination signal (%s), shutting down...", signal_number)
    sys.exit(0)


signal.signal(signal.SIGINT, handle_termination_signal)
signal.signal(signal.SIGTERM, handle_termination_signal)


def add_hook_to_database(hook: Hook):
    global flatpack_directory

    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    db_path = os.path.join(flatpack_directory, 'build', 'flatpack.db')

    try:
        ensure_database_initialized()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO flatpack_hooks (hook_name, hook_script, hook_type)
            VALUES (?, ?, ?)
        """, (hook.hook_name, hook.hook_script, hook.hook_type))

        hook_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return {"message": "Hook added successfully.", "hook_id": hook_id}
    except Error as e:
        logger.error("An error occurred while adding the hook: %s", e)
        print(f"[ERROR] An error occurred while adding the hook: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while adding the hook: {e}")


def authenticate_token(request: Request):
    """Authenticate the token."""
    global VALIDATION_ATTEMPTS
    token = request.headers.get('Authorization')
    stored_token = get_token()

    if not stored_token:
        logger.error("Stored token is not set")
        print("[ERROR] Stored token is not set")
        return

    if token is None or token != f"Bearer {stored_token}":
        VALIDATION_ATTEMPTS += 1
        logger.error(f"Invalid or missing token. Attempt {VALIDATION_ATTEMPTS}")
        print(f"[ERROR] Invalid or missing token. Attempt {VALIDATION_ATTEMPTS}")
        if VALIDATION_ATTEMPTS >= MAX_ATTEMPTS:
            shutdown_server()
        raise HTTPException(status_code=403, detail="Invalid or missing token")


async def check_and_run_schedules():
    global SERVER_START_TIME

    if not flatpack_directory:
        logger.error("Flatpack directory is not set.")
        print("[ERROR] Flatpack directory is not set.")
        return

    now = datetime.now(timezone.utc)
    logger.debug(f"[DEBUG] Current time: {now}")
    print(f"[DEBUG] Current time: {now}")

    if SERVER_START_TIME and (now - SERVER_START_TIME) < COOLDOWN_PERIOD:
        logger.info("In cooldown period. Skipping schedule check.")
        print("[INFO] In cooldown period. Skipping schedule check.")
        return

    db_path = os.path.join(flatpack_directory, 'build', 'flatpack.db')

    async with schedule_lock:
        try:
            logger.debug(f"[DEBUG] Opening database connection to {db_path}")
            print(f"[DEBUG] Opening database connection to {db_path}")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT id, type, pattern, datetimes, last_run FROM flatpack_schedule")
            schedules = cursor.fetchall()
            logger.debug(f"[DEBUG] Fetched schedules: {schedules}")
            print(f"[DEBUG] Fetched schedules: {schedules}")

            for schedule_id, schedule_type, pattern, datetimes, last_run in schedules:
                logger.debug(
                    f"[DEBUG] Processing schedule ID: {schedule_id}, Type: {schedule_type}, Pattern: {pattern}, Datetimes: {datetimes}, Last Run: {last_run}")

                if schedule_type == 'recurring':
                    if pattern:
                        cron = croniter(pattern, now)
                        prev_run = cron.get_prev(datetime)
                        next_run = cron.get_next(datetime)
                        logger.debug(f"[DEBUG] Cron pattern: {pattern}, Previous run: {prev_run}, Next run: {next_run}")

                        if last_run:
                            last_run_dt = datetime.fromisoformat(last_run)
                        else:
                            last_run_dt = None

                        if prev_run <= now < next_run:
                            if last_run_dt is None or last_run_dt < prev_run:
                                logger.debug(f"[DEBUG] Running build for schedule {schedule_id} at {now}")

                                await run_build_process(schedule_id)

                                cursor.execute("UPDATE flatpack_schedule SET last_run = ? WHERE id = ?",
                                               (now.isoformat(), schedule_id))
                                conn.commit()

                                logger.info(f"Executed recurring build for schedule {schedule_id}")
                                print(f"[INFO] Executed recurring build for schedule {schedule_id}")
                            else:
                                logger.debug(
                                    f"[DEBUG] Skipped recurring build for schedule {schedule_id}. Already run in this window.")
                                print(
                                    f"[DEBUG] Skipped recurring build for schedule {schedule_id}. Already run in this window.")
                        else:
                            logger.debug(
                                f"[DEBUG] Skipped recurring build for schedule {schedule_id}. Not within run window.")
                            print(f"[DEBUG] Skipped recurring build for schedule {schedule_id}. Not within run window.")

                elif schedule_type == 'manual':
                    if datetimes:
                        datetimes_list = json.loads(datetimes)
                        logger.debug(f"[DEBUG] Manual schedule with datetimes: {datetimes_list}")

                        for dt in datetimes_list:
                            scheduled_time = datetime.fromisoformat(dt).replace(tzinfo=timezone.utc)
                            logger.debug(f"[DEBUG] Checking manual schedule ID {schedule_id} for time {scheduled_time}")

                            if scheduled_time <= now:
                                logger.debug(f"[DEBUG] Running manual build for schedule {schedule_id} at {now}")

                                await run_build_process(schedule_id)
                                cursor.execute("UPDATE flatpack_schedule SET last_run = ? WHERE id = ?",
                                               (now.isoformat(), schedule_id))
                                conn.commit()

                                logger.info(f"Executed manual build for schedule {schedule_id}")
                                print(f"[INFO] Executed manual build for schedule {schedule_id}")
                            else:
                                logger.debug(
                                    f"[DEBUG] Skipped manual build for schedule {schedule_id}. Scheduled time {scheduled_time} is in the future.")
                                print(
                                    f"[DEBUG] Skipped manual build for schedule {schedule_id}. Scheduled time {scheduled_time} is in the future.")

        except sqlite3.Error as e:
            logger.error("Database error: %s", e)
            print(f"[ERROR] Database error: {e}")
        except Exception as e:
            logger.error("An error occurred: %s", e)
            print(f"[ERROR] An error occurred: {e}")
        finally:
            if conn:
                conn.close()
                logger.debug(f"[DEBUG] Closing database connection to {db_path}")
                print(f"[DEBUG] Closing database connection to {db_path}")


def create_temp_sh(custom_sh_path: Path, temp_sh_path: Path, use_euxo: bool = False):
    try:
        with custom_sh_path.open('r') as infile:
            script = infile.read()

        script = strip_html(script)
        lines = script.splitlines()
        parts = []

        i = 0

        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('part_bash """') or line.startswith('part_python """') or line.startswith(
                    'disabled part_bash """') or line.startswith('disabled part_python """'):
                start_line = i
                i += 1
                while i < len(lines) and not lines[i].strip().endswith('"""'):
                    i += 1
                end_line = i
                if i < len(lines):
                    end_line += 1
                parts.append('\n'.join(lines[start_line:end_line]).strip())
            i += 1

        last_count = sum(1 for part in parts if not part.startswith('disabled'))

        temp_sh_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as context_python_script:
            context_python_script_path = Path(context_python_script.name)
            context_python_script.write('')

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as exec_python_script:
            exec_python_script_path = Path(exec_python_script.name)
            exec_python_script.write('')

        with temp_sh_path.open('w') as outfile:
            outfile.write("#!/bin/bash\n")

            if use_euxo:
                outfile.write("set -euxo pipefail\n")
            else:
                outfile.write("set -euo pipefail\n")

            outfile.write(f"CONTEXT_PYTHON_SCRIPT=\"{context_python_script_path}\"\n")
            outfile.write("EVAL_BUILD=\"$(dirname \"$SCRIPT_DIR\")/web/eval_build.json\"\n")
            outfile.write(f"EXEC_PYTHON_SCRIPT=\"{exec_python_script_path}\"\n")
            outfile.write("CURR=0\n")

            outfile.write("trap 'rm -f \"$CONTEXT_PYTHON_SCRIPT\" \"$EXEC_PYTHON_SCRIPT\"' EXIT\n")
            outfile.write("rm -f \"$CONTEXT_PYTHON_SCRIPT\" c\"$EXEC_PYTHON_SCRIPT\"\n")
            outfile.write("touch \"$CONTEXT_PYTHON_SCRIPT\" \"$EXEC_PYTHON_SCRIPT\"\n")

            outfile.write("datetime=$(date -u +\"%Y-%m-%d %H:%M:%S\")\n")

            outfile.write("DATA_FILE=\"$(dirname \"$SCRIPT_DIR\")/web/eval_data.json\"\n")
            outfile.write("echo '[]' > \"$DATA_FILE\"\n")
            outfile.write("\n")
            outfile.write("function log_data() {\n")
            outfile.write("    local part_number=\"$1\"\n")
            outfile.write("    local new_files=$(find \"$SCRIPT_DIR\" -type f -newer \"$DATA_FILE\")\n")
            outfile.write("    if [ -n \"$new_files\" ]; then\n")
            outfile.write("        local temp_file=$(mktemp)\n")
            outfile.write("        for file in $new_files; do\n")
            outfile.write(
                "            if [[ \"$file\" == *\"eval_data.json\" || \"$file\" == *\"eval_build.json\" ]]; then continue; fi\n")
            outfile.write("            local mime_type=$(file --mime-type -b \"$file\")\n")
            outfile.write(
                "            local json_entry=\"{\\\"part\\\": $part_number, \\\"file\\\": \\\"$file\\\", \\\"mime_type\\\": \\\"$mime_type\\\"}\"\n")
            outfile.write(
                "            jq \". + [$json_entry]\" \"$DATA_FILE\" > \"$temp_file\" && mv \"$temp_file\" \"$DATA_FILE\"\n")
            outfile.write("        done\n")
            outfile.write("    fi\n")
            outfile.write("    touch \"$DATA_FILE\"\n")
            outfile.write("}\n")
            outfile.write("\n")

            outfile.write(
                "echo \"{"
                "\\\"curr\\\": $CURR, "
                f"\\\"last\\\": {last_count}, "
                "\\\"eval\\\": 1, "
                "\\\"datetime\\\": \\\"$datetime\\\""
                "}\" > \"$EVAL_BUILD\"\n"
            )

            for part in parts:
                part_lines = part.splitlines()
                if len(part_lines) < 2:
                    print(f"Skipping part with insufficient lines: {part}")
                    continue

                header = part_lines[0].strip()
                code_lines = part_lines[1:-1]

                if header.startswith('disabled'):
                    print(f"Skipping disabled part: {header}")
                    continue

                language = 'bash' if 'part_bash' in header else 'python' if 'part_python' in header else None
                code = '\n'.join(code_lines).strip().replace('\\"', '"')

                if language == 'bash':
                    code = code.replace('$', '\$').replace('\\$\\$', '$$')
                    outfile.write(f"{code}\n")
                    outfile.write("((CURR++))\n")

                elif language == 'python':
                    escaped_code = code.replace('"', '\\"')

                    outfile.write(f"echo \"\"\"{escaped_code}\"\"\" >> \"$CONTEXT_PYTHON_SCRIPT\"\n")

                    outfile.write("echo \"try:\" > \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("cat \"$CONTEXT_PYTHON_SCRIPT\" | sed 's/^/    /' >> \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("echo \"except Exception as e:\" >> \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("echo \"    print(e)\" >> \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("echo \"    import sys; sys.exit(1)\" >> \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("$VENV_PYTHON \"$EXEC_PYTHON_SCRIPT\"\n")

                    outfile.write("((CURR++))\n")

                else:
                    print(f"Skipping part with unsupported language: {language}")
                    continue

                outfile.write("log_data \"$CURR\"\n")

                outfile.write(
                    f"if [ \"$CURR\" -eq \"{last_count}\" ]; then\n"
                    "    EVAL=\"null\"\n"
                    "else\n"
                    "    EVAL=$((CURR + 1))\n"
                    "fi\n"
                )

                outfile.write(
                    "echo \"{"
                    "\\\"curr\\\": $CURR, "
                    f"\\\"last\\\": {last_count}, "
                    "\\\"eval\\\": $EVAL, "
                    "\\\"datetime\\\": \\\"$datetime\\\""
                    "}\" > \"$EVAL_BUILD\"\n"
                )

        context_python_script_path.unlink()
        exec_python_script_path.unlink()

        logger.info("Temp script generated successfully at %s", temp_sh_path)
        print(f"[INFO] Temp script generated successfully at {temp_sh_path}")

    except Exception as e:
        logger.error("An error occurred while creating temp script: %s", e)
        print(f"[ERROR] An error occurred while creating temp script: {e}")


def create_venv(venv_dir: str):
    """
    Create a virtual environment in the specified directory.

    Args:
        venv_dir (str): The directory where the virtual environment will be created.

    Returns:
        None
    """
    python_executable = sys.executable

    try:
        subprocess.run(
            [python_executable, "-m", "venv", venv_dir],
            check=True
        )
        logger.info("Virtual environment created successfully in '%s'.", venv_dir)
        print(f"[INFO] Virtual environment created successfully in '{venv_dir}'.")
    except subprocess.CalledProcessError as e:
        logger.error("Failed to create virtual environment: %s", e)
        print(f"[ERROR] Failed to create virtual environment: {e}")
    except Exception as e:
        logger.error("An unexpected error occurred while creating virtual environment: %s", e)
        print(f"[ERROR] An unexpected error occurred while creating virtual environment: {e}")


def decompress_data(input_path, output_path, allowed_dir=None):
    try:
        abs_input_path = validate_file_path(input_path, allowed_dir=allowed_dir)
        abs_output_path = validate_file_path(output_path, is_input=False, allowed_dir=allowed_dir)

        with open(abs_input_path, 'rb') as f:
            compressed_data = f.read()

        decompressed_data = zstd.decompress(compressed_data)

        try:
            with tempfile.NamedTemporaryFile() as tmp_file:
                tmp_file.write(decompressed_data)
                tmp_file.seek(0)

                with tarfile.open(fileobj=tmp_file, mode='r:') as tar:

                    for member in tar.getmembers():
                        member_path = os.path.join(abs_output_path, member.name)
                        if not os.path.commonprefix([abs_output_path, os.path.abspath(member_path)]) == abs_output_path:
                            raise Exception(f"Attempted Path Traversal in Tar File: {member.name}")
                    tar.extractall(path=abs_output_path)
        except tarfile.ReadError:

            with open(abs_output_path, 'wb') as f:
                f.write(decompressed_data)

    except Exception as e:
        logging.error("An error occurred while decompressing: %s", e)
        print(f"[ERROR] An error occurred while decompressing: {e}")


def ensure_database_initialized():
    if flatpack_directory:
        db_path = os.path.join(flatpack_directory, 'build', 'flatpack.db')
        initialize_database(db_path)


def escape_content_parts(content: str) -> str:
    """Escape special characters within content parts."""
    parts = content.split('part_')
    escaped_content = parts[0]
    for part in parts[1:]:
        if part.startswith('bash """') or part.startswith('python """'):
            type_and_content = part.split('"""', 1)
            if len(type_and_content) > 1:
                type_and_header, code = type_and_content
                code, footer = code.rsplit('"""', 1)
                escaped_content += f'part_{type_and_header}"""{escape_special_chars(code)}"""{footer}'
            else:
                escaped_content += f'part_{part}'
        else:
            escaped_content += f'part_{part}'
    return escaped_content


def escape_special_chars(content: str) -> str:
    """Escape special characters in a given string."""
    return content.replace('"', '\\"')


def generate_secure_token(length=32):
    """Generate a secure token of the specified length.

    Args:
        length (int): The length of the token to generate. Default is 32.

    Returns:
        str: A securely generated token.
    """
    alphabet = string.ascii_letters + string.digits + '-._~'
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def get_all_hooks_from_database():
    global flatpack_directory

    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    db_path = os.path.join(flatpack_directory, 'build', 'flatpack.db')

    try:
        ensure_database_initialized()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, hook_name, hook_script, hook_type FROM flatpack_hooks")
        hooks = cursor.fetchall()
        conn.close()

        hooks_list = [{"id": hook[0], "hook_name": hook[1], "hook_script": hook[2], "hook_type": hook[3]} for hook in
                      hooks]
        return hooks_list

    except Error as e:
        logger.error("An error occurred while fetching hooks: %s", e)
        print(f"[ERROR] An error occurred while fetching hooks: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching hooks: {e}")


def get_token() -> Optional[str]:
    """Retrieve the token from the configuration file."""
    config = load_config()
    return config.get('token')


def initialize_database(db_path: str):
    """Initialize the SQLite database if it doesn't exist."""
    try:
        db_dir = os.path.dirname(db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"[DEBUG] Created directory for database at path: {db_dir}")

        if os.path.exists(db_path):
            return

        print(f"[DEBUG] Initializing database at path: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flatpack_comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                block_id TEXT NOT NULL,
                selected_text TEXT NOT NULL,
                comment TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flatpack_hooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hook_name TEXT NOT NULL,
                hook_script TEXT NOT NULL,
                hook_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flatpack_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT NOT NULL,
                value TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS flatpack_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                pattern TEXT,
                datetimes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_run TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully at %s", db_path)
        print(f"[INFO] Database initialized successfully at {db_path}")
    except Error as e:
        logger.error("An error occurred while initializing the database: %s", e)
        print(f"[ERROR] An error occurred while initializing the database: {e}")


def load_config():
    """Load the configuration from the file.

    Returns:
        dict: The loaded configuration.
    """
    if not os.path.exists(CONFIG_FILE_PATH):
        logger.warning("Configuration file does not exist: %s", CONFIG_FILE_PATH)
        print(f"[WARNING] Configuration file does not exist: {CONFIG_FILE_PATH}")
        return {}

    try:
        with open(CONFIG_FILE_PATH, 'r') as config_file:
            config = toml.load(config_file)
            return config
    except Exception as e:
        error_message = f"Error loading config: {e}"
        logger.error("Error loading config: %s", e)
        print(f"[ERROR] {error_message}")
        return {}


async def run_build_process(schedule_id=None):
    logger.info("Running build process...")
    try:
        update_build_status("in_progress", schedule_id)

        steps = [
            ("Preparing build environment", 1),
            ("Compiling source code", 1),
            ("Running tests", 1),
            ("Packaging application", 1)
        ]

        for step_name, duration in steps:
            update_build_status(f"in_progress: {step_name}", schedule_id)
            await asyncio.sleep(duration)

        loop = asyncio.get_event_loop()
        update_build_status("in_progress: Running build script", schedule_id)
        await loop.run_in_executor(None, partial(fpk_build, flatpack_directory))

        update_build_status("completed", schedule_id)
        logger.info("Build process completed.")
    except Exception as e:
        logger.error("Build process failed: %s", e)
        update_build_status("failed", schedule_id, error=str(e))


async def run_scheduler():
    while True:
        await check_and_run_schedules()
        await asyncio.sleep(60)


def save_config(config):
    """Save the configuration to the file, sorting keys alphabetically.

    Args:
        config (dict): The configuration to save.
    """
    sorted_config = {k: config[k] for k in sorted(config)}

    try:
        with open(CONFIG_FILE_PATH, "w") as config_file:
            toml.dump(sorted_config, config_file)
        os.chmod(CONFIG_FILE_PATH, 0o600)
        logger.info("Configuration saved successfully to %s", CONFIG_FILE_PATH)
        print(f"[INFO] Configuration saved successfully to {CONFIG_FILE_PATH}")
    except Exception as e:
        error_message = f"Error saving config: {e}"
        logger.error("Error saving config: %s", error_message)
        print(f"[ERROR] {error_message}")


def set_token(token: str):
    try:
        config = load_config()
        config['token'] = token
        save_config(config)
        logger.info("Token set successfully.")
        print("[INFO] Token set successfully!")
    except Exception as e:
        logger.error("Failed to set token: %s", str(e))
        print(f"[ERROR] Failed to set token: {str(e)}")


def setup_static_directory(fastapi_app: FastAPI, directory: str):
    """Setup the static directory for serving static files."""
    global flatpack_directory
    flatpack_directory = os.path.abspath(directory)

    if os.path.exists(flatpack_directory) and os.path.isdir(flatpack_directory):
        static_dir = os.path.join(flatpack_directory, 'web')
        fastapi_app.mount(
            "/",
            StaticFiles(directory=static_dir, html=True),
            name="static"
        )
        logger.info("Static files will be served from: %s", static_dir)
        print(f"[INFO] Static files will be served from: {static_dir}")
    else:
        logger.error("The directory '%s' does not exist or is not a directory.", flatpack_directory)
        print(f"[ERROR] The directory '{flatpack_directory}' does not exist or is not a directory.")
        raise ValueError(error_message)


def shutdown_server():
    """Shutdown the FastAPI server."""
    logging.getLogger("uvicorn.error").info("Shutting down the server after maximum validation attempts.")
    os._exit(0)


def strip_html(content: str) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    return soup.get_text()


def unescape_content_parts(content: str) -> str:
    """Unescape special characters within content parts."""
    parts = content.split('part_')
    unescaped_content = parts[0]
    for part in parts[1:]:
        if part.startswith('bash """') or part.startswith('python """'):
            type_and_content = part.split('"""', 1)
            if len(type_and_content) > 1:
                type_and_header, code = type_and_content
                code, footer = code.rsplit('"""', 1)
                unescaped_content += f'part_{type_and_header}"""{unescape_special_chars(code)}"""{footer}'
            else:
                unescaped_content += f'part_{part}'
        else:
            unescaped_content += f'part_{part}'
    return unescaped_content


def unescape_special_chars(content: str) -> str:
    """Unescape special characters in a given string."""
    return content.replace('\\"', '"')


def update_build_status(status, schedule_id=None, error=None):
    status_file = os.path.join(flatpack_directory, 'build', 'build_status.json')
    status_data = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "schedule_id": schedule_id
    }
    if error:
        status_data["error"] = str(error)
    with open(status_file, 'w') as f:
        json.dump(status_data, f)


def validate_api_token(api_token: str) -> bool:
    """Validate the API token."""
    return api_token == get_token()


def validate_file_path(path, is_input=True, allowed_dir=None):
    """
    Validate the file path to prevent directory traversal attacks.

    Parameters:
        path (str): The path to validate.
        is_input (bool): Flag indicating if the path is for input. Defaults to True.
        allowed_dir (str): The allowed directory for the path. Defaults to None.

    Returns:
        str: The absolute path if valid.

    Raises:
        ValueError: If the path is outside the allowed directory or invalid.
        FileNotFoundError: If the path does not exist.
    """
    absolute_path = os.path.abspath(path)

    if allowed_dir:
        allowed_dir_absolute = os.path.abspath(allowed_dir)
        if not absolute_path.startswith(allowed_dir_absolute):
            raise ValueError(
                f"Path '{path}' is outside the allowed directory '{allowed_dir}'."
            )

    if is_input:
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(f"The path '{absolute_path}' does not exist.")
        if not (os.path.isfile(absolute_path) or os.path.isdir(absolute_path)):
            raise ValueError(
                f"The path '{absolute_path}' is neither a file nor a directory."
            )
    else:
        output_dir = os.path.dirname(absolute_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    return absolute_path


def fpk_build(directory: Union[str, None], use_euxo: bool = False):
    """Build a flatpack.

    Args:
        directory (Union[str, None]): The directory to use for building the flatpack. If None, a cached directory will be used if available.
        use_euxo (bool): Whether to use 'set -euxo pipefail' in the shell script. Defaults to False.

    Returns:
        None
    """
    cache_file_path = HOME_DIR / ".fpk_unbox.cache"

    if directory:
        flatpack_dir = Path.cwd() / directory
        if not flatpack_dir.exists():
            logger.error("The directory '%s' does not exist.", flatpack_dir)
            print(f"[ERROR] The directory '{flatpack_dir}' does not exist.")
            return
        last_unboxed_flatpack = str(flatpack_dir)
    elif cache_file_path.exists():
        logger.info("Found cached flatpack in %s", cache_file_path)
        print(f"[INFO] Found cached flatpack in {cache_file_path}.")
        last_unboxed_flatpack = cache_file_path.read_text().strip()
    else:
        logger.error("No cached flatpack found, and no valid directory provided.")
        print("[ERROR] No cached flatpack found, and no valid directory provided.")
        return

    flatpack_dir = Path(last_unboxed_flatpack)

    if not flatpack_dir.exists():
        logger.error("The flatpack directory '%s' does not exist.", flatpack_dir)
        print(f"[ERROR] The flatpack directory '{flatpack_dir}' does not exist.")
        return

    build_dir = flatpack_dir / 'build'

    if not build_dir.exists():
        logger.error("The build directory '%s' does not exist.", build_dir)
        print(f"[ERROR] The build directory '{build_dir}' does not exist.")
        return

    custom_sh_path = build_dir / 'custom.sh'

    if not custom_sh_path.exists() or not custom_sh_path.is_file():
        logger.error("custom.sh not found in %s. Build process canceled.", build_dir)
        print(f"[ERROR] custom.sh not found in {build_dir}. Build process canceled.")
        return

    temp_sh_path = build_dir / 'temp.sh'
    create_temp_sh(custom_sh_path, temp_sh_path, use_euxo=use_euxo)

    building_script_path = build_dir / 'build.sh'

    if not building_script_path.exists() or not building_script_path.is_file():
        logger.error("Building script not found in %s", build_dir)
        print(f"[ERROR] Building script not found in {build_dir}.")
        return

    log_dir = build_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_time = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
    log_filename = f"build_{log_file_time}.log"
    build_log_file_path = log_dir / log_filename

    safe_script_path = shlex.quote(str(building_script_path.resolve()))

    try:
        with open(build_log_file_path, 'w') as log_file:
            process = subprocess.Popen(
                ['/bin/bash', '-u', safe_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            for line in process.stdout:
                logger.info(line.strip())
                log_file.write(line)
                print(line, end='')

            process.wait()

            web_dir = flatpack_dir / 'web'

            if not web_dir.exists():
                logger.error("The web directory '%s' does not exist.", web_dir)
                print(f"[ERROR] The web directory '{web_dir}' does not exist.")
                return

            eval_data_path = web_dir / "eval_data.json"

            if not eval_data_path.exists():
                logger.error("The 'eval_data.json' file does not exist in '%s'.", web_dir)
                print(f"[ERROR] The 'eval_data.json' file does not exist in '{web_dir}'.")
                return
            else:
                output_dir = web_dir / "output"

                if output_dir.exists():
                    shutil.rmtree(output_dir)

                output_dir.mkdir(parents=True)

                with eval_data_path.open('r') as file:
                    eval_data = json.load(file)

                    for item in eval_data:
                        original_path = Path(item['file'])

                        relative_path = None
                        for parent in original_path.parents:
                            if parent.parts[-1] == 'build':
                                relative_path = original_path.relative_to(parent)
                                break

                        if relative_path:
                            source_file = build_dir / relative_path

                            if source_file.exists():
                                dest_path = output_dir / source_file.name
                                shutil.copy2(source_file, dest_path)
                                print(f"[INFO] Copied {source_file} to {dest_path}")
                            else:
                                print(f"[ERROR] File {source_file} not found.")
                        else:
                            print(f"[ERROR] Could not determine relative path for {original_path}")

            db_path = build_dir / 'flatpack.db'
            initialize_database(str(db_path))

            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT hook_name, hook_script, hook_type FROM flatpack_hooks")
                hooks = cursor.fetchall()
                conn.close()

                for hook_name, hook_script, hook_type in hooks:
                    logger.info("Executing hook: %s", hook_name)
                    print(f"[INFO] Executing hook: {hook_name}")

                    if hook_type == "bash":
                        hook_command = shlex.split(hook_script)
                        hook_process = subprocess.Popen(
                            hook_command,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True
                        )

                        for line in hook_process.stdout:
                            logger.info(line.strip())
                            log_file.write(line)
                            print(line, end='')

                        hook_process.wait()

                    elif hook_type == "python":
                        try:
                            exec(hook_script)
                        except Exception as e:
                            logger.error("An error occurred while executing Python hook %s: %s", hook_name, e)
                            print(f"[ERROR] An error occurred while executing Python hook {hook_name}: {e}")
                    else:
                        logger.warning("Unsupported hook type for %s. Skipping.", hook_name)
                        print(f"[WARNING] Unsupported hook type for {hook_name}. Skipping.")

            except sqlite3.Error as e:
                logger.error("An error occurred while retrieving or executing hooks: %s", e)
                print(f"[ERROR] An error occurred while retrieving or executing hooks: {e}")

    except subprocess.CalledProcessError as e:
        logger.error("An error occurred while executing the build script: %s", e)
        print(f"[ERROR] An error occurred while executing the build script: {e}")
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        print(f"[ERROR] An unexpected error occurred: {e}")


def fpk_cache_unbox(directory_name: str):
    """Cache the last unboxed flatpack's directory name to a file.

    Args:
        directory_name (str): The name of the directory to cache.

    Returns:
        None
    """
    cache_file_path = HOME_DIR / ".fpk_unbox.cache"

    try:
        with open(cache_file_path, 'w') as f:
            f.write(directory_name)

        logger.info(
            "Cached the directory name '%s' to %s",
            directory_name,
            cache_file_path
        )
        print(f"[INFO] Cached the directory name '{directory_name}' to {cache_file_path}.")

    except IOError as e:
        logger.error("Failed to cache the directory name '%s': %s", directory_name, e)
        print(f"[ERROR] Failed to cache the directory name '{directory_name}': {e}")


def fpk_check_ngrok_auth():
    """Check if the NGROK_AUTHTOKEN environment variable is set.

    Returns:
        None: Exits the program if the NGROK_AUTHTOKEN is not set.
    """
    ngrok_auth_token = os.environ.get('NGROK_AUTHTOKEN')

    if not ngrok_auth_token:
        message = (
            "NGROK_AUTHTOKEN is not set. Please set it using:\n"
            "export NGROK_AUTHTOKEN='your_ngrok_auth_token'"
        )
        logger.error("%s", message)
        print(f"[ERROR] {message}")
        sys.exit(1)
    else:
        message = "NGROK_AUTHTOKEN is set."
        logger.info("%s", message)
        print(f"[INFO] {message}")


def fpk_colorize(text, color):
    """Colorize a given text with the specified color.

    Args:
        text (str): The text to be colorized.
        color (str): The color to apply to the text.

    Returns:
        str: Colorized text or the original text if the color is invalid.
    """
    colors = {
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "default": "\033[0m",
        "green": "\033[92m",
        "grey": "\033[90m",
        "magenta": "\033[95m",
        "red": "\033[91m",
        "white": "\033[97m",
        "yellow": "\033[93m"
    }

    if color not in colors:
        logger.error("Invalid color '%s' provided. Returning the original text.", color)
        print(f"[ERROR] Invalid color '{color}' provided. Returning the original text.")
        return text

    return colors[color] + text + colors["default"]


def fpk_create(flatpack_name, repo_url=TEMPLATE_REPO_URL):
    """Create a new flatpack from a template repository.

    Args:
        flatpack_name (str): The name of the flatpack to create.
        repo_url (str, optional): The URL of the template repository. Defaults to TEMPLATE_REPO_URL.

    Raises:
        ValueError: If the flatpack name format is invalid or if the directory already exists.
    """
    if not re.match(r'^[a-z0-9-]+$', flatpack_name):
        error_message = "Invalid name format. Only lowercase letters, numbers, and hyphens are allowed."
        logger.error("%s", error_message)
        print(f"[ERROR] {error_message}")
        raise ValueError(error_message)

    flatpack_name = flatpack_name.lower().replace(' ', '-')
    current_dir = os.getcwd()
    flatpack_dir = os.path.join(current_dir, flatpack_name)

    if os.path.exists(flatpack_dir):
        error_message = f"Directory '{flatpack_name}' already exists. Aborting creation."
        logger.error("%s", error_message)
        print(f"[ERROR] {error_message}")
        raise ValueError(error_message)

    try:
        template_dir = fpk_download_and_extract_template(repo_url, current_dir)
    except Exception as e:
        error_message = f"Failed to download and extract template: {e}"
        logger.error("Failed to download and extract template: %s", e)
        print(f"[ERROR] {error_message}")
        return

    try:
        os.makedirs(flatpack_dir, exist_ok=True)
        logger.info("Created flatpack directory: %s", flatpack_dir)
        print(f"[INFO] Created flatpack directory: {flatpack_dir}")
    except OSError as e:
        error_message = f"Failed to create flatpack directory: {e}"
        logger.error("Failed to create flatpack directory: %s", e)
        print(f"[ERROR] {error_message}")
        return

    try:
        for item in os.listdir(template_dir):
            if item in ['.gitignore', 'LICENSE']:
                continue
            s = os.path.join(template_dir, item)
            d = os.path.join(flatpack_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
        logger.info("Copied template files to flatpack directory: %s", flatpack_dir)
        print(f"[INFO] Copied template files to flatpack directory: {flatpack_dir}")
    except OSError as e:
        error_message = f"Failed to copy template files: {e}"
        logger.error("Failed to copy template files: %s", e)
        print(f"[ERROR] {error_message}")
        return

    files_to_edit = [
        (
            os.path.join(flatpack_dir, "README.md"),
            r"# template",
            f"# {flatpack_name}"
        ),
        (
            os.path.join(flatpack_dir, "flatpack.toml"),
            r"{{model_name}}",
            flatpack_name
        ),
        (
            os.path.join(flatpack_dir, "build.sh"),
            r"export DEFAULT_REPO_NAME=template",
            f"export DEFAULT_REPO_NAME={flatpack_name}"
        ),
        (
            os.path.join(flatpack_dir, "build.sh"),
            r"export FLATPACK_NAME=template",
            f"export FLATPACK_NAME={flatpack_name}"
        )
    ]

    try:
        for file_path, pattern, replacement in files_to_edit:
            with open(file_path, 'r') as file:
                filedata = file.read()
            newdata = re.sub(pattern, replacement, filedata)
            with open(file_path, 'w') as file:
                file.write(newdata)
        logger.info("Edited template files for flatpack: %s", flatpack_name)
        print(f"[INFO] Edited template files for flatpack: {flatpack_name}")
    except OSError as e:
        error_message = f"Failed to edit template files: {e}"
        logger.error("Failed to edit template files: %s", e)
        print(f"[ERROR] {error_message}")
        return

    try:
        shutil.rmtree(template_dir)
        logger.info("Removed temporary template directory: %s", template_dir)
        print(f"[INFO] Removed temporary template directory: {template_dir}")
    except OSError as e:
        error_message = f"Failed to remove temporary template directory: {e}"
        logger.error("Failed to remove temporary template directory: %s", e)
        print(f"[ERROR] {error_message}")
        return

    logger.info("Successfully created %s", flatpack_name)
    print(f"[INFO] Successfully created {flatpack_name}.")


def fpk_display_disclaimer(directory_name: str, local: bool):
    """Display a disclaimer message with details about a specific flatpack.

    Args:
        directory_name (str): Name of the flatpack directory.
        local (bool): Indicates if the flatpack is local.
    """
    disclaimer_template = """
-----------------------------------------------------
STOP AND READ BEFORE YOU PROCEED
https://pypi.org/project/flatpack
Copyright 2024 Romlin Group AB

Licensed under the Apache License, Version 2.0
(the "License"); you may NOT use this Python package
except in compliance with the License. You may obtain
a copy of the License at:

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in
writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing
permissions and limitations under the License.
{please_note}
To accept, type 'YES'. To decline, type 'NO'.
-----------------------------------------------------
    """

    if not local:
        please_note_content = f"""
PLEASE NOTE: The flatpack you are about to unbox is
governed by its own licenses and terms, separate from
this software. You may find further details at:

https://fpk.ai/w/{directory_name}
        """
        please_note_colored = fpk_colorize(please_note_content, "yellow")
    else:
        please_note_colored = ""

    logger.info(
        "Displayed disclaimer for flatpack '%s' with local set to %s.",
        directory_name,
        local
    )
    disclaimer_message = disclaimer_template.format(please_note=please_note_colored)
    print(disclaimer_message)


def fpk_download_and_extract_template(repo_url, dest_dir):
    """
    Download and extract a template repository using GitHub API.

    Args:
        repo_url (str): The GitHub API URL of the template repository.
        dest_dir (str): The destination directory to extract the template into.

    Returns:
        str: The path to the extracted template directory.

    Raises:
        RuntimeError: If downloading or extracting the template fails.
    """
    template_dir = os.path.join(dest_dir, "template")
    try:
        repo_info_response = requests.get(repo_url)
        repo_info_response.raise_for_status()
        repo_info = repo_info_response.json()

        default_branch = repo_info['default_branch']

        zip_url = f"{repo_url}/zipball/{default_branch}"

        zip_response = requests.get(zip_url)
        zip_response.raise_for_status()

        with ZipFile(BytesIO(zip_response.content)) as zip_ref:
            top_level_dir = zip_ref.namelist()[0].split('/')[0]
            zip_ref.extractall(dest_dir)

        extracted_dir = os.path.join(dest_dir, top_level_dir)
        os.rename(extracted_dir, template_dir)

        index_html_path = os.path.join(template_dir, "index.html")
        if os.path.exists(index_html_path):
            os.remove(index_html_path)

        logger.info(
            "Downloaded and extracted template from %s to %s", repo_url, dest_dir
        )
        print(f"[INFO] Downloaded and extracted template from {repo_url} to {dest_dir}")
        return template_dir
    except requests.RequestException as e:
        error_message = f"Failed to download template from {repo_url}: {e}"
        logger.error("%s", error_message)
        print(f"[ERROR] {error_message}")
        raise RuntimeError(error_message)
    except (OSError, IOError) as e:
        error_message = f"Failed to extract template or remove index.html in {dest_dir}: {e}"
        logger.error("%s", error_message)
        print(f"[ERROR] {error_message}")
        raise RuntimeError(error_message)


def fpk_fetch_flatpack_toml_from_dir(directory_name: str, session: httpx.Client) -> Optional[str]:
    """Fetch the flatpack TOML configuration from a specific directory.

    Args:
        directory_name (str): Name of the flatpack directory.
        session (httpx.Client): HTTP client session for making requests.

    Returns:
        Optional[str]: The TOML content if found, otherwise None.
    """
    if not fpk_valid_directory_name(directory_name):
        message = f"Invalid directory name: '{directory_name}'."

        logger.error("%s", message)
        print(f"[ERROR] {message}")

        return None

    fpk_url = f"{BASE_URL}/{directory_name}/{directory_name}.fpk"
    toml_url = f"{BASE_URL}/{directory_name}/flatpack.toml"

    try:
        response = session.get(toml_url)
        response.raise_for_status()

        logger.info("Successfully fetched TOML from %s", toml_url)
        print(f"[INFO] Successfully fetched TOML from {toml_url}")

        return response.text
    except httpx.HTTPStatusError as e:
        message = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"

        logger.error("%s", message)
        print(f"[ERROR] {message}")

        return None
    except httpx.RequestError as e:
        message = f"Network error occurred: {e}"

        logger.error("%s", message)
        print(f"[ERROR] {message}")

        return None
    except Exception as e:
        message = f"An unexpected error occurred: {e}"

        logger.error("%s", message)
        print(f"[ERROR] {message}")

        return None


def fpk_fetch_github_dirs(session: httpx.Client) -> List[str]:
    """
    Fetch a list of directory names from the GitHub repository.
    Uses local caching to reduce GitHub API calls.

    Args:
        session (httpx.Client): HTTP client session for making requests.

    Returns:
        List[str]: List of directory names.
    """
    if os.path.exists(GITHUB_CACHE):
        with open(GITHUB_CACHE, 'r') as f:
            cache_data = json.load(f)

        cache_time = datetime.fromisoformat(cache_data['timestamp'])

        if datetime.now() - cache_time < GITHUB_CACHE_EXPIRY:
            return cache_data['directories']

    try:
        response = session.get(f"{GITHUB_REPO_URL}/contents/warehouse")
        response.raise_for_status()
        json_data = response.json()

        if isinstance(json_data, list):
            directories = [
                item['name'] for item in json_data
                if (
                        isinstance(item, dict) and item.get('type') == 'dir' and
                        item.get('name', '').lower()
                )
            ]

            directories = sorted(directories)

            os.makedirs(HOME_DIR, exist_ok=True)

            with open(GITHUB_CACHE, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'directories': directories
                }, f)

            logger.info("Cached GitHub dirs: %s", directories)
            print(f"[INFO] Cached GitHub dirs: {directories}")

            return directories

        message = f"Unexpected response format from GitHub: {json_data}"
        logger.error("%s", message)
        print(f"[ERROR] {message}")
        return []

    except httpx.HTTPError as e:
        message = f"Unable to connect to GitHub: {e}"
        logger.error("%s", message)
        print(f"[ERROR] {message}")
        sys.exit(1)
    except (ValueError, KeyError) as e:
        message = f"Error processing the response from GitHub: {e}"
        logger.error("%s", message)
        print(f"[ERROR] {message}")
        return []


def fpk_find_models(directory_path: str = None) -> List[str]:
    """Find model files in a specified directory or the current directory.

    Args:
        directory_path (Optional[str]): Path to the directory to search in. Defaults to the current directory.

    Returns:
        List[str]: List of found model file paths.
    """
    if directory_path is None:
        directory_path = os.getcwd()

    logger.info("Searching for model files in directory: %s", directory_path)

    model_file_formats = ['.caffemodel', '.ckpt', '.gguf', '.h5', '.mar', '.mlmodel', '.model', '.onnx',
                          '.params', '.pb', '.pkl', '.pickle', '.pt', '.pth', '.sav', '.tflite', '.weights']
    model_files = []

    try:
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.endswith(fmt) for fmt in model_file_formats):
                    model_file_path = os.path.join(root, file)
                    model_files.append(model_file_path)
                    logger.info("Found model file: %s", model_file_path)

        logger.info("Total number of model files found: %d", len(model_files))
        print(f"[INFO] Found {len(model_files)} model file(s).")

    except Exception as e:
        error_message = f"An error occurred while searching for model files: {e}"

        logger.error("%s", error_message)
        print(f"[ERROR] {error_message}")

    return model_files


def fpk_get_api_key() -> Optional[str]:
    """Retrieve the API key from the configuration file.

    Returns:
        Optional[str]: The API key if found, otherwise None.
    """
    try:
        config = load_config()
        api_key = config.get('api_key')
        if api_key:
            logger.info("API key retrieved successfully.")
            print("[INFO] API key retrieved successfully.")
        else:
            logger.info("API key not found in the configuration.")
            print("[INFO] API key not found in the configuration.")
        return api_key
    except Exception as e:
        error_message = f"An error occurred while retrieving the API key: {e}"
        logger.error(error_message)
        print(f"[ERROR] {error_message}")
        return None


def fpk_get_last_flatpack() -> Optional[str]:
    """Retrieve the last unboxed flatpack's directory name from the cache file.

    Returns:
        Optional[str]: The last unboxed flatpack directory name if found, otherwise None.
    """
    cache_file_path = HOME_DIR / ".fpk_unbox.cache"
    try:
        if cache_file_path.exists():
            with cache_file_path.open('r') as cache_file:
                last_flatpack = cache_file.read().strip()
                logger.info(
                    "Last unboxed flatpack directory retrieved: %s",
                    last_flatpack
                )
                print(
                    "[INFO] Last unboxed flatpack directory retrieved: %s",
                    last_flatpack
                )
                return last_flatpack
        else:
            logger.warning("Cache file does not exist: %s", cache_file_path)
            print("[WARNING] Cache file does not exist: %s", cache_file_path)
    except (OSError, IOError) as e:
        error_message = f"An error occurred while accessing the cache file: {e}"
        logger.error("%s", error_message)
        print("[ERROR] %s", error_message)
    return None


def fpk_initialize_vector_manager(args):
    """Initialize the Vector Manager.

    Args:
        args: The command-line arguments.

    Returns:
        VectorManager: An instance of VectorManager.
    """
    data_dir = getattr(args, 'data_dir', '.')
    logger.info(
        "Initializing Vector Manager and data directory: %s",
        data_dir
    )
    print(f"[INFO] Initializing Vector Manager and data directory: {data_dir}")
    return VectorManager(model_id='all-MiniLM-L6-v2', directory=data_dir)


def fpk_is_raspberry_pi() -> bool:
    """Check if we're running on a Raspberry Pi.

    Returns:
        bool: True if running on a Raspberry Pi, False otherwise.
    """
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('Hardware') and 'BCM' in line:
                    logger.info("Running on a Raspberry Pi.")
                    print("[INFO] Running on a Raspberry Pi.")
                    return True
    except IOError as e:
        logger.warning("Could not access /proc/cpuinfo: %s", e)
        print("[WARNING] Could not access /proc/cpuinfo:", e)
    logger.info("Not running on a Raspberry Pi.")
    print("[INFO] Not running on a Raspberry Pi.")
    return False


def fpk_list_directories(session: httpx.Client) -> str:
    """Fetch a list of directories from GitHub and return as a newline-separated string.

    Parameters:
    - session (httpx.Client): HTTP client session for making requests.

    Returns:
    - str: A newline-separated string of directory names.
    """
    try:
        dirs = fpk_fetch_github_dirs(session)
        directories_str = "\n".join(dirs)
        logger.info("Fetched directories: %s", directories_str)
        return directories_str
    except Exception as e:
        error_message = f"An error occurred while listing directories: {e}"
        logger.error(error_message)
        print("[ERROR] %s", error_message)
        return ""


def fpk_safe_cleanup():
    """Safely clean up temporary files."""
    try:
        files_to_delete = ["flatpack.sh"]
        current_directory = Path.cwd()

        for filename in files_to_delete:
            file_path = current_directory / filename

            if file_path.exists():
                file_path.unlink()
                print(f"Deleted {filename}.")
    except Exception as e:
        print(f"Exception during safe_cleanup: {e}")


def fpk_set_secure_file_permissions(file_path):
    """Set secure file permissions for the specified file.

    Args:
        file_path (str): Path to the file for which to set secure permissions.
    """
    try:
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
        logger.info("Set secure file permissions for %s", file_path)
        print(f"[INFO] Set secure file permissions for {file_path}")
    except OSError as e:
        error_message = f"Failed to set secure file permissions for {file_path}: {e}"
        logger.error("Failed to set secure file permissions for %s: %s", file_path, e)
        print(f"[ERROR] {error_message}")


def fpk_unbox(directory_name: str, session: httpx.Client, local: bool = False) -> bool:
    """Unbox a flatpack from GitHub or a local directory.

    Args:
        directory_name (str): Name of the flatpack directory.
        session (httpx.Client): HTTP client session for making requests.
        local (bool): Indicates if the flatpack is local. Defaults to False.

    Returns:
        bool: True if unboxing was successful, False otherwise.
    """
    if not fpk_valid_directory_name(directory_name):
        message = f"Invalid directory name: '{directory_name}'."
        logger.error("Invalid directory name: '%s'", directory_name)
        print(f"[ERROR] {message}")
        return False

    flatpack_dir = Path.cwd() / directory_name

    if local:
        if not flatpack_dir.exists():
            message = f"Local directory '{directory_name}' does not exist."
            logger.error(message)
            print(f"[ERROR] {message}")
            return False
        toml_path = flatpack_dir / 'flatpack.toml'
        if not toml_path.exists():
            message = f"flatpack.toml not found in the specified directory: '{directory_name}'."
            logger.error("%s", message)
            print(f"[ERROR] {message}")
            return False
    else:
        if flatpack_dir.exists():
            message = f"Directory '{directory_name}' already exists. Unboxing aborted to prevent conflicts."
            logger.error(message)
            print(f"[ERROR] {message}")
            return False
        flatpack_dir.mkdir(parents=True, exist_ok=True)

    web_dir = flatpack_dir / "web"

    try:
        web_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created /web directory: {web_dir}")
        print(f"[INFO] Created /web directory: {web_dir}")

        index_html_url = f"{TEMPLATE_REPO_URL}/contents/index.html"

        try:
            response = session.get(index_html_url)
            response.raise_for_status()
            index_html_content = response.json()['content']
            index_html_decoded = base64.b64decode(index_html_content).decode('utf-8')

            index_html_path = web_dir / "index.html"
            with open(index_html_path, 'w') as f:
                f.write(index_html_decoded)
            logger.info(f"Copied index.html to {index_html_path}")
            print(f"[INFO] Copied index.html to {index_html_path}")
        except Exception as e:
            logger.error(f"Failed to fetch or save index.html: {e}")
            print(f"[ERROR] Failed to fetch or save index.html: {e}")
    except Exception as e:
        message = f"Failed to create /web directory: {e}"
        logger.error(message)
        print(f"[ERROR] {message}")
        return False

    build_dir = flatpack_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    db_path = build_dir / 'flatpack.db'
    temp_toml_path = build_dir / 'temp_flatpack.toml'

    if not local:
        fpk_url = f"{BASE_URL}/{directory_name}/{directory_name}.fpk"
        try:
            response = session.head(fpk_url)
            if response.status_code != 200:
                logger.error(".fpk file does not exist at %s", fpk_url)
                print(f"[ERROR] .fpk file does not exist at {fpk_url}")
                return False
            logger.info(".fpk file found at %s", fpk_url)
            print(f"[INFO] .fpk file found at {fpk_url}")
        except httpx.RequestError as e:
            logger.error("Network error occurred while checking .fpk file: %s", e)
            print(f"[ERROR] Network error occurred while checking .fpk file: {e}")
            return False
        except Exception as e:
            logger.error("An unexpected error occurred: %s", e)
            print(f"[ERROR] An unexpected error occurred: {e}")
            return False

        fpk_path = build_dir / f"{directory_name}.fpk"

        try:
            with open(fpk_path, "wb") as fpk_file:
                download_response = session.get(fpk_url)
                fpk_file.write(download_response.content)
            logger.info("Downloaded .fpk file to %s", fpk_path)
            print(f"[INFO] Downloaded .fpk file to {fpk_path}")
        except httpx.RequestError as e:
            logger.error("Failed to download the .fpk file: %s", e)
            print(f"[ERROR] Failed to download the .fpk file: {e}")
            return False
        except Exception as e:
            logger.error("An unexpected error occurred during the .fpk download: %s", e)
            print(f"[ERROR] An unexpected error occurred during the .fpk download: {e}")
            return False

        try:
            decompress_data(fpk_path, build_dir)
            logger.info("Decompressed .fpk file into %s", build_dir)
            print(f"[INFO] Decompressed .fpk file into {build_dir}")
        except Exception as e:
            logger.error("Failed to decompress .fpk file: %s", e)
            print(f"[ERROR] Failed to decompress .fpk file: {e}")
            return False

        toml_path = build_dir / 'flatpack.toml'
        if not toml_path.exists():
            logger.error("flatpack.toml not found in %s", build_dir)
            print(f"[ERROR] flatpack.toml not found in {build_dir}.")
            return False

    toml_content = toml_path.read_text()
    temp_toml_path.write_text(toml_content)

    bash_script_content = parse_toml_to_venv_script(str(temp_toml_path), env_name=flatpack_dir)
    bash_script_path = build_dir / 'flatpack.sh'
    bash_script_path.write_text(bash_script_content)

    temp_toml_path.unlink()

    message = f"Unboxing {directory_name}..."
    logger.info("%s", message)
    print(f"[INFO] {message}")

    safe_script_path = shlex.quote(str(bash_script_path.resolve()))

    try:
        subprocess.run(['/bin/bash', safe_script_path], check=True)

        logger.info("All done!")
        print("[INFO] All done!")

        initialize_database(str(db_path))
        print(f"[INFO] Database initialized at {db_path}")

        fpk_cache_unbox(str(flatpack_dir))

        return True

    except subprocess.CalledProcessError as e:
        message = f"Failed to execute the bash script: {e}"
        logger.error(message)
        print(f"[ERROR] {message}")
        return False
    except Exception as e:
        message = f"An unexpected error occurred: {e}"
        logger.error(message)
        print(f"[ERROR] {message}")
        return False
    finally:
        bash_script_path.unlink()


def fpk_valid_directory_name(name: str) -> bool:
    """Validate that the directory name contains only alphanumeric characters, dashes, underscores, and slashes.

    Args:
        name (str): The directory name to validate.

    Returns:
        bool: True if the name is valid, False otherwise.
    """
    pattern = r'^[\w\-/]+$'
    is_valid = re.match(pattern, name) is not None
    return is_valid


def fpk_update(flatpack_name: str, session: requests.Session, branch: str = "main"):
    """
    Update files of the specified flatpack with the latest versions from the template.
    Files are placed in the /build directory, overwriting existing ones.

    Args:
        flatpack_name (str): The name of the flatpack to update.
        session (requests.Session): The HTTP session for making requests.
        branch (str): The branch to fetch files from. Defaults to "main".

    Returns:
        None
    """
    files_to_update = ['device.sh', 'index.html']

    flatpack_dir = Path.cwd() / flatpack_name
    if not flatpack_dir.exists() or not flatpack_dir.is_dir():
        logger.error(f"The flatpack '{flatpack_name}' does not exist or is not a directory.")
        print(f"[ERROR] The flatpack '{flatpack_name}' does not exist or is not a directory.")
        return

    build_dir = flatpack_dir / 'build'
    if not build_dir.exists():
        build_dir.mkdir(parents=True)
        logger.info(f"Created build directory for flatpack '{flatpack_name}'")
        print(f"[INFO] Created build directory for flatpack '{flatpack_name}'")

    for file in files_to_update:
        file_url = f"{TEMPLATE_REPO_URL}/contents/{file}?ref={branch}"
        local_file_path = build_dir / file

        try:
            response = session.get(file_url)
            response.raise_for_status()
            file_data = response.json()

            if 'content' in file_data:
                content = base64.b64decode(file_data['content']).decode('utf-8')

                with open(local_file_path, 'w') as local_file:
                    local_file.write(content)

                if local_file_path.exists():
                    logger.info(f"Replaced existing {file} in flatpack '{flatpack_name}/build'")
                    print(f"[INFO] Replaced existing {file} in flatpack '{flatpack_name}/build'")
                else:
                    logger.info(f"Added new {file} to flatpack '{flatpack_name}/build'")
                    print(f"[INFO] Added new {file} to flatpack '{flatpack_name}/build'")
            else:
                logger.error(f"Failed to retrieve content for {file}")
                print(f"[ERROR] Failed to retrieve content for {file}")

        except requests.RequestException as e:
            logger.error(f"Failed to update {file}: {e}")
            print(f"[ERROR] Failed to update {file}: {e}")

    logger.info(f"Flatpack '{flatpack_name}' update completed.")
    print(f"[INFO] Flatpack '{flatpack_name}' update completed.")


def fpk_verify(directory: Union[str, None]):
    """Verify a flatpack.

    Args:
        directory (Union[str, None]): The directory to use for verification. If None, a cached directory will be used if available.

    Returns:
        None
    """
    cache_file_path = HOME_DIR / ".fpk_unbox.cache"
    logger.info("Looking for cached flatpack in %s", cache_file_path)
    print(f"[INFO] Looking for cached flatpack in {cache_file_path}.")

    last_unboxed_flatpack = None

    if directory and fpk_valid_directory_name(directory):
        logger.info("Using provided directory: %s", directory)
        print(f"[INFO] Using provided directory: {directory}")
        last_unboxed_flatpack = directory
    elif cache_file_path.exists():
        logger.info("Found cached flatpack in %s", cache_file_path)
        print(f"[INFO] Found cached flatpack in {cache_file_path}.")
        last_unboxed_flatpack = cache_file_path.read_text().strip()
    else:
        message = "No cached flatpack found, and no valid directory provided."
        logger.error(message)
        print(f"[ERROR] {message}")
        return

    if not last_unboxed_flatpack:
        message = "No valid flatpack directory found."
        logger.error(message)
        print(f"[ERROR] {message}")
        return

    verification_script_path = Path(last_unboxed_flatpack) / 'build' / 'build.sh'

    if not verification_script_path.exists() or not verification_script_path.is_file():
        message = f"Verification script not found in {last_unboxed_flatpack}."
        logger.error(message)
        print(f"[ERROR] {message}")
        return

    safe_script_path = shlex.quote(str(verification_script_path.resolve()))

    try:
        env_vars = {'VERIFY_MODE': 'true'}
        subprocess.run(
            ['/bin/bash', '-u', safe_script_path],
            check=True,
            env={**env_vars, **os.environ}
        )
        logger.info("Verification script executed successfully.")
        print("[INFO] Verification script executed successfully.")
    except subprocess.CalledProcessError as e:
        message = f"An error occurred while executing the verification script: {e}"
        logger.error(message)
        print(f"[ERROR] {message}")
    except Exception as e:
        message = f"An unexpected error occurred: {e}"
        logger.error(message)
        print(f"[ERROR] {message}")


def setup_arg_parser():
    """
    Set up the argument parser for the Flatpack command line interface.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description='Flatpack command line interface'
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands'
    )

    # General commands
    parser_find = subparsers.add_parser(
        'find',
        help='Find model files in the current directory'
    )

    parser_find.set_defaults(
        func=fpk_cli_handle_find
    )

    parser_help = subparsers.add_parser(
        'help',
        help='Display help for commands'
    )

    parser_help.set_defaults(
        func=fpk_cli_handle_help
    )

    parser_list = subparsers.add_parser(
        'list',
        help='List available flatpack directories'
    )

    parser_list.set_defaults(
        func=fpk_cli_handle_list
    )

    parser_version = subparsers.add_parser(
        'version',
        help='Display the version of flatpack'
    )

    parser_version.set_defaults(
        func=fpk_cli_handle_version
    )

    # Agents
    parser_agents = subparsers.add_parser(
        'agents',
        help='Manage agents'
    )

    agent_subparsers = parser_agents.add_subparsers(
        dest='agent_command',
        help='Available agent commands'
    )

    ## Add command to list active agents
    parser_list_agents = agent_subparsers.add_parser(
        'list',
        help='List active agents'
    )

    parser_list_agents.set_defaults(
        func=fpk_cli_handle_list_agents
    )

    ## Add command to spawn an agent
    parser_spawn = agent_subparsers.add_parser(
        'spawn',
        help='Spawn a new agent with a script'
    )

    parser_spawn.add_argument(
        'script_path',
        type=str,
        help='Path to the script to execute'
    )

    parser_spawn.set_defaults(
        func=fpk_cli_handle_spawn_agent
    )

    ## Add command to terminate an agent
    parser_terminate = agent_subparsers.add_parser(
        'terminate',
        help='Terminate an active agent'
    )

    parser_terminate.add_argument(
        'pid',
        type=int,
        help='Process ID of the agent to terminate'
    )

    parser_terminate.set_defaults(
        func=fpk_cli_handle_terminate_agent
    )

    # API Key management
    parser_api_key = subparsers.add_parser(
        'api-key',
        help='API key management commands'
    )

    api_key_subparsers = parser_api_key.add_subparsers(
        dest='api_key_command'
    )

    ## Get API key
    parser_get_api = api_key_subparsers.add_parser(
        'get',
        help='Get the current API key'
    )

    parser_get_api.set_defaults(
        func=fpk_cli_handle_get_api_key
    )

    ## Set API key
    parser_set_api = api_key_subparsers.add_parser(
        'set',
        help='Set the API key'
    )

    parser_set_api.add_argument(
        'api_key',
        type=str,
        help='API key to set'
    )

    parser_set_api.set_defaults(
        func=fpk_cli_handle_set_api_key
    )

    # Build commands
    parser_build = subparsers.add_parser(
        'build',
        help='Build a flatpack'
    )

    parser_build.add_argument(
        'directory',
        nargs='?',
        default=None,
        help='The directory of the flatpack to build'
    )

    parser_build.add_argument(
        '--use-euxo',
        action='store_true',
        help="Use 'set -euxo pipefail' in the shell script (default is 'set -euo pipefail')"
    )

    parser_build.set_defaults(
        func=fpk_cli_handle_build
    )

    # Create flatpack
    parser_create = subparsers.add_parser(
        'create',
        help='Create a new flatpack'
    )

    parser_create.add_argument(
        'input',
        nargs='?',
        default=None,
        help='The name of the flatpack to create'
    )

    parser_create.set_defaults(
        func=fpk_cli_handle_create
    )

    # Model compression
    parser_compress = subparsers.add_parser(
        'compress',
        help='Compress a model for deployment'
    )

    parser_compress.add_argument(
        'model_id',
        type=str,
        help='The name of the Hugging Face repository (format: username/repo_name)'
    )

    parser_compress.add_argument(
        '--token',
        type=str,
        default=None,
        help='Hugging Face token for private repositories'
    )

    parser_compress.set_defaults(
        func=fpk_cli_handle_compress
    )

    # Run server
    parser_run = subparsers.add_parser(
        'run',
        help='Run the FastAPI server'
    )

    parser_run.add_argument(
        'input',
        nargs='?',
        default=None,
        help='The name of the flatpack to run'
    )

    parser_run.add_argument(
        '--share',
        action='store_true',
        help='Share using ngrok'
    )

    parser_run.set_defaults(
        func=fpk_cli_handle_run
    )

    # Unbox commands
    parser_unbox = subparsers.add_parser(
        'unbox',
        help='Unbox a flatpack from GitHub or a local directory'
    )

    parser_unbox.add_argument(
        'input',
        nargs='?',
        default=None,
        help='The name of the flatpack to unbox'
    )

    parser_unbox.add_argument(
        '--local',
        action='store_true',
        help='Unbox from a local directory instead of GitHub'
    )

    parser_unbox.set_defaults(
        func=fpk_cli_handle_unbox
    )

    # Update flatpack
    parser_update = subparsers.add_parser(
        'update',
        help='Update index.html and device.sh of a flatpack from the template'
    )

    parser_update.add_argument(
        'flatpack_name',
        help='The name of the flatpack to update'
    )

    parser_update.set_defaults(
        func=fpk_cli_handle_update
    )

    # Vector database management
    parser_vector = subparsers.add_parser(
        'vector',
        help='Vector database management'
    )

    vector_subparsers = parser_vector.add_subparsers(
        dest='vector_command'
    )

    ## Add PDF
    parser_add_pdf = vector_subparsers.add_parser(
        'add-pdf',
        help='Add text from a PDF file to the vector database'
    )

    parser_add_pdf.add_argument(
        'pdf_path',
        help='Path to the PDF file to add'
    )

    parser_add_pdf.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory path for storing the vector database and metadata files'
    )

    parser_add_pdf.set_defaults(
        func=fpk_cli_handle_vector_commands
    )

    ## Add texts
    parser_add_text = vector_subparsers.add_parser(
        'add-texts',
        help='Add new texts to generate embeddings and store them'
    )

    parser_add_text.add_argument(
        'texts',
        nargs='+',
        help='Texts to add'
    )

    parser_add_text.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory path for storing the vector database and metadata files'
    )

    parser_add_text.set_defaults(
        func=fpk_cli_handle_vector_commands
    )

    ## Add URL
    parser_add_url = vector_subparsers.add_parser(
        'add-url',
        help='Add text from a URL to the vector database'
    )

    parser_add_url.add_argument(
        'url',
        help='URL to add'
    )

    parser_add_url.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory path for storing the vector database and metadata files'
    )

    parser_add_url.set_defaults(
        func=fpk_cli_handle_vector_commands
    )

    ## Add Wikipedia
    parser_add_wikipedia_page = vector_subparsers.add_parser(
        'add-wikipedia',
        help='Add text from a Wikipedia page to the vector database'
    )

    parser_add_wikipedia_page.add_argument(
        'page_title',
        help='The title of the Wikipedia page to add'
    )

    parser_add_wikipedia_page.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory path for storing the vector database and metadata files'
    )

    parser_add_wikipedia_page.set_defaults(
        func=fpk_cli_handle_vector_commands
    )

    ## Search text
    parser_search_text = vector_subparsers.add_parser(
        'search-text',
        help='Search for texts similar to the given query'
    )

    parser_search_text.add_argument(
        'query',
        help='Text query to search for'
    )

    parser_search_text.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory path for storing the vector database and metadata files'
    )

    parser_search_text.set_defaults(
        func=fpk_cli_handle_vector_commands
    )

    # Verify commands
    parser_verify = subparsers.add_parser(
        'verify',
        help='Verify a flatpack'
    )

    parser_verify.add_argument(
        'directory',
        nargs='?',
        default=None,
        help='The directory of the flatpack to verify'
    )

    parser_verify.set_defaults(
        func=fpk_cli_handle_verify
    )

    return parser


def fpk_cli_handle_add_pdf(pdf_path, vm):
    """
    Handle the addition of a PDF file to the vector database.

    Args:
        pdf_path (str): The path to the PDF file to add.
        vm: The vector manager instance.

    Returns:
        None
    """
    if not os.path.exists(pdf_path):
        logger.error("PDF file does not exist: '%s'", pdf_path)
        print(f"[ERROR] PDF file does not exist: '{pdf_path}'.")
        return

    try:
        vm.add_pdf(pdf_path, pdf_path)
        logger.info("Added text from PDF: '%s' to the vector database.", pdf_path)
        print(f"[INFO] Added text from PDF: '{pdf_path}' to the vector database.")
    except Exception as e:
        logger.error("Failed to add PDF to the vector database: %s", e)
        print(f"[ERROR] Failed to add PDF to the vector database: {e}")


def fpk_cli_handle_add_url(url, vm):
    """
    Handle the addition of text from a URL to the vector database.

    Args:
        url (str): The URL to add.
        vm: The vector manager instance.

    Returns:
        None
    """
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if 200 <= response.status_code < 400:
            vm.add_url(url)
            logger.info("Added text from URL: '%s' to the vector database.", url)
            print(f"[INFO] Added text from URL: '{url}' to the vector database.")
        else:
            logger.error("URL is not accessible: '%s'. HTTP Status Code: %d", url, response.status_code)
            print(f"[ERROR] URL is not accessible: '{url}'. HTTP Status Code: {response.status_code}")
    except requests.RequestException as e:
        logger.error("Failed to access URL: '%s'. Error: %s", url, e)
        print(f"[ERROR] Failed to access URL: '{url}'. Error: {e}")


def fpk_cli_handle_build(args, session):
    """
    Handle the build command for the flatpack CLI.

    Args:
        args: The command-line arguments.
        session: The HTTP session.

    Returns:
        None
    """
    directory_name = args.directory

    if directory_name is None:
        logger.info("No directory name provided. Using cached directory if available.")
        print("[INFO] No directory name provided. Using cached directory if available.")

    fpk_build(directory_name, use_euxo=args.use_euxo)


def fpk_cli_handle_create(args, session):
    """
    Handle the create command for the flatpack CLI.

    Args:
        args: The command-line arguments.
        session: The HTTP session.

    Returns:
        None
    """
    if not args.input:
        logger.error("No flatpack name specified.")
        print("[ERROR] Please specify a name for the new flatpack.")
        return

    flatpack_name = args.input

    if not fpk_valid_directory_name(flatpack_name):
        logger.error("Invalid flatpack name: '%s'.", flatpack_name)
        print(f"[ERROR] Invalid flatpack name: '{flatpack_name}'.")
        return

    try:
        fpk_create(flatpack_name)
        logger.info("Flatpack '%s' created successfully.", flatpack_name)
    except Exception as e:
        logger.error("Failed to create flatpack: %s", e)
        print(f"[ERROR] Failed to create flatpack: {e}")


def fpk_cli_handle_compress(args, session: httpx.Client):
    """
    Handle the model compression command for the Flatpack CLI.

    Args:
        args: The command-line arguments.
        session: The HTTP session.

    Returns:
        None
    """
    model_id = args.model_id
    token = args.token

    if not re.match(r'^[\w-]+/[\w.-]+$', model_id):
        logger.error("Invalid Hugging Face repository format specified.")
        print("[ERROR] Please specify a valid Hugging Face repository in the format 'username/repo_name'.")
        return

    repo_name = model_id.split('/')[-1]
    local_dir = repo_name

    if os.path.exists(local_dir):
        logger.info("The model '%s' is already downloaded in the directory '%s'.", model_id, local_dir)
        print(f"[INFO] The model '{model_id}' is already downloaded in the directory '{local_dir}'.")
    else:
        try:
            if token:
                logger.info("Downloading model '%s' with provided token...", model_id)
                print(f"[INFO] Downloading model '{model_id}' with provided token...")
                snapshot_download(
                    repo_id=model_id,
                    local_dir=local_dir,
                    revision="main",
                    token=token
                )
            else:
                logger.info("Downloading model '%s'...", model_id)
                print(f"[INFO] Downloading model '{model_id}'...")
                snapshot_download(
                    repo_id=model_id,
                    local_dir=local_dir,
                    revision="main"
                )
            logger.info("Finished downloading %s into the directory '%s'", model_id, local_dir)
            print(f"[INFO] Finished downloading {model_id} into the directory '{local_dir}'")
        except Exception as e:
            logger.error("Failed to download the model. Error: %s", e)
            print(f"[ERROR] Failed to download the model. Error: {e}")
            return

    llama_cpp_dir = "llama.cpp"
    ready_file = os.path.join(llama_cpp_dir, "ready")
    requirements_file = os.path.join(llama_cpp_dir, "requirements.txt")

    venv_dir = os.path.join(llama_cpp_dir, "venv")
    venv_python = os.path.join(venv_dir, "bin", "python")

    convert_script = os.path.join(llama_cpp_dir, 'convert_hf_to_gguf.py')

    if not os.path.exists(llama_cpp_dir):

        git_executable = shutil.which("git")

        if not git_executable:
            logger.error("The 'git' executable was not found in your PATH.")
            print("[ERROR] The 'git' executable was not found in your PATH.")
            return

        try:
            logger.info("Cloning llama.cpp repository...")
            print("[INFO] Cloning llama.cpp repository...")

            subprocess.run(
                [
                    git_executable,
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/ggerganov/llama.cpp",
                    llama_cpp_dir
                ],
                check=True
            )

            logger.info("Finished cloning llama.cpp repository into '%s'", llama_cpp_dir)
            print(f"[INFO] Finished cloning llama.cpp repository into '{llama_cpp_dir}'")

        except subprocess.CalledProcessError as e:
            logger.error("Failed to clone the llama.cpp repository. Error: %s", e)
            print(f"[ERROR] Failed to clone the llama.cpp repository. Error: {e}")
            return

    if not os.path.exists(ready_file):
        try:
            logger.info("Running 'make' in the llama.cpp directory...")
            print("[INFO] Running 'make' in the llama.cpp directory...")

            make_executable = shutil.which("make")

            if not make_executable:
                logger.error("'make' executable not found in PATH.")
                print("[ERROR] 'make' executable not found in PATH.")
                return

            subprocess.run(
                [make_executable],
                cwd=llama_cpp_dir,
                check=True
            )

            logger.info("Finished running 'make' in the llama.cpp directory")
            print("[INFO] Finished running 'make' in the llama.cpp directory")

            if not os.path.exists(venv_dir):
                logger.info("Creating virtual environment in '%s'...", venv_dir)
                print(f"[INFO] Creating virtual environment in '{venv_dir}'...")
                create_venv(venv_dir)
                logger.info("Virtual environment created.")
                print("[INFO] Virtual environment created.")
            else:
                logger.info("Virtual environment already exists in '%s'", venv_dir)
                print(f"[INFO] Virtual environment already exists in '{venv_dir}'")

            logger.info("Installing llama.cpp dependencies in virtual environment...")
            print("[INFO] Installing llama.cpp dependencies in virtual environment...")

            pip_command = [
                "/bin/bash", "-c",
                (
                    f"source {shlex.quote(os.path.join(venv_dir, 'bin', 'activate'))} && "
                    f"pip install -r {shlex.quote(requirements_file)}"
                )
            ]
            subprocess.run(pip_command, check=True)

            logger.info("Finished installing llama.cpp dependencies")
            print("[INFO] Finished installing llama.cpp dependencies")

            with open(ready_file, 'w') as f:
                f.write("Ready")
        except subprocess.CalledProcessError as e:
            logger.error("Failed to build llama.cpp. Error: %s", e)
            print(f"[ERROR] Failed to build llama.cpp. Error: {e}")
            return
        except Exception as e:
            logger.error("An error occurred during the setup of llama.cpp. Error: %s", e)
            print(f"[ERROR] An error occurred during the setup of llama.cpp. Error: {e}")
            return

    output_file = os.path.join(local_dir, f"{repo_name}-fp16.bin")
    quantized_output_file = os.path.join(local_dir, f"{repo_name}-Q4_K_S.gguf")
    outtype = "f16"

    if not os.path.exists(convert_script):
        logger.error("The conversion script '%s' does not exist.", convert_script)
        print(f"[ERROR] The conversion script '{convert_script}' does not exist.")
        return

    if not os.path.exists(output_file):
        try:
            logger.info("Converting the model using llama.cpp...")
            print("[INFO] Converting the model using llama.cpp...")

            venv_activate = os.path.join(venv_dir, "bin", "activate")

            convert_command = [
                "/bin/bash", "-c",
                (
                    f"source {shlex.quote(venv_activate)} && {shlex.quote(venv_python)} "
                    f"{shlex.quote(convert_script)} {shlex.quote(local_dir)} --outfile "
                    f"{shlex.quote(output_file)} --outtype {shlex.quote(outtype)}"
                )
            ]
            print(f"[DEBUG] Running command: {convert_command}")
            subprocess.run(convert_command, check=True)

            logger.info("Conversion complete. The model has been compressed and saved as '%s'", output_file)
            print(f"[INFO] Conversion complete. The model has been compressed and saved as '{output_file}'")
        except subprocess.CalledProcessError as e:
            logger.error("Conversion failed. Error: %s", e)
            print(f"[ERROR] Conversion failed. Error: {e}")
            return
        except Exception as e:
            logger.error("An error occurred during the model conversion. Error: %s", e)
            print(f"[ERROR] An error occurred during the model conversion. Error: {e}")
            return
    else:
        logger.info("The model has already been converted and saved as '%s'.", output_file)
        print(f"[INFO] The model has already been converted and saved as '{output_file}'.")

    if os.path.exists(output_file):
        try:
            logger.info("Quantizing the model...")
            print("[INFO] Quantizing the model...")

            quantize_command = [
                os.path.join(llama_cpp_dir, 'llama-quantize'),
                output_file,
                quantized_output_file,
                "Q4_K_S"
            ]
            subprocess.run(quantize_command, check=True)

            logger.info("Quantization complete. The quantized model has been saved as '%s'.", quantized_output_file)
            print(f"[INFO] Quantization complete. The quantized model has been saved as '{quantized_output_file}'.")

            logger.info("Deleting the original .bin file '%s'...", output_file)
            print(f"[INFO] Deleting the original .bin file '{output_file}'...")

            os.remove(output_file)

            logger.info("Deleted the original .bin file '%s'.", output_file)
            print(f"[INFO] Deleted the original .bin file '{output_file}'.")

        except subprocess.CalledProcessError as e:
            logger.error("Quantization failed. Error: %s", e)
            print(f"[ERROR] Quantization failed. Error: {e}")
            return
        except Exception as e:
            logger.error("An error occurred during the quantization process. Error: %s", e)
            print(f"[ERROR] An error occurred during the quantization process. Error: {e}")
            return
    else:
        logger.error("The original model file '%s' does not exist.", output_file)
        print(f"[ERROR] The original model file '{output_file}' does not exist.")


def fpk_cli_handle_find(args, session):
    """Handle the 'find' command to search for model files."""
    logger.info("Searching for files...")
    model_files = fpk_find_models()
    if model_files:
        logger.info("Found the following files:")
        print("[INFO] Found the following files:")
        for model_file in model_files:
            logger.info(" - %s", model_file)
            print(f" - {model_file}")
    else:
        logger.info("No files found.")
        print("[INFO] No files found.")


def fpk_cli_handle_get_api_key(args, session):
    """Handle the 'get' command to retrieve the API key."""
    logger.info("Retrieving API key...")
    api_key = fpk_get_api_key()
    if api_key:
        logger.info("API Key: %s", api_key)
        print(f"API Key: {api_key}")
    else:
        logger.error("No API key found.")
        print("[ERROR] No API key found.")


def fpk_cli_handle_help(args, session):
    """Handle the 'help' command to display the help message."""
    parser = setup_arg_parser()
    if args.command:
        subparser = parser._subparsers._actions[1].choices.get(args.command)
        if subparser:
            subparser.print_help()
            logger.info("Displayed help for command '%s'.", args.command)
        else:
            logger.error("Command '%s' not found.", args.command)
            print(f"[ERROR] Command '{args.command}' not found.")
    else:
        parser.print_help()
        logger.info("Displayed general help.")


def fpk_cli_handle_list(args, session):
    """Handle the 'list' command to fetch and print the list of directories."""
    directories = fpk_list_directories(session)
    if directories:
        table = PrettyTable()
        table.field_names = ["Index", "Directory Name"]
        table.align["Index"] = "r"
        table.align["Directory Name"] = "l"

        for index, directory in enumerate(directories.split('\n'), start=1):
            table.add_row([index, directory])

        print(table)
        logger.info("Directories found: %s", directories)
    else:
        logger.error("No directories found.")
        print("[ERROR] No directories found.")


def fpk_cli_handle_list_agents(args, session):
    """List active agents."""
    agent_manager = AgentManager()
    agents = agent_manager.list_agents()

    if agents:
        logger.info("Active agents:")
        print("Active agents:")
        for agent in agents:
            agent_info = (f"PID: {agent['pid']}, Script: {agent['script']}, "
                          f"Start Time: {agent['start_time']}, Port: {agent['port']}")
            logger.info(agent_info)
            print(agent_info)
    else:
        logger.info("No active agents found.")
        print("[INFO] No active agents found.")


atexit.register(fpk_safe_cleanup)

app = FastAPI()

uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.addFilter(EndpointFilter())

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

flatpack_directory = None


@app.on_event("startup")
async def startup_event():
    global SERVER_START_TIME
    SERVER_START_TIME = datetime.now(timezone.utc)
    logger.info("Server started at %s. Cooldown period: %s", SERVER_START_TIME, COOLDOWN_PERIOD)
    print(f"[INFO] Server started at {SERVER_START_TIME}. Cooldown period: {COOLDOWN_PERIOD}")
    asyncio.create_task(run_scheduler())


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = Path(flatpack_directory) / "build" / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    else:
        return Response(status_code=204)


@app.get("/load_file")
async def load_file(
        request: Request,
        filename: str,
        token: str = Depends(authenticate_token)
):
    """Load and return the contents of a specified file from the flatpack directory."""
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    file_path = os.path.join(flatpack_directory, 'build', filename)

    if not os.path.commonpath([flatpack_directory, os.path.realpath(file_path)]).startswith(
            os.path.realpath(flatpack_directory)):
        raise HTTPException(status_code=403, detail="Access to the requested file is forbidden")

    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                unescaped_content = unescape_content_parts(content)
                return JSONResponse(content={"content": unescaped_content})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {e}")
    else:
        raise HTTPException(status_code=404, detail="File not found")


@app.post("/save_file")
async def save_file(
        request: Request,
        filename: str = Form(...),
        content: str = Form(...),
        token: str = Depends(authenticate_token)
):
    """Save the provided content to the specified file within the flatpack directory."""
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    file_path = os.path.join(flatpack_directory, 'build', filename)

    if not os.path.commonpath([flatpack_directory, os.path.realpath(file_path)]).startswith(
            os.path.realpath(flatpack_directory)):
        raise HTTPException(status_code=403, detail="Access to the requested file is forbidden")

    try:
        normalized_content = content.replace('\r\n', '\n').replace('\r', '\n')
        escaped_content = escape_content_parts(normalized_content)
        with open(file_path, 'w', newline='\n') as file:
            file.write(escaped_content)
        return JSONResponse(content={"message": "File saved successfully!"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")


@app.get("/test", response_class=HTMLResponse)
async def test():
    return """
    <html>
        <head>
            <title>Hello, World!</title>
        </head>
        <body>
            <h1>Hello, World!</h1>
        </body>
    </html>
    """


@app.get("/test_db")
async def test_db():
    db_path = os.path.join(flatpack_directory, 'build', 'flatpack.db')
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        conn.close()
        return {"message": "Database connection successful"}
    except Error as e:
        logger.error("Database connection failed: %s", e)
        return {"message": f"Database connection failed: {e}"}


@app.get("/api/build-status")
async def get_build_status(token: str = Depends(authenticate_token)):
    """Get the current build status."""
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    status_file = os.path.join(flatpack_directory, 'build', 'build_status.json')

    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status_data = json.load(f)
        return JSONResponse(content=status_data)
    else:
        return JSONResponse(content={"status": "no_builds"})


@app.post("/api/clear-build-status")
async def clear_build_status(token: str = Depends(authenticate_token)):
    """Clear the current build status."""
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    status_file = os.path.join(flatpack_directory, 'build', 'build_status.json')

    try:
        if os.path.exists(status_file):
            os.remove(status_file)
        return JSONResponse(content={"message": "Build status cleared successfully."})
    except Exception as e:
        logger.error("Error clearing build status: %s", e)
        raise HTTPException(status_code=500, detail=f"Error clearing build status: {e}")


@app.post("/api/comments")
async def add_comment(comment: Comment, token: str = Depends(authenticate_token)):
    """Add a new comment to the database."""
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    db_path = os.path.join(flatpack_directory, 'build', 'flatpack.db')

    try:
        ensure_database_initialized()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO flatpack_comments (block_id, selected_text, comment)
            VALUES (?, ?, ?)
        """, (comment.block_id, comment.selected_text, comment.comment))

        comment_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return JSONResponse(content={
            "message": "Comment added successfully.",
            "comment_id": comment_id,
            "created_at": datetime.utcnow().isoformat()
        }, status_code=201)

    except sqlite3.Error as e:
        logger.error("An error occurred while adding the comment: %s", e)
        raise HTTPException(status_code=500, detail=f"An error occurred while adding the comment: {e}")


@app.delete("/api/comments/{comment_id}")
async def delete_comment(comment_id: int, token: str = Depends(authenticate_token)):
    """Delete a comment from the database by its ID."""
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    db_path = os.path.join(flatpack_directory, 'build', 'flatpack.db')

    try:
        ensure_database_initialized()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM flatpack_comments WHERE id = ?", (comment_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Comment not found")
        conn.commit()
        conn.close()

        return JSONResponse(content={"message": "Comment deleted successfully."}, status_code=200)
    except Error as e:
        logger.error("An error occurred while deleting the comment: %s", e)
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the comment: {e}")


@app.get("/api/comments")
async def get_all_comments(token: str = Depends(authenticate_token)):
    """Retrieve all comments from the database."""
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    db_path = os.path.join(flatpack_directory, 'build', 'flatpack.db')

    try:
        ensure_database_initialized()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, block_id, selected_text, comment, created_at
            FROM flatpack_comments
            ORDER BY created_at DESC
        """)
        comments = cursor.fetchall()
        conn.close()

        return [{"id": row[0], "block_id": row[1], "selected_text": row[2], "comment": row[3], "created_at": row[4]} for
                row in comments]
    except Error as e:
        logger.error("An error occurred while retrieving comments: %s", e)
        raise HTTPException(status_code=500, detail=f"An error occurred while retrieving comments: {e}")


@app.post("/api/build")
async def build_flatpack(
        request: Request,
        background_tasks: BackgroundTasks,
        token: str = Depends(authenticate_token)
):
    """Trigger the build process for the flatpack."""
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    try:
        background_tasks.add_task(run_build_process, schedule_id=None)
        logger.info("Started build process for flatpack located at %s", flatpack_directory)
        return JSONResponse(
            content={"flatpack": flatpack_directory, "message": "Build process started in background."},
            status_code=200)
    except Exception as e:
        logger.error("Failed to start build process: %s", e)
        return JSONResponse(content={"flatpack": flatpack_directory, "message": f"Failed to start build process: {e}"},
                            status_code=500)


@app.get("/api/heartbeat")
async def heartbeat():
    """Endpoint to check the server heartbeat."""
    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    return JSONResponse(content={"server_time": current_time}, status_code=200)


@app.post("/api/hooks")
async def add_hook(hook: Hook, token: str = Depends(authenticate_token)):
    try:
        response = add_hook_to_database(hook)
        return JSONResponse(content=response, status_code=201)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Failed to add hook: %s", e)
        raise HTTPException(status_code=500, detail="Failed to add hook.")


@app.delete("/api/hooks/{hook_id}")
async def delete_hook(hook_id: int, token: str = Depends(authenticate_token)):
    """Delete a hook from the database by its ID."""
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    db_path = os.path.join(flatpack_directory, 'build', 'flatpack.db')

    try:
        ensure_database_initialized()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM flatpack_hooks WHERE id = ?", (hook_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Hook not found")
        conn.commit()
        conn.close()

        return JSONResponse(content={"message": "Hook deleted successfully."}, status_code=200)
    except Error as e:
        logger.error("An error occurred while deleting the hook: %s", e)
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the hook: {e}")


@app.get("/api/hooks", response_model=List[Hook])
async def get_hooks(token: str = Depends(authenticate_token)):
    try:
        hooks = get_all_hooks_from_database()
        return JSONResponse(content={"hooks": hooks}, status_code=200)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error("Failed to retrieve hooks: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve hooks.")


@app.get("/api/logs")
async def get_all_logs(request: Request, token: str = Depends(authenticate_token)):
    """Get a list of all available logs ordered by date."""
    global flatpack_directory

    if flatpack_directory:
        logs_directory = Path(flatpack_directory) / 'build' / 'logs'
    else:
        cache_file_path = HOME_DIR / ".fpk_unbox.cache"
        if cache_file_path.exists():
            last_unboxed_flatpack = cache_file_path.read_text().strip()
            logs_directory = Path(last_unboxed_flatpack) / 'build' / 'logs'
        else:
            raise HTTPException(status_code=500, detail="No cached flatpack directory found")

    try:
        if not logs_directory.exists():
            logs_directory.mkdir(parents=True, exist_ok=True)
            if not logs_directory.exists():
                raise HTTPException(status_code=500, detail="Failed to create logs directory")

        log_files = sorted(
            [f for f in os.listdir(logs_directory) if f.startswith("build_") and f.endswith(".log")],
            key=lambda x: datetime.strptime(x, "build_%Y_%m_%d_%H_%M_%S.log"),
            reverse=True
        )

        return JSONResponse(content={"logs": log_files}, status_code=200)

    except HTTPException:
        raise

    except Exception as e:
        logger.error("Failed to list log files: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to list log files: {e}")


@app.get("/api/logs/{log_filename}")
async def get_log_file(request: Request, log_filename: str, token: str = Depends(authenticate_token)):
    """Get the content of a specific log file."""
    global flatpack_directory

    if flatpack_directory:
        logs_directory = Path(flatpack_directory) / 'build' / 'logs'
    else:
        cache_file_path = HOME_DIR / ".fpk_unbox.cache"
        if cache_file_path.exists():
            last_unboxed_flatpack = cache_file_path.read_text().strip()
            logs_directory = Path(last_unboxed_flatpack) / 'build' / 'logs'
        else:
            raise HTTPException(status_code=500, detail="No cached flatpack directory found")

    log_path = logs_directory / log_filename
    if log_path.exists() and log_path.is_file():
        try:
            with open(log_path, 'r') as file:
                content = file.read()
            return JSONResponse(content={"log": content}, status_code=200)
        except Exception as e:
            logger.error("Error reading log file '%s': %s", log_filename, e)
            raise HTTPException(status_code=500, detail=f"Error reading log file: {e}")
    else:
        raise HTTPException(status_code=404, detail="Log file not found")


@app.delete("/api/schedule/{schedule_id}")
async def delete_schedule_entry(schedule_id: int, datetime_index: Optional[int] = None,
                                token: str = Depends(authenticate_token)):
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    db_path = os.path.join(flatpack_directory, 'build', 'flatpack.db')

    try:
        ensure_database_initialized()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, type, datetimes FROM flatpack_schedule WHERE id = ?", (schedule_id,))
        schedule = cursor.fetchone()

        if not schedule:
            raise HTTPException(status_code=404, detail="Schedule not found")

        schedule_id, schedule_type, datetimes_json = schedule

        if datetime_index is not None and schedule_type == 'manual':
            datetimes = json.loads(datetimes_json)
            if 0 <= datetime_index < len(datetimes):
                del datetimes[datetime_index]
                cursor.execute("UPDATE flatpack_schedule SET datetimes = ? WHERE id = ?",
                               (json.dumps(datetimes), schedule_id))
                conn.commit()
                return JSONResponse(content={"message": "Schedule datetime entry deleted successfully."},
                                    status_code=200)
            else:
                raise HTTPException(status_code=404, detail="Datetime entry not found")
        else:
            cursor.execute("DELETE FROM flatpack_schedule WHERE id = ?", (schedule_id,))
            conn.commit()
            return JSONResponse(content={"message": "Entire schedule deleted successfully."}, status_code=200)

    except sqlite3.Error as e:
        logger.error("A database error occurred while deleting the schedule entry: %s", e)
        raise HTTPException(status_code=500, detail=f"A database error occurred while deleting the schedule entry: {e}")
    except json.JSONDecodeError as e:
        logger.error("A JSON decoding error occurred: %s", e)
        raise HTTPException(status_code=500, detail=f"An error occurred while processing schedule data: {e}")
    except Exception as e:
        logger.error("An unexpected error occurred while deleting the schedule entry: %s", e)
        raise HTTPException(status_code=500,
                            detail=f"An unexpected error occurred while deleting the schedule entry: {e}")
    finally:
        if 'conn' in locals():
            conn.close()


@app.get("/api/schedule")
async def get_schedule(token: str = Depends(authenticate_token)):
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    db_path = os.path.join(flatpack_directory, 'build', 'flatpack.db')

    try:
        ensure_database_initialized()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id, type, pattern, datetimes, last_run FROM flatpack_schedule ORDER BY created_at DESC")
        results = cursor.fetchall()
        conn.close()

        if results:
            schedules = []
            for schedule_id, schedule_type, pattern, datetimes, last_run in results:
                schedules.append({
                    "id": schedule_id,
                    "type": schedule_type,
                    "pattern": pattern,
                    "datetimes": json.loads(datetimes),
                    "last_run": last_run
                })
            return JSONResponse(content={"schedules": schedules}, status_code=200)
        else:
            return JSONResponse(content={"schedules": []}, status_code=200)
    except Exception as e:
        logger.error("An error occurred while retrieving the schedules: %s", e)
        raise HTTPException(status_code=500, detail=f"An error occurred while retrieving the schedules: {e}")


@app.post("/api/schedule")
async def save_schedule(request: Request, token: str = Depends(authenticate_token)):
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    db_path = os.path.join(flatpack_directory, 'build', 'flatpack.db')

    try:
        data = await request.json()
        schedule_type = data.get('type')
        pattern = data.get('pattern')
        datetimes = data.get('datetimes', [])

        if schedule_type == 'manual':
            datetimes = [datetime.fromisoformat(dt).astimezone(timezone.utc).isoformat() for dt in datetimes]

        ensure_database_initialized()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO flatpack_schedule (type, pattern, datetimes)
            VALUES (?, ?, ?)
        """, (schedule_type, pattern, json.dumps(datetimes)))

        new_schedule_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return JSONResponse(content={"message": "Schedule saved successfully.", "id": new_schedule_id}, status_code=200)
    except Exception as e:
        logger.error("An error occurred while saving the schedule: %s", e)
        raise HTTPException(status_code=500, detail=f"An error occurred while saving the schedule: {e}")


@app.post("/api/validate_token")
async def validate_token(request: Request, api_token: str = Form(...)):
    """Validate the provided API token."""
    global VALIDATION_ATTEMPTS
    stored_token = get_token()
    if not stored_token:
        return JSONResponse(content={"message": "API token is not set."}, status_code=200)
    if validate_api_token(api_token):
        return JSONResponse(content={"message": "API token is valid."}, status_code=200)
    VALIDATION_ATTEMPTS += 1
    if VALIDATION_ATTEMPTS >= MAX_ATTEMPTS:
        shutdown_server()
    return JSONResponse(content={"message": "Invalid API token."}, status_code=401)


@app.post("/api/verify")
async def verify_flatpack(
        request: Request,
        token: str = Depends(authenticate_token)
):
    """Trigger the verification process for the flatpack."""
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    try:
        fpk_verify(flatpack_directory)
        return JSONResponse(content={"message": "Verification process completed successfully."}, status_code=200)
    except Exception as e:
        logger.error("Verification process failed: %s", e)
        return JSONResponse(content={"message": f"Verification process failed: {e}"}, status_code=500)


def fpk_cli_handle_run(args, session):
    """Handle the 'run' command to start the FastAPI server."""
    if not args.input:
        logger.error("Please specify a flatpack for the run command.")
        print("[ERROR] Please specify a flatpack for the run command.")
        return

    directory = Path(args.input).resolve()
    allowed_directory = Path.cwd()

    if not directory.is_dir() or not directory.exists():
        logger.error("The flatpack '%s' does not exist or is not a directory.", directory)
        print(f"[ERROR] The flatpack '{directory}' does not exist or is not a directory.")
        return

    if not directory.is_relative_to(allowed_directory):
        logger.error("The specified directory is not within allowed paths.")
        print("[ERROR] The specified directory is not within allowed paths.")
        return

    if args.share:
        if not os.environ.get('NGROK_AUTHTOKEN'):
            logger.error("NGROK_AUTHTOKEN environment variable is not set.")
            print("[ERROR] NGROK_AUTHTOKEN environment variable is not set.")
            return

    token = generate_secure_token()
    logger.info("API token generated and displayed to user.")
    print(f"[INFO] Generated API token: {token}")
    print("[INFO] Please save this API token securely. You will not be able to retrieve it again.")

    try:
        while True:
            confirmation = input("Have you saved the API token? Type 'YES' to continue: ").strip().upper()
            if confirmation == 'YES':
                break
            print("[INFO] Please save the API token before continuing.")
    except KeyboardInterrupt:
        print("\n[INFO] Process interrupted. Exiting.")
        return

    set_token(token)
    setup_static_directory(app, str(directory))

    try:
        host = "127.0.0.1"

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, 0))
        port = sock.getsockname()[1]
        sock.close()

        logger.info("Selected available port: %d", port)
        print(f"[INFO] Selected available port: {port}")

        if args.share:
            listener = ngrok.forward(f"{host}:{port}", authtoken_from_env=True)
            public_url = listener.url()
            logger.info("Ingress established at %s", public_url)
            print(f"[INFO] Ingress established at {public_url}")

        config = uvicorn.Config(app, host=host, port=port)
        server = uvicorn.Server(config)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(run_scheduler)

        server.run()

    except Exception as e:
        logger.exception("An unexpected error occurred during server run: %s", str(e))
        print("[ERROR] An unexpected error occurred during server run.")

    finally:
        if args.share:
            try:
                ngrok.disconnect(public_url)
                logger.info("Disconnected ngrok ingress.")
                print("[INFO] Disconnected ngrok ingress.")
            except Exception as e:
                logger.error("Failed to disconnect ngrok ingress: %s", str(e))
                print("[ERROR] Failed to disconnect ngrok ingress.")


def fpk_cli_handle_set_api_key(args, session):
    """Handle the 'set' command to set the API key."""
    logger.info("Setting API key: %s", args.api_key)
    print(f"[INFO] Setting API key: {args.api_key}")

    api_key = args.api_key
    config = load_config()
    config['api_key'] = api_key
    save_config(config)

    logger.info("API key set successfully!")
    print("[INFO] API key set successfully!")

    try:
        test_key = fpk_get_api_key()
        if test_key == api_key:
            logger.info("Verification successful: API key matches.")
            print("[INFO] Verification successful: API key matches.")
        else:
            logger.error("Verification failed: API key does not match.")
            print("[ERROR] Verification failed: API key does not match.")
    except Exception as e:
        logger.error("Error during API key verification: %s", e)
        print(f"[ERROR] Error during API key verification: {e}")


def fpk_cli_handle_spawn_agent(args, session):
    """Handle the 'spawn' command to spawn a new agent with a script.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    agent_manager = AgentManager()

    try:
        pid = agent_manager.spawn_agent(args.script_path)
        logger.info("Agent spawned with PID: %s", pid)
        print(f"[INFO] Agent spawned with PID: {pid}")
    except Exception as e:
        logger.error("Failed to spawn agent: %s", e)
        print(f"[ERROR] Failed to spawn agent: {e}")


def fpk_cli_handle_terminate_agent(args, session):
    """Handle the 'terminate' command to terminate an active agent.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    agent_manager = AgentManager()

    try:
        agent_manager.terminate_agent(args.pid)
        logger.info("Agent with PID %s terminated successfully.", args.pid)
        print(f"[INFO] Agent with PID {args.pid} terminated successfully.")
    except Exception as e:
        logger.error("Failed to terminate agent with PID %s: %s", args.pid, e)
        print(f"[ERROR] Failed to terminate agent with PID {args.pid}: {e}")


def fpk_cli_handle_unbox(args, session):
    """Handle the 'unbox' command to unbox a flatpack from GitHub or a local directory.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    if not args.input:
        logger.error("No flatpack specified for the unbox command.")
        print("[ERROR] Please specify a flatpack for the unbox command.")
        return

    directory_name = args.input
    existing_dirs = fpk_fetch_github_dirs(session)

    if directory_name not in existing_dirs and not args.local:
        logger.error("The flatpack '%s' does not exist.", directory_name)
        print(f"[ERROR] The flatpack '{directory_name}' does not exist.")
        return

    fpk_display_disclaimer(directory_name, local=args.local)

    while True:
        user_response = input().strip().upper()
        if user_response == "YES":
            break
        if user_response == "NO":
            logger.info("Installation aborted by user.")
            print("[INFO] Installation aborted by user.")
            return
        logger.error("Invalid input from user. Expected 'YES' or 'NO'.")
        print("[ERROR] Invalid input. Please type 'YES' to accept or 'NO' to decline.")

    if args.local:
        local_directory_path = Path(directory_name)
        if not local_directory_path.exists() or not local_directory_path.is_dir():
            logger.error("Local directory does not exist: '%s'.", directory_name)
            print(f"[ERROR] Local directory does not exist: '{directory_name}'.")
            return
        toml_path = local_directory_path / 'flatpack.toml'
        if not toml_path.exists():
            logger.error("flatpack.toml not found in the specified directory: '%s'.", directory_name)
            print(f"[ERROR] flatpack.toml not found in '{directory_name}'.")
            return

    logger.info("Directory name resolved to: '%s'", directory_name)
    print(f"[INFO] Directory name resolved to: '{directory_name}'")

    try:
        unbox_result = fpk_unbox(directory_name, session, local=args.local)
        if unbox_result:
            logger.info("Unboxed flatpack '%s' successfully.", directory_name)
            print(f"[INFO] Unboxed flatpack '{directory_name}' successfully.")
        else:
            logger.info("Unboxing of flatpack '%s' was aborted.", directory_name)
            print(f"[INFO] Unboxing of flatpack '{directory_name}' was aborted.")
    except Exception as e:
        logger.error("Failed to unbox flatpack '%s': %s", directory_name, e)
        print(f"[ERROR] Failed to unbox flatpack '{directory_name}': {e}")


def fpk_cli_handle_update(args, session):
    """Handle the 'update' command to update a flatpack's files from the template.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    if not args.flatpack_name:
        logger.error("No flatpack specified for the update command.")
        print("[ERROR] Please specify a flatpack for the update command.")
        return

    fpk_update(args.flatpack_name, session)


def fpk_cli_handle_vector_commands(args, session, vm):
    """Handle vector database commands.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
        vm: The Vector Manager instance.
    """
    logger.info("Handling vector commands...")
    print("[INFO] Handling vector commands...")

    if args.vector_command == 'add-texts':
        vm.add_texts(args.texts, "manual")
        logger.info("Added %d texts to the database.", len(args.texts))
        print(f"[INFO] Added {len(args.texts)} texts to the database.")
    elif args.vector_command == 'search-text':
        results = vm.search_vectors(args.query)
        if results:
            logger.info("Search results:")
            print("[INFO] Search results:")
            for result in results:
                logger.info("%s: %s", result['id'], result['text'])
                print(f"{result['id']}: {result['text']}\n")
        else:
            logger.info("No results found.")
            print("[INFO] No results found.")
    elif args.vector_command == 'add-pdf':
        fpk_cli_handle_add_pdf(args.pdf_path, vm)
    elif args.vector_command == 'add-url':
        fpk_cli_handle_add_url(args.url, vm)
    elif args.vector_command == 'add-wikipedia':
        vm.add_wikipedia_page(args.page_title)
        logger.info(
            "Added text from Wikipedia page: '%s' to the vector database.",
            args.page_title
        )
        print(
            f"[INFO] Added text from Wikipedia page: '{args.page_title}' "
            "to the vector database."
        )
    else:
        logger.error("Unknown vector command.")
        print("[ERROR] Unknown vector command.")


def fpk_cli_handle_verify(args, session):
    """Handle the 'verify' command to verify a flatpack.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    directory_name = args.directory
    if not directory_name:
        logger.error("No directory specified for the verify command.")
        print("[ERROR] Please specify a directory for the verify command.")
        return

    logger.info("Verifying flatpack in directory: %s", directory_name)
    print(f"[INFO] Verifying flatpack in directory: {directory_name}")

    try:
        fpk_verify(directory_name)
        logger.info("Verification successful for directory: %s", directory_name)
        print(f"[INFO] Verification successful for directory: {directory_name}")
    except Exception as e:
        logger.error("Verification failed for directory '%s': %s", directory_name, e)
        print(f"[ERROR] Verification failed for directory '{directory_name}': {e}")


def fpk_cli_handle_version(args, session):
    """Handle the 'version' command to display the version of flatpack.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    logger.info("Flatpack version: %s", VERSION)
    print(f"[INFO] Flatpack version: {VERSION}")


def main():
    """Main entry point for the flatpack command line interface."""
    try:
        with SessionManager() as session:
            parser = setup_arg_parser()
            args = parser.parse_args()

            vm = None
            if args.command == 'vector':
                vm = fpk_initialize_vector_manager(args)

            if hasattr(args, 'func'):
                if args.command == 'vector' and 'vector_command' in args:
                    args.func(args, session, vm)
                else:
                    args.func(args, session)
            else:
                parser.print_help()

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting gracefully...")
        sys.exit(1)
    except Exception as e:
        logger.error("An unexpected error occurred: %s", str(e))
        print(f"[ERROR] An unexpected error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
