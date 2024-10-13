import os
import sys

from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

PACKAGE_DIR = Path(sys.modules['flatpack'].__file__).parent
IMPORT_CACHE_FILE = PACKAGE_DIR / ".fpk_import_cache"

console = Console()

if not IMPORT_CACHE_FILE.exists():
    ascii_art = """ _____ __    _____ _____ _____ _____ _____ _____ 
|   __|  |  |  _  |_   _|  _  |  _  |     |  |  |
|   __|  |__|     | | | |   __|     |   --|    -|
|__|  |_____|__|__| |_| |__|  |__|__|_____|__|__|                                                                                                       
    """

    console.print(f"[bold green]{ascii_art}[/bold green]")
    console.print("[bold green]Initialising Flatpack for the first time. This may take a moment...[/bold green]")

import argparse
import asyncio
import base64
import errno
import json
import logging
import mimetypes
import pty
import random
import re
import secrets
import select
import shlex
import shutil
import signal
import socket
import stat
import string
import subprocess
import tarfile
import tempfile
import time
import traceback
import unicodedata

from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from io import BytesIO
from logging.handlers import RotatingFileHandler
from textwrap import wrap
from typing import List, Optional, Union
from zipfile import ZipFile

import httpx
import requests
import toml
import uvicorn

from fastapi import Cookie, Depends, FastAPI, Form, Header, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.security import APIKeyCookie
from itsdangerous import BadSignature, SignatureExpired, TimestampSigner
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from .database_manager import DatabaseManager
from .error_handling import safe_exit, setup_exception_handling, setup_signal_handling
from .parsers import parse_toml_to_venv_script
from .session_manager import SessionManager
from .vector_manager import VectorManager

if not IMPORT_CACHE_FILE.exists():
    IMPORT_CACHE_FILE.touch()
    console.print("")
    console.print("[bold green]First-time initialisation complete! ✨[/bold green]")
    console.print("")


def lazy_import(module_name, package=None, callable_name=None):
    import importlib
    try:
        module = importlib.import_module(module_name, package)
        if callable_name:
            return getattr(module, callable_name)
        return module
    except ImportError:
        return None


BackgroundTasks = lazy_import('fastapi', callable_name='BackgroundTasks')
BeautifulSoup = lazy_import('bs4', callable_name='BeautifulSoup')
CORSMiddleware = lazy_import('fastapi.middleware.cors', callable_name='CORSMiddleware')
croniter = lazy_import('croniter', callable_name='croniter')
Depends = lazy_import('fastapi', callable_name='Depends')
FileResponse = lazy_import('fastapi.responses', callable_name='FileResponse')
Form = lazy_import('fastapi', callable_name='Form')
HTMLResponse = lazy_import('fastapi.responses', callable_name='HTMLResponse')
HTTPException = lazy_import('fastapi', callable_name='HTTPException')
JSONResponse = lazy_import('fastapi.responses', callable_name='JSONResponse')
ngrok = lazy_import('ngrok')
PrettyTable = lazy_import('prettytable', callable_name='PrettyTable')
Request = lazy_import('fastapi', callable_name='Request')
Response = lazy_import('fastapi.responses', callable_name='Response')
snapshot_download = lazy_import('huggingface_hub', callable_name='snapshot_download')
StaticFiles = lazy_import('fastapi.staticfiles', callable_name='StaticFiles')
warnings = lazy_import('warnings')
zstd = lazy_import('zstandard')

HOME_DIR = Path.home() / ".fpk"
HOME_DIR.mkdir(exist_ok=True)

CONFIG_FILE_PATH = HOME_DIR / ".fpk_config.toml"
GITHUB_CACHE = HOME_DIR / ".fpk_github.cache"
HOOKS_FILE = "build/hooks.json"

BASE_URL = "https://raw.githubusercontent.com/RomlinGroup/Flatpack/main/warehouse"
GITHUB_REPO_URL = "https://api.github.com/repos/RomlinGroup/Flatpack"
TEMPLATE_REPO_URL = "https://api.github.com/repos/RomlinGroup/template"

VERSION = version("flatpack")

COOLDOWN_PERIOD = timedelta(minutes=1)
GITHUB_CACHE_EXPIRY = timedelta(hours=1)
SERVER_START_TIME = None

MAX_ATTEMPTS = 5
VALIDATION_ATTEMPTS = 0

CSRF_EXEMPT_PATHS = [
    "/",
    "/csrf-token",
    "/favicon.ico",
    "/static"
]

active_sessions = {}

abort_requested = False
build_in_progress = False
shutdown_requested = False

console = Console()


class Comment(BaseModel):
    block_id: str
    selected_text: str
    comment: str


class EndpointFilter(logging.Filter):
    def filter(self, record):
        return all(
            endpoint not in record.getMessage()
            for endpoint in [
                'GET /api/heartbeat',
                'GET /api/build-status',
                'GET /api/clear-build-status',
                'GET /csrf-token'
            ]
        )


class Hook(BaseModel):
    hook_name: str
    hook_placement: str
    hook_script: str
    hook_type: str


csrf_cookie = APIKeyCookie(name="csrf_token")
db_manager = None


def initialize_database_manager(flatpack_directory):
    global db_manager

    if flatpack_directory is None:
        raise ValueError("flatpack_directory is not initialized.")

    db_path = os.path.join(flatpack_directory, 'build', 'flatpack.db')

    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path))

        console.print(
            f"[bold green]SUCCESS:[/bold green] Created directory for database at path: {os.path.dirname(db_path)}"
        )

    db_manager = DatabaseManager(db_path)
    db_manager.initialize_database()


def setup_logging(log_path: Path):
    """Set up logging configuration."""
    new_logger = logging.getLogger(__name__)
    new_logger.setLevel(logging.INFO)

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


async def abort_build_process():
    global build_in_progress, abort_requested
    if build_in_progress:
        abort_requested = True
        logger.info("Build abort requested.")
        while build_in_progress:
            await asyncio.sleep(0.5)
        abort_requested = False
        logger.info("Build process aborted.")
    else:
        logger.info("No build in progress to abort.")


def add_hook_to_database(hook: Hook):
    global db_manager
    ensure_database_initialized()
    try:
        if db_manager.hook_exists(hook.hook_name):
            existing_hook = db_manager.get_hook_by_name(hook.hook_name)
            return {
                "message": "Hook with this name already exists.",
                "existing_hook": existing_hook,
                "new_hook": hook.dict()
            }
        hook_id = db_manager.add_hook(hook.hook_name, hook.hook_placement, hook.hook_script, hook.hook_type)
        return {"message": "Hook added successfully.", "hook_id": hook_id}
    except Exception as e:
        logger.error("An error occurred while adding the hook: %s", e)
        raise HTTPException(status_code=500, detail=f"An error occurred while adding the hook: {e}")


def authenticate_token(request: Request):
    """Authenticate using either token or session."""
    token = request.headers.get('Authorization')
    session_id = request.cookies.get('session_id')
    stored_token = get_token()

    if session_id and validate_session(session_id):
        return session_id

    if token and token.startswith('Bearer '):
        token = token.split(' ')[1]
        if token == stored_token:
            return token

    raise HTTPException(status_code=401, detail="Invalid or missing authentication")


async def check_and_run_schedules():
    global SERVER_START_TIME, db_manager

    if not db_manager:
        return

    now = datetime.now(timezone.utc)

    if SERVER_START_TIME and (now - SERVER_START_TIME) < COOLDOWN_PERIOD:
        logger.info("In cooldown period. Skipping schedule check.")
        return

    async with schedule_lock:
        try:
            schedules = db_manager.get_all_schedules()

            for schedule in schedules:
                schedule_id = schedule['id']
                schedule_type = schedule['type']
                pattern = schedule['pattern']
                datetimes = schedule['datetimes']
                last_run = schedule['last_run']

                if schedule_type == 'recurring':
                    if pattern:
                        cron = lazy_import('croniter').croniter(pattern, now)
                        prev_run = cron.get_prev(datetime)
                        next_run = cron.get_next(datetime)

                        if last_run:
                            last_run_dt = datetime.fromisoformat(last_run)
                        else:
                            last_run_dt = None

                        if (
                                prev_run <= now < next_run
                                and last_run_dt is None
                                or last_run_dt < prev_run
                        ):
                            await run_build_process(schedule_id)
                            db_manager.update_schedule_last_run(schedule_id, now)
                            logger.info("Executed recurring build for schedule %s", schedule_id)

                elif schedule_type == 'manual':
                    if datetimes:
                        executed_datetimes = []

                        for dt in datetimes:
                            scheduled_time = datetime.fromisoformat(dt).replace(tzinfo=timezone.utc)
                            if scheduled_time <= now:
                                await run_build_process(schedule_id)
                                logger.info("Executed manual build for schedule %s", schedule_id)
                                executed_datetimes.append(dt)

                        remaining_datetimes = [dt for dt in datetimes if dt not in executed_datetimes]

                        if not remaining_datetimes:
                            db_manager.delete_schedule(schedule_id)
                        else:
                            db_manager.update_schedule(schedule_id, schedule_type, pattern, remaining_datetimes)

        except Exception as e:
            logger.error("An error occurred: %s", e)


def check_node_and_run_npm_install(web_dir):
    if web_dir is None or not isinstance(web_dir, (str, os.PathLike)):
        console.print(Panel(
            "[bold red]Invalid web directory provided.[/bold red]\n\n"
            "[yellow]Aborting further operations due to this error.[/yellow]",
            title="Error: Invalid Directory", expand=False
        ))
        return False

    original_dir = os.getcwd()

    try:
        console.print("")
        try:
            node_path = get_executable_path('node')
            npm_path = get_executable_path('npm')

            if node_path is None or npm_path is None:
                raise FileNotFoundError("Node.js or npm not found")

            node_version = subprocess.run(
                [node_path, "--version"], check=True, capture_output=True, text=True
            ).stdout.strip()
            console.print(f"[green]Node.js version:[/green] {node_version}")
            console.print("")
            console.print("[green]npm is assumed to be installed with Node.js[/green]")
            console.print("")

            web_dir_path = Path(web_dir).resolve()

            if not web_dir_path.exists() or not web_dir_path.is_dir():
                raise FileNotFoundError(f"Web directory not found: {web_dir_path}")

            os.chdir(web_dir_path)

            with console.status("[bold green]Running npm install..."):
                subprocess.run([npm_path, "install"], check=True, capture_output=True)

            console.print("[bold green]Successfully ran 'npm install' in the web directory[/bold green]")

        except FileNotFoundError as e:
            console.print(Panel(
                f"[bold red]{str(e)}[/bold red]\n\n"
                "To resolve this issue:\n"
                "1. Download and install Node.js from [link=https://nodejs.org]https://nodejs.org[/link]\n"
                "2. npm is included with Node.js installation\n"
                "3. After installation, restart your terminal and run this script again\n\n"
                "[yellow]Aborting further operations due to missing Node.js or npm.[/yellow]",
                title="Error: Node.js or npm not found", expand=False
            ))
            return False

        except subprocess.CalledProcessError as e:
            console.print(Panel(
                f"[bold red]An error occurred while running a command:[/bold red]\n\n{e}\n\n"
                "[yellow]Aborting further operations due to this error.[/yellow]",
                title="Command Error", expand=False
            ))
            return False

        except Exception as e:
            console.print(Panel(
                f"[bold red]An unexpected error occurred:[/bold red]\n\n{str(e)}\n\n"
                "[yellow]Aborting further operations due to this error.[/yellow]",
                title="Unexpected Error", expand=False
            ))
            return False

    finally:
        os.chdir(original_dir)
        console.print("")

    return True


def cleanup_and_shutdown():
    """Perform cleanup operations."""
    logger.info("Starting cleanup process...")

    try:
        files_to_delete = ["flatpack.sh"]
        current_directory = Path.cwd()

        for filename in files_to_delete:
            file_path = current_directory / filename
            if file_path.exists():
                file_path.unlink()
                logger.info("Deleted %s.", filename)
    except Exception as e:
        logger.error("Exception during file cleanup: %s", e)

    logger.info("Cleanup complete.")


def create_session(token):
    session_id = secrets.token_urlsafe(32)
    expiration = datetime.now() + timedelta(hours=24)
    active_sessions[session_id] = {
        "token": token,
        "expiration": expiration
    }
    return session_id


def create_temp_sh(custom_json_path: Path, temp_sh_path: Path, use_euxo: bool = False, hooks: List[dict] = None):
    global flatpack_directory

    if hooks is None:
        hooks = []

    try:
        with custom_json_path.open('r', encoding='utf-8') as infile:
            code_blocks = json.load(infile)

        def is_block_disabled(block):
            return block.get('disabled', False)

        last_count = sum(1 for block in code_blocks if not is_block_disabled(block))

        temp_sh_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as context_python_script:
            context_python_script_path = Path(context_python_script.name)

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as exec_python_script:
            exec_python_script_path = Path(exec_python_script.name)

        ensure_database_initialized()
        hooks = db_manager.get_all_hooks()

        with temp_sh_path.open('w', encoding='utf-8') as outfile:
            outfile.write("#!/bin/bash\n")
            outfile.write(f"set -{'eux' if use_euxo else 'eu'}o pipefail\n")

            outfile.write('export SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n')

            outfile.write("VENV_PYTHON=${VENV_PYTHON:-python}\n")

            outfile.write(f"CONTEXT_PYTHON_SCRIPT=\"{context_python_script_path}\"\n")

            outfile.write("EVAL_BUILD=\"$(dirname \"$SCRIPT_DIR\")/web/output/eval_build.json\"\n")

            outfile.write(f"EXEC_PYTHON_SCRIPT=\"{exec_python_script_path}\"\n")
            outfile.write("CURR=0\n")
            outfile.write(f"last_count={last_count}\n")

            outfile.write("trap 'rm -f \"$CONTEXT_PYTHON_SCRIPT\" \"$EXEC_PYTHON_SCRIPT\"; exit' EXIT INT TERM\n")
            outfile.write("rm -f \"$CONTEXT_PYTHON_SCRIPT\" \"$EXEC_PYTHON_SCRIPT\"\n")
            outfile.write("touch \"$CONTEXT_PYTHON_SCRIPT\" \"$EXEC_PYTHON_SCRIPT\"\n")

            outfile.write("datetime=$(date -u +\"%Y-%m-%d %H:%M:%S\")\n")

            outfile.write("DATA_FILE=\"$(dirname \"$SCRIPT_DIR\")/web/output/eval_data.json\"\n")

            outfile.write("echo '[]' > \"$DATA_FILE\"\n\n")

            outfile.write("function log_data() {\n")
            outfile.write("    local part_number=\"$1\"\n")
            outfile.write(
                "    local new_files=$(find \"$SCRIPT_DIR\" -type f -newer \"$DATA_FILE\" \\( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' -o -name '*.txt' -o -name '*.wav' \\) ! -path '*/bin/*' ! -path '*/lib/*')\n")
            outfile.write("    if [ -n \"$new_files\" ]; then\n")
            outfile.write("        local log_entries=\"[]\"\n")
            outfile.write("        local temp_file=$(mktemp)\n")
            outfile.write("        for file in $new_files; do\n")
            outfile.write("            local mime_type=$(file --mime-type -b \"$file\")\n")
            outfile.write("            local web=$(basename \"$file\")\n")
            outfile.write(
                "            local json_entry=\"{\\\"eval\\\": $part_number, \\\"file\\\": \\\"$file\\\", \\\"public\\\": \\\"/output/$web\\\", \\\"type\\\": \\\"$mime_type\\\"}\"\n")
            outfile.write(
                "            if ! jq -e \". | any(.file == \\\"$file\\\")\" \"$DATA_FILE\" > /dev/null; then\n")
            outfile.write("                log_entries=$(echo \"$log_entries\" | jq \". + [$json_entry]\")\n")
            outfile.write("            fi\n")
            outfile.write("        done\n")
            outfile.write("        if [ \"$(echo \"$log_entries\" | jq '. | length')\" -gt 0 ]; then\n")
            outfile.write(
                "            jq -s '.[0] + .[1]' \"$DATA_FILE\" <(echo \"$log_entries\") > \"$temp_file\" && mv \"$temp_file\" \"$DATA_FILE\"\n")
            outfile.write("        fi\n")
            outfile.write("    fi\n")
            outfile.write("    touch \"$DATA_FILE\"\n")
            outfile.write("}\n\n")

            outfile.write("function update_eval_build() {\n")
            outfile.write("    local curr=\"$1\"\n")
            outfile.write("    local eval=\"$2\"\n")
            outfile.write("    echo \"{\n")
            outfile.write("        \\\"curr\\\": $curr,\n")
            outfile.write("        \\\"last\\\": $last_count,\n")
            outfile.write("        \\\"eval\\\": $eval,\n")
            outfile.write("        \\\"datetime\\\": \\\"$datetime\\\"\n")
            outfile.write("    }\" > \"$EVAL_BUILD\"\n")
            outfile.write("}\n\n")

            outfile.write("function execute_hook() {\n")
            outfile.write("    local hook_name=\"$1\"\n")
            outfile.write("    local hook_type=\"$2\"\n")
            outfile.write("    local hook_script=\"$3\"\n")
            outfile.write("    echo \"Executing $hook_type hook: $hook_name\"\n")
            outfile.write("    if [ \"$hook_type\" = \"bash\" ]; then\n")
            outfile.write("        eval \"$hook_script\"\n")
            outfile.write("    elif [ \"$hook_type\" = \"python\" ]; then\n")
            outfile.write("        echo \"$hook_script\" >> \"$CONTEXT_PYTHON_SCRIPT\"\n")
            outfile.write("    else\n")
            outfile.write("        echo \"Unsupported hook type: $hook_type\"\n")
            outfile.write("    fi\n")
            outfile.write("}\n\n")

            outfile.write("update_eval_build \"$CURR\" 1\n\n")

            outfile.write("# Execute 'before' hooks\n")
            for hook in hooks:
                if hook.get('hook_placement') == 'before':
                    hook_script = hook['hook_script'].replace('"', '\\"')
                    outfile.write(f"execute_hook \"{hook['hook_name']}\" \"{hook['hook_type']}\" \"{hook_script}\"\n")
            outfile.write("\n")

            for block in code_blocks:
                if is_block_disabled(block):
                    continue

                language = block.get('type')
                code = block.get('code', '').replace('\r\n', '\n').replace('\r', '\n')

                if language == 'bash':
                    outfile.write(f"{code}\n")
                    outfile.write("((CURR++))\n")

                elif language == 'python':
                    context_code = []
                    execution_code = []
                    in_execution = False
                    code_lines = code.splitlines()

                    for line in code_lines:
                        stripped_line = line.strip()

                        if stripped_line.startswith(('subprocess.run(', 'print(', 'with open(')):
                            in_execution = True

                        if in_execution:
                            execution_code.append(line)
                        else:
                            context_code.append(line)

                    if context_code:
                        context_code_str = '\n'.join(context_code)
                        context_code_escaped = context_code_str.replace('"', r'\"').replace('`', r'\`')
                        outfile.write(f"echo \"{context_code_escaped}\" >> \"$CONTEXT_PYTHON_SCRIPT\"\n")

                    outfile.write("echo \"try:\" > \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("sed 's/^/    /' \"$CONTEXT_PYTHON_SCRIPT\" >> \"$EXEC_PYTHON_SCRIPT\"\n")

                    if execution_code:
                        execution_code_str = '\n'.join(execution_code)
                        execution_code_escaped = execution_code_str.replace('"', r'\"').replace('`', r'\`')

                        outfile.write(
                            f"echo \"{execution_code_escaped}\" | sed 's/^/    /' >> \"$EXEC_PYTHON_SCRIPT\"\n")

                    outfile.write("echo \"except Exception as e:\" >> \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("echo \"    print(e)\" >> \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("echo \"    import sys; sys.exit(1)\" >> \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("$VENV_PYTHON \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("((CURR++))\n")

                else:
                    continue

                outfile.write("log_data \"$CURR\"\n")

                outfile.write("if [ \"$CURR\" -eq \"$last_count\" ]; then\n")
                outfile.write("    EVAL=\"null\"\n")
                outfile.write("else\n")
                outfile.write("    EVAL=$((CURR + 1))\n")
                outfile.write("fi\n")

                outfile.write("update_eval_build \"$CURR\" \"$EVAL\"\n\n")

            outfile.write("# Execute 'after' hooks\n")

            for hook in hooks:
                if hook.get('hook_placement') == 'after':
                    hook_script = hook['hook_script'].replace('"', '\\"')
                    outfile.write(f"execute_hook \"{hook['hook_name']}\" \"{hook['hook_type']}\" \"{hook_script}\"\n")

            outfile.write("\n")

        logger.info("Temp script generated successfully at %s", temp_sh_path)

    except Exception as e:
        logger.error("An error occurred while creating temp script: %s", e, exc_info=True)
        raise


def create_venv(venv_dir: str):
    """
    Create a virtual environment in the specified directory using the current Python version.

    Args:
        venv_dir (str): The directory where the virtual environment will be created.

    Returns:
        None
    """
    python_executable = sys.executable

    try:
        subprocess.run(
            [python_executable, "-m", "venv", venv_dir],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("Virtual environment created successfully in '%s' using %s.", venv_dir, python_executable)
        console.print(
            f"[bold green]SUCCESS:[/bold green] Virtual environment created successfully in '{venv_dir}' using {python_executable}.")
    except subprocess.CalledProcessError as e:
        logger.error("Failed to create virtual environment: %s", e.stderr)
        console.print(f"[bold red]ERROR:[/bold red] Failed to create virtual environment: {e.stderr}")
    except Exception as e:
        logger.error("An unexpected error occurred while creating virtual environment: %s", e)
        console.print(
            f"[bold red]ERROR:[/bold red] An unexpected error occurred while creating virtual environment: {e}")


def create_security_notice():
    security_text = "This environment runs code with your permission, meaning it can connect to the Internet, install new software, which might be risky, read and change files on your computer, and slow down your computer if it does big tasks. Be careful about what code you run here."

    security_message = Text()
    security_message.append(security_text, style="bold yellow")

    return security_message


def create_warning_message():
    warning_text = "Sharing your environment online exposes it to the Internet and may result in the exposure of sensitive data. You are solely responsible for managing and understanding the security risks. We are not responsible for data breaches or unauthorised access from the --share option."

    warning_message = Text()
    warning_message.append(warning_text, style="bold red")

    return warning_message


async def csrf_protect(request: Request):
    csrf_token_cookie = request.cookies.get("csrf_token")
    csrf_token_header = request.headers.get("X-CSRF-Token")

    if not csrf_token_cookie or not csrf_token_header:
        raise HTTPException(status_code=403, detail="CSRF token missing")

    try:
        unsigned_token = request.app.state.signer.unsign(csrf_token_cookie, max_age=3600)
        timestamp, token = unsigned_token.decode().split(':')

        if not secrets.compare_digest(token, csrf_token_header):
            raise HTTPException(status_code=403, detail="CSRF token invalid")

    except (SignatureExpired, BadSignature):
        raise HTTPException(status_code=403, detail="CSRF token expired or invalid")


def decompress_data(input_path, output_path, allowed_dir=None):
    try:
        abs_input_path = validate_file_path(input_path, allowed_dir=allowed_dir)
        abs_output_path = validate_file_path(output_path, is_input=False, allowed_dir=allowed_dir)

        with open(abs_input_path, 'rb') as f:
            compressed_data = f.read()

        decompressed_data = lazy_import('zstandard').decompress(compressed_data)

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


def end_session(session_id):
    if session_id in active_sessions:
        del active_sessions[session_id]


def ensure_database_initialized():
    global db_manager, flatpack_directory

    if db_manager is None:

        if flatpack_directory is None:
            raise ValueError("flatpack_directory is not set")

        db_path = os.path.join(flatpack_directory, 'build', 'flatpack.db')
        logger.info("Initializing database at %s", db_path)
        db_manager = DatabaseManager(db_path)
        db_manager.initialize_database()
        logger.info("Database initialized successfully")


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


def generate_csrf_token():
    token = secrets.token_urlsafe(32)
    timestamp = str(int(time.time()))
    return f"{timestamp}:{token}"


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
    global db_manager
    ensure_database_initialized()
    try:
        return db_manager.get_all_hooks()
    except Exception as e:
        logger.error("An error occurred while fetching hooks: %s", e)
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching hooks: {e}")


def get_executable_path(executable):
    """Securely get the full path of an executable."""
    if sys.platform.startswith('win'):
        system_root = os.environ.get('SystemRoot', 'C:\\Windows')
        where_cmd = os.path.join(system_root, 'System32', 'where.exe')
        try:
            result = subprocess.run([where_cmd, executable],
                                    check=True,
                                    capture_output=True,
                                    text=True)
            paths = result.stdout.strip().split('\n')
            return paths[0] if paths else None
        except subprocess.CalledProcessError:
            return None
    else:
        try:
            which_cmd = '/usr/bin/which'
            if not os.path.exists(which_cmd):
                which_cmd = '/bin/which'

            result = subprocess.run([which_cmd, executable],
                                    check=True,
                                    capture_output=True,
                                    text=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None


def get_secret_key():
    return secrets.token_urlsafe(32)


def get_token() -> Optional[str]:
    """Retrieve the token from the configuration file."""
    config = load_config()
    return config.get('token')


def initialize_fastapi_app(secret_key):
    app = FastAPI(openapi_url=None)

    app.add_middleware(SessionMiddleware, secret_key=secret_key)
    app.state.signer = TimestampSigner(secret_key)

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

    app = setup_routes(app)

    return app


def is_user_logged_in(session_id: str) -> bool:
    if session_id in active_sessions:
        session = active_sessions[session_id]
        if datetime.now() < session["expiration"]:
            return True
    return False


def load_config():
    """Load the configuration from the file.

    Returns:
        dict: The loaded configuration.
    """
    if not os.path.exists(CONFIG_FILE_PATH):
        logger.warning("Configuration file does not exist: %s", CONFIG_FILE_PATH)
        return {}

    try:
        with open(CONFIG_FILE_PATH, 'r') as config_file:
            config = toml.load(config_file)
            return config
    except Exception as e:
        error_message = f"Error loading config: {e}"
        logger.error("Error loading config: %s", e)
        return {}


async def run_build_process(schedule_id=None):
    global build_in_progress, abort_requested

    build_in_progress = True
    logger.info("Running build process...")

    try:
        await update_build_status("in_progress", schedule_id)

        steps = [
            ("Preparing build environment", 1),
            ("Compiling source code", 1),
            ("Running tests", 1),
            ("Packaging application", 1)
        ]

        for step_name, duration in steps:
            if abort_requested:
                logger.info("Build aborted during step: %s", step_name)
                await update_build_status("aborted", schedule_id)
                return

            await update_build_status(f"in_progress: {step_name}", schedule_id)
            await asyncio.sleep(duration)

        if abort_requested:
            logger.info("Build aborted before running build script")
            await update_build_status("aborted", schedule_id)
            return

        await update_build_status("in_progress: Running build script", schedule_id)
        await fpk_build(flatpack_directory)

        if abort_requested:
            logger.info("Build aborted after running build script")
            await update_build_status("aborted", schedule_id)
        else:
            await update_build_status("completed", schedule_id)
            logger.info("Build process completed.")
    except Exception as e:
        logger.error("Build process failed: %s", e)
        await update_build_status("failed", schedule_id, error=str(e))
    finally:
        build_in_progress = False
        abort_requested = False


async def run_scheduler():
    global build_in_progress
    while True:
        if not build_in_progress:
            await check_and_run_schedules()
        else:
            logger.info("Build in progress. Skipping schedule check.")
        await asyncio.sleep(60)


# Based on https://tbrink.science/blog/2017/04/30/processing-the-output-of-a-subprocess-with-python-in-realtime/ (CC0)
class OutStream:
    def __init__(self, fileno):
        self._fileno = fileno
        self._buffer = b""

    def read_lines(self):
        try:
            output = os.read(self._fileno, 4096)
        except OSError as e:
            if e.errno != errno.EIO:
                raise
            output = b""

        lines = output.split(b"\n")
        if self._buffer:
            lines[0] = self._buffer + lines[0]

        if output:
            self._buffer = lines[-1]
            finished_lines = lines[:-1]
            readable = True
        else:
            self._buffer = b""
            if len(lines) == 1 and not lines[0]:
                lines = []
            finished_lines = lines
            readable = False
            os.close(self._fileno)

        finished_lines = [line.decode(errors='replace') for line in finished_lines]
        return finished_lines, readable, output

    def fileno(self):
        return self._fileno


def filter_log_line(line):
    line = line.strip().replace('\r', '')

    exclude_patterns = [
        r'^$',
        r'^\s*$',
        r'^\.{3,}$',
        r'^={3,}$',
        r'^.*\d+%.*$',
        r'^\s*\d+%\[=*>?\s*\]\s*\d+(\.\d+)?[KMGT]?\s+\d+(\.\d+)?[KMGT]?B/s(\s+eta\s+\d+[smh])?\s+\S+(\s+\d+%)?$',
        r'━+\s*\d+(\.\d+)?/\d+(\.\d+)?\s+[KMGT]?B\s+\d+(\.\d+)?\s+[KMGT]?B/s(\s+eta\s+\d+:\d+:\d+)?',
        r'^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}\s+\(\d+(\.\d+)?\s+[KMGT]B/s\)\s+-\s+\S+\s+saved\s+\[\d+/\d+\]$'
    ]

    if any(re.match(pattern, line) for pattern in exclude_patterns):
        return None

    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned_line = ansi_escape.sub('', line)

    if cleaned_line.startswith('%'):
        return None

    stripped_line = cleaned_line.strip()

    if not stripped_line:
        return None

    return stripped_line


async def run_subprocess(command, log_file, timeout=21600):
    global shutdown_requested, abort_requested
    out_r, out_w = pty.openpty()
    err_r, err_w = pty.openpty()
    process = subprocess.Popen(
        command,
        stdout=out_w,
        stderr=err_w,
        stdin=subprocess.PIPE,
        start_new_session=True
    )
    os.close(out_w)
    os.close(err_w)
    streams = [OutStream(out_r), OutStream(err_r)]
    start_time = time.time()

    while streams and not shutdown_requested and not abort_requested:
        if time.time() - start_time > timeout:
            logger.warning("Timeout after %s seconds. Terminating the process.", timeout)
            break

        try:
            rlist, _, _ = select.select(streams, [], [], 1.0)
        except select.error as e:
            if e.args[0] != errno.EINTR:
                raise
            continue

        for stream in rlist:
            try:
                lines, readable, raw_output = stream.read_lines()
                if raw_output:
                    sys.stdout.buffer.write(raw_output)
                    sys.stdout.buffer.flush()

                for line in lines:
                    filtered_line = filter_log_line(line)
                    if filtered_line is not None:
                        log_file.write(filtered_line + '\n')
                        log_file.flush()

                if not readable:
                    streams.remove(stream)
            except Exception as e:
                logger.error("Error processing stream: %s", e)
                logger.debug(traceback.format_exc())
                streams.remove(stream)

        await asyncio.sleep(0)

    if process.poll() is None:
        if abort_requested:
            logger.info("Aborting subprocess...")
        else:
            logger.info("Terminating subprocess...")

        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("Process did not terminate in time. Forcing termination.")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    return process.returncode


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
    except Exception as e:
        error_message = f"Error saving config: {e}"
        logger.error("Error saving config: %s", error_message)


def secure_filename(filename):
    """
    Sanitize a filename to make it secure.

    Args:
        filename (str): The filename to sanitize.

    Returns:
        str: A sanitized version of the filename.
    """
    filename = unicodedata.normalize('NFKD', filename)
    filename = filename.encode('ascii', 'ignore').decode('ascii')
    filename = re.sub(r'[^\w\.-]', '_', filename)
    filename = filename.strip('._')
    return filename


def set_token(token: str):
    try:
        config = load_config()
        config['token'] = token
        save_config(config)
        logger.info("Token set successfully.")
    except Exception as e:
        logger.error("Failed to set token: %s", str(e))


def setup_signal_handlers(process=None):
    def signal_handler(signum, frame):
        global shutdown_requested
        signame = signal.Signals(signum).name

        if process:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

        shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def setup_static_directory(fastapi_app: FastAPI, directory: str):
    """Setup the static directory for serving static files, excluding 'output' folder for unauthenticated users."""
    global flatpack_directory
    flatpack_directory = os.path.abspath(directory)

    if os.path.exists(flatpack_directory) and os.path.isdir(flatpack_directory):
        static_dir = os.path.join(flatpack_directory, 'web')

        @fastapi_app.middleware("http")
        async def block_output_directory(request: Request, call_next):
            if request.url.path.startswith("/output/"):
                session_id = request.cookies.get('session_id')
                if session_id and is_user_logged_in(session_id):
                    return await call_next(request)
                return JSONResponse(status_code=403, content={
                    "detail": "Access to the /output directory is forbidden for unauthenticated users"})
            return await call_next(request)

        fastapi_app.mount(
            "/",
            StaticFiles(directory=static_dir, html=True),
            name="static"
        )
        logger.info("Static files will be served from: %s", static_dir)
    else:
        logger.error("The directory '%s' does not exist or is not a directory.", flatpack_directory)
        raise ValueError(f"The directory '{directory}' does not exist or is not a directory.")


async def shutdown(sig, loop):
    await asyncio.sleep(0.1)
    graceful_shutdown(loop)


def shutdown_server():
    """Shutdown the FastAPI server."""
    logging.getLogger("uvicorn.error").info("Shutting down the server after maximum validation attempts.")
    os._exit(0)


def strip_html(script):
    MarkupResemblesLocatorWarning = lazy_import('bs4', callable_name='MarkupResemblesLocatorWarning')
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    soup = BeautifulSoup(script, "html.parser")
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


async def update_build_status(status, schedule_id=None, error=None):
    global flatpack_directory

    if not flatpack_directory:
        logging.error("flatpack_directory is not set")
        return

    status_data = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "schedule_id": schedule_id
    }
    if error:
        status_data["error"] = str(error)

    status_file = os.path.join(flatpack_directory, 'build', 'build_status.json')

    try:
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        with open(status_file, 'w') as f:
            json.dump(status_data, f)
        logging.info("Updated build status: %s", status)
    except Exception as e:
        logging.error("Failed to update build status: %s", e)


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


def validate_session(session_id):
    if session_id in active_sessions:
        session = active_sessions[session_id]
        if datetime.now() < session["expiration"]:
            return True
    return False


def write_status_to_file(status_data):
    status_file = os.path.join(flatpack_directory, 'build', 'build_status.json')

    with open(status_file, 'w') as f:
        json.dump(status_data, f)


async def fpk_build(directory: Union[str, None], use_euxo: bool = False):
    """Asynchronous function to build a flatpack."""
    global flatpack_directory
    cache_file_path = HOME_DIR / ".fpk_unbox.cache"

    if directory:
        flatpack_dir = Path.cwd() / directory
        if not flatpack_dir.exists():
            logger.error("The directory '%s' does not exist.", flatpack_dir)
            raise ValueError(f"The directory '{flatpack_dir}' does not exist.")

        last_unboxed_flatpack = str(flatpack_dir)
        flatpack_directory = last_unboxed_flatpack
    elif cache_file_path.exists():
        logger.info("Found cached flatpack in %s", cache_file_path)

        last_unboxed_flatpack = cache_file_path.read_text().strip()
        flatpack_directory = last_unboxed_flatpack
    else:
        logger.error("No cached flatpack found, and no valid directory provided.")
        raise ValueError("No cached flatpack found, and no valid directory provided.")

    flatpack_dir = Path(last_unboxed_flatpack)

    if not flatpack_dir.exists():
        logger.error("The flatpack directory '%s' does not exist.", flatpack_dir)
        raise ValueError(f"The flatpack directory '{flatpack_dir}' does not exist.")

    build_dir = flatpack_dir / 'build'

    if not build_dir.exists():
        logger.error("The build directory '%s' does not exist.", build_dir)
        raise ValueError(f"The build directory '{build_dir}' does not exist.")

    sync_hooks_to_db_on_startup()

    custom_json_path = build_dir / 'custom.json'

    if not custom_json_path.exists() or not custom_json_path.is_file():
        logger.error("custom.json not found in %s. Build process canceled.", build_dir)
        raise FileNotFoundError(f"custom.json not found in {build_dir}. Build process canceled.")

    hooks = load_and_get_hooks()
    temp_sh_path = build_dir / 'temp.sh'
    create_temp_sh(custom_json_path, temp_sh_path, use_euxo=use_euxo, hooks=hooks)

    building_script_path = build_dir / 'build.sh'

    if not building_script_path.exists() or not building_script_path.is_file():
        logger.error("Building script not found in %s", build_dir)
        raise FileNotFoundError(f"Building script not found in {build_dir}.")

    log_dir = build_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_time = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
    log_filename = f"build_{log_file_time}.log"
    build_log_file_path = log_dir / log_filename

    safe_script_path = shlex.quote(str(building_script_path.resolve()))

    with open(build_log_file_path, 'w') as log_file:
        await run_subprocess(['/bin/bash', '-u', safe_script_path], log_file)

        web_dir = flatpack_dir / 'web'

        if not web_dir.exists():
            logger.error("The web directory '%s' does not exist.", web_dir)
            raise FileNotFoundError(f"The web directory '{web_dir}' does not exist.")

        output_dir = web_dir / "output"

        eval_build_path = output_dir / "eval_build.json"
        eval_data_path = output_dir / "eval_data.json"

        if not eval_data_path.exists():
            logger.error("The 'eval_data.json' file does not exist in '%s'.", output_dir)
            raise FileNotFoundError(f"The 'eval_data.json' file does not exist in '{output_dir}'.")

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
                        allowed_mimetypes = ['audio/wav', 'audio/x-wav', 'image/jpeg', 'image/png', 'text/plain']
                        mime_type, _ = mimetypes.guess_type(source_file)

                        if mime_type in allowed_mimetypes:
                            dest_path = output_dir / source_file.name
                            shutil.copy2(source_file, dest_path)
                            logger.info("Copied %s to %s", source_file, dest_path)
                    else:
                        logger.error("File %s not found.", source_file)
                else:
                    logger.error("Could not determine relative path for %s", original_path)


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

    except IOError as e:
        logger.error("Failed to cache the directory name '%s': %s", directory_name, e)


def fpk_check_ngrok_auth():
    """
    Check if the NGROK_AUTHTOKEN environment variable is set.

    Raises:
        EnvironmentError: If the NGROK_AUTHTOKEN is not set.
    """
    ngrok_auth_token = os.environ.get('NGROK_AUTHTOKEN')

    if not ngrok_auth_token:
        message = (
            "NGROK_AUTHTOKEN is not set. Please set it using:\n"
            "export NGROK_AUTHTOKEN='your_ngrok_auth_token'"
        )
        logger.error(message)
        raise EnvironmentError(message)

    logger.info("NGROK_AUTHTOKEN is set.")


def fpk_create(flatpack_name, repo_url=TEMPLATE_REPO_URL):
    """Create a new flatpack from a template repository."""
    if not re.match(r'^[a-z0-9-]+$', flatpack_name):
        raise ValueError("Invalid name format. Only lowercase letters, numbers, and hyphens are allowed.")

    flatpack_name = flatpack_name.lower().replace(' ', '-')
    current_dir = os.getcwd()
    flatpack_dir = os.path.join(current_dir, flatpack_name)
    template_dir = None

    if os.path.exists(flatpack_dir):
        raise ValueError(f"Directory '{flatpack_name}' already exists.")

    try:
        template_dir = fpk_download_and_extract_template(repo_url, current_dir)
        os.makedirs(flatpack_dir, exist_ok=True)
        logger.info("Created flatpack directory: %s", flatpack_dir)

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

        files_to_edit = [
            (os.path.join(flatpack_dir, "README.md"), r"# template", f"# {flatpack_name}"),
            (os.path.join(flatpack_dir, "flatpack.toml"), r"{{model_name}}", flatpack_name),
            (os.path.join(flatpack_dir, "build.sh"), r"export DEFAULT_REPO_NAME=template",
             f"export DEFAULT_REPO_NAME={flatpack_name}"),
            (os.path.join(flatpack_dir, "build.sh"), r"export FLATPACK_NAME=template",
             f"export FLATPACK_NAME={flatpack_name}")
        ]

        for file_path, pattern, replacement in files_to_edit:
            with open(file_path, 'r') as file:
                filedata = file.read()
            newdata = re.sub(pattern, replacement, filedata)
            with open(file_path, 'w') as file:
                file.write(newdata)

        logger.info("Edited template files for flatpack: %s", flatpack_name)
        shutil.rmtree(template_dir)
        logger.info("Removed temporary template directory: %s", template_dir)

    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user. Cleaning up...")
        if template_dir and os.path.exists(template_dir):
            shutil.rmtree(template_dir)
        if os.path.exists(flatpack_dir):
            shutil.rmtree(flatpack_dir)
        raise

    except Exception as e:
        logger.error("An error occurred: %s", str(e))
        if os.path.exists(flatpack_dir):
            shutil.rmtree(flatpack_dir)
        raise

    logger.info("Successfully created %s", flatpack_name)
    return flatpack_dir


def fpk_display_disclaimer(directory_name: str, local: bool):
    """Display a disclaimer message with details about a specific flatpack.

    Args:
        directory_name (str): Name of the flatpack directory.
        local (bool): Indicates if the flatpack is local.
    """
    disclaimer_template = """
-----------------------------------------------------
[bold red]STOP AND READ BEFORE YOU PROCEED[/bold red]
https://pypi.org/project/flatpack
[bold]Copyright 2024 Romlin Group AB[/bold]

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
[bold yellow]To accept, type 'YES'. To decline, type 'NO'.[/bold yellow]
-----------------------------------------------------
    """

    if not local:
        please_note_content = f"""
[bold yellow]PLEASE NOTE:[/bold yellow] The flatpack you are about to unbox is
governed by its own licenses and terms, separate from
this software. You may find further details at:

https://fpk.ai/w/{directory_name}
        """
    else:
        please_note_content = ""

    logger.info(
        "Displayed disclaimer for flatpack '%s' with local set to %s.",
        directory_name,
        local
    )

    disclaimer_message = disclaimer_template.format(please_note=please_note_content)
    console.print(disclaimer_message)


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

        files_to_remove = ['app.css', 'app.js', 'index.html', 'package.json', 'robotomono.woff2']

        for file in files_to_remove:
            file_path = os.path.join(template_dir, file)
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print(f"File not found: {file}")

        logger.info(
            "Downloaded and extracted template from %s to %s", repo_url, dest_dir
        )

        return template_dir
    except requests.RequestException as e:
        error_message = f"Failed to download template from {repo_url}: {e}"
        logger.error("%s", error_message)
        raise RuntimeError(error_message)
    except (OSError, IOError) as e:
        error_message = f"Failed to extract template or remove index.html in {dest_dir}: {e}"
        logger.error("%s", error_message)
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

        return None

    fpk_url = f"{BASE_URL}/{directory_name}/{directory_name}.fpk"
    toml_url = f"{BASE_URL}/{directory_name}/flatpack.toml"

    try:
        response = session.get(toml_url)
        response.raise_for_status()

        logger.info("Successfully fetched TOML from %s", toml_url)

        return response.text
    except httpx.HTTPStatusError as e:
        message = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
        logger.error("%s", message)
        return None
    except httpx.RequestError as e:
        message = f"Network error occurred: {e}"
        logger.error("%s", message)
        return None
    except Exception as e:
        message = f"An unexpected error occurred: {e}"
        logger.error("%s", message)
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

            return directories

        message = f"Unexpected response format from GitHub: {json_data}"
        logger.error("%s", message)
        return []

    except httpx.HTTPError as e:
        message = f"Unable to connect to GitHub: {e}"
        logger.error("%s", message)
        sys.exit(1)
    except (ValueError, KeyError) as e:
        message = f"Error processing the response from GitHub: {e}"
        logger.error("%s", message)
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

    except Exception as e:
        error_message = f"An error occurred while searching for model files: {e}"
        logger.error("%s", error_message)

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
        else:
            logger.info("API key not found in the configuration.")
        return api_key
    except Exception as e:
        error_message = f"An error occurred while retrieving the API key: {e}"
        logger.error(error_message)
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
                return last_flatpack
        else:
            logger.warning("Cache file does not exist: %s", cache_file_path)
    except (OSError, IOError) as e:
        error_message = f"An error occurred while accessing the cache file: {e}"
        logger.error("%s", error_message)
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
                    return True
    except IOError as e:
        logger.warning("Could not access /proc/cpuinfo: %s", e)
    logger.info("Not running on a Raspberry Pi.")
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
        return ""


def fpk_set_secure_file_permissions(file_path):
    """Set secure file permissions for the specified file.

    Args:
        file_path (str): Path to the file for which to set secure permissions.
    """
    try:
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
        logger.info("Set secure file permissions for %s", file_path)
    except OSError as e:
        error_message = f"Failed to set secure file permissions for {file_path}: {e}"
        logger.error("Failed to set secure file permissions for %s: %s", file_path, e)


def fpk_unbox(directory_name: str, session: httpx.Client, local: bool = False) -> bool:
    """Unbox a flatpack from GitHub or a local directory."""
    logger.info("Starting fpk_unbox with directory_name: '%s', local: %s", directory_name, local)

    if directory_name is None or not isinstance(directory_name, str) or directory_name.strip() == "":
        logger.error("Invalid directory name: %s", directory_name)
        return False

    if not fpk_valid_directory_name(directory_name):
        logger.error("Invalid directory name: '%s'", directory_name)
        return False

    flatpack_dir = Path.cwd() / directory_name
    logger.info("Flatpack directory: %s", flatpack_dir)

    if not os.path.exists(CONFIG_FILE_PATH):
        logger.info("Config file not found. Creating initial configuration.")
        default_config = {}
        save_config(default_config)

    if local:
        if not flatpack_dir.exists():
            logger.error("Local directory '%s' does not exist.", directory_name)
            return False
        toml_path = flatpack_dir / 'flatpack.toml'
        if not toml_path.exists():
            logger.error("flatpack.toml not found in the specified directory: '%s'.", directory_name)
            return False
    else:
        if flatpack_dir.exists():
            logger.error("Directory '%s' already exists. Unboxing aborted to prevent conflicts.", directory_name)
            return False
        flatpack_dir.mkdir(parents=True, exist_ok=True)

    web_dir = flatpack_dir / "web"
    build_dir = flatpack_dir / "build"
    output_dir = web_dir / "output"

    try:
        web_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created /web directory: %s", web_dir)

        build_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created /build directory: %s", build_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Created /web/output directory: %s", output_dir)

        eval_data_path = output_dir / "eval_data.json"

        with open(eval_data_path, 'w') as f:
            json.dump([], f)

        logger.info("Created empty eval_data.json in %s", eval_data_path)

        files_to_download = {
            'build': [],
            'web': ['app.css', 'app.js', 'index.html', 'package.json', 'robotomono.woff2']
        }

        hooks_json_path = flatpack_dir / 'build' / 'hooks.json'

        if not hooks_json_path.exists():
            files_to_download['build'].append('hooks.json')

        for dir_name, files in files_to_download.items():
            target_dir = web_dir if dir_name == 'web' else build_dir
            for file in files:
                file_url = f"{TEMPLATE_REPO_URL}/contents/{file}"
                response = session.get(file_url)
                response.raise_for_status()

                file_content = response.json()['content']
                file_decoded = base64.b64decode(file_content)

                file_path = target_dir / file

                if file.endswith(".woff2"):
                    with open(file_path, 'wb') as f:
                        f.write(file_decoded)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(file_decoded.decode('utf-8'))

                logger.info("Downloaded and saved %s to %s", file, file_path)

        if not check_node_and_run_npm_install(web_dir):
            console.print("[yellow]Cleaning up: Removing the flatpack directory due to failure.[/yellow]")

            try:
                shutil.rmtree(flatpack_dir)
                console.print("[green]Cleanup successful: Flatpack directory removed.[/green]")
            except Exception as e:
                console.print(f"[red]Error during cleanup: {e}[/red]")

            sys.exit(1)

    except httpx.RequestError as e:
        logger.error("Network error occurred while fetching files: %s", e)
        return False
    except KeyError as e:
        logger.error("Unexpected response structure when fetching files: %s", e)
        return False
    except Exception as e:
        logger.error("Failed to create directories or fetch files: %s", e)
        return False

    build_dir = flatpack_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    db_path = build_dir / 'flatpack.db'
    temp_toml_path = build_dir / 'temp_flatpack.toml'

    if not local:
        fpk_url = f"{BASE_URL}/{directory_name}/{directory_name}.fpk"
        try:
            response = session.head(fpk_url)
            response.raise_for_status()
            logger.info(".fpk file found at %s", fpk_url)
        except httpx.HTTPStatusError:
            logger.error(".fpk file does not exist at %s", fpk_url)
            return False
        except httpx.RequestError as e:
            logger.error("Network error occurred while checking .fpk file: %s", e)

            return False
        except Exception as e:
            logger.error("An unexpected error occurred: %s", e)
            return False

        fpk_path = build_dir / f"{directory_name}.fpk"

        try:
            download_response = session.get(fpk_url)
            download_response.raise_for_status()
            with open(fpk_path, "wb") as fpk_file:
                fpk_file.write(download_response.content)
            logger.info("Downloaded .fpk file to %s", fpk_path)
        except httpx.RequestError as e:
            logger.error("Failed to download the .fpk file: %s", e)
            return False
        except Exception as e:
            logger.error("An unexpected error occurred during the .fpk download: %s", e)
            return False

        try:
            decompress_data(fpk_path, build_dir)
            logger.info("Decompressed .fpk file into %s", build_dir)
        except Exception as e:
            logger.error("Failed to decompress .fpk file: %s", e)
            return False

        toml_path = build_dir / 'flatpack.toml'
        if not toml_path.exists():
            logger.error("flatpack.toml not found in %s", build_dir)
            return False

    try:
        toml_content = toml_path.read_text()
        temp_toml_path.write_text(toml_content)

        bash_script_content = parse_toml_to_venv_script(str(temp_toml_path), env_name=flatpack_dir)
        bash_script_path = build_dir / 'flatpack.sh'
        bash_script_path.write_text(bash_script_content)

        temp_toml_path.unlink()

        logger.info("Unboxing %s...", directory_name)

        safe_script_path = shlex.quote(str(bash_script_path.resolve()))

        subprocess.run(['/bin/bash', safe_script_path], check=True)

        logger.info("All done!")

        initialize_database_manager(str(flatpack_dir))

        fpk_cache_unbox(str(flatpack_dir))

        return True

    except subprocess.CalledProcessError as e:
        logger.error("Failed to execute the bash script: %s", e)
        return False
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        return False
    finally:
        if bash_script_path.exists():
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
    Files are placed in the /web or /build directory, overwriting existing ones.

    Args:
        flatpack_name (str): The name of the flatpack to update.
        session (requests.Session): The HTTP session for making requests.
        branch (str): The branch to fetch files from. Defaults to "main".

    Returns:
        None
    """
    files_to_update = {
        'build': ['device.sh'],
        'web': ['app.css', 'app.js', 'index.html', 'package.json', 'robotomono.woff2']
    }

    binary_extensions = ['.sh', '.woff2']

    flatpack_dir = Path.cwd() / flatpack_name

    if not flatpack_dir.exists() or not flatpack_dir.is_dir():
        console.print(
            f"[bold red]Error:[/bold red] The flatpack '{flatpack_name}' does not exist or is not a directory.")
        return

    web_dir = flatpack_dir / 'web'
    build_dir = flatpack_dir / 'build'

    if not web_dir.exists():
        console.print(
            f"[bold red]Error:[/bold red] No web directory found for flatpack '{flatpack_name}'. Aborting update.")
        return

    if not build_dir.exists():
        build_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"Created [bold cyan]build[/bold cyan] directory for flatpack '{flatpack_name}'.")

    for dir_name, files in files_to_update.items():
        target_dir = web_dir if dir_name == 'web' else build_dir
        for file in files:
            file_url = f"{TEMPLATE_REPO_URL}/contents/{file}?ref={branch}"
            local_file_path = target_dir / file
            console.print(f"Updating [bold cyan]{file}[/bold cyan]...")

            try:
                response = session.get(file_url)
                response.raise_for_status()
                file_data = response.json()

                if 'content' in file_data:
                    file_content = base64.b64decode(file_data['content'])
                    if any(file.endswith(ext) for ext in binary_extensions):
                        with open(local_file_path, 'wb') as local_file:
                            local_file.write(file_content)
                    else:
                        content = file_content.decode('utf-8')
                        with open(local_file_path, 'w', encoding='utf-8') as local_file:
                            local_file.write(content)

                    if local_file_path.exists():
                        console.print(
                            f"[bold green]Replaced[/bold green] existing {file} in flatpack '{flatpack_name}/{dir_name}'")
                    else:
                        console.print(
                            f"[bold green]Added[/bold green] new {file} to flatpack '{flatpack_name}/{dir_name}'")
                else:
                    console.print(f"[bold red]Error:[/bold red] Failed to retrieve content for {file}")

            except requests.RequestException as e:
                console.print(f"[bold red]Error:[/bold red] Failed to update {file}: {str(e)}")
            except UnicodeDecodeError as e:
                console.print(f"[bold red]Error:[/bold red] Failed to decode content for {file}: {str(e)}")

    console.print(f"[bold green]Flatpack '{flatpack_name}' update completed.[/bold green]")


def fpk_verify(directory: Union[str, None]):
    """Verify a flatpack.

    Args:
        directory (Union[str, None]): The directory to use for verification.
            If None, a cached directory will be used if available.

    Returns:
        None
    """
    console.print("[yellow]Flatpack verification functionality is not yet implemented.[/yellow]")

    cache_file_path = HOME_DIR / ".fpk_unbox.cache"
    last_unboxed_flatpack = None

    if directory and fpk_valid_directory_name(directory):
        last_unboxed_flatpack = directory
    elif cache_file_path.exists():
        last_unboxed_flatpack = cache_file_path.read_text().strip()
    else:
        console.print("[red]No valid flatpack directory found.[/red]")
        return

    verification_script_path = Path(last_unboxed_flatpack) / 'build' / 'build.sh'
    if not verification_script_path.exists() or not verification_script_path.is_file():
        console.print(f"[red]Verification script not found in {last_unboxed_flatpack}.[/red]")
        return


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

    parser_compress.add_argument(
        '--method',
        type=str,
        default='llama.cpp',
        choices=['llama.cpp'],
        help='Compression method to use (default: llama.cpp)'
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

    parser_run.add_argument(
        '--domain',
        type=str,
        default=None,
        help='Custom ngrok domain'
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
        help='Update a flatpack from the template'
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
        return

    try:
        vm.add_pdf(pdf_path, pdf_path)
        logger.info("Added text from PDF: '%s' to the vector database.", pdf_path)
    except Exception as e:
        logger.error("Failed to add PDF to the vector database: %s", e)


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
        else:
            logger.error("URL is not accessible: '%s'. HTTP Status Code: %d", url, response.status_code)
    except requests.RequestException as e:
        logger.error("Failed to access URL: '%s'. Error: %s", url, e)


@safe_exit
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
        console.print("No directory name provided. Using cached directory if available.", style="bold yellow")

    console.print("Running build process...", style="bold green")

    try:
        asyncio.run(fpk_build(directory_name, use_euxo=args.use_euxo))
    except KeyboardInterrupt:
        logger.info("Build process was interrupted by user.")
        console.print("\nBuild process was interrupted by user.", style="bold yellow")
    except Exception as e:
        logger.error("An error occurred during the build process: %s", e)
        console.print(f"\nAn error occurred during the build process: {e}", style="bold red")
    finally:
        cleanup_and_shutdown()


def fpk_cli_handle_create(args, session):
    if not args.input:
        console.print("[bold red]Error:[/bold red] No flatpack name specified.")
        return

    flatpack_name = args.input
    if not fpk_valid_directory_name(flatpack_name):
        console.print(
            f"[bold red]Error:[/bold red] Invalid flatpack name: '{flatpack_name}'. Only lowercase letters, numbers, and hyphens are allowed.")
        return

    try:
        with console.status(f"[bold blue]Creating flatpack '{flatpack_name}'..."):
            fpk_create(flatpack_name)

        console.print(f"[bold green]Success:[/bold green] Flatpack '{flatpack_name}' created successfully.")
    except ValueError as e:
        console.print(f"[bold yellow]Warning:[/bold yellow] {str(e)}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] An unexpected error occurred: {str(e)}")


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
    method = getattr(args, 'method', 'llama.cpp')

    if not re.match(r'^[\w-]+/[\w.-]+$', model_id):
        logger.error("Invalid Hugging Face repository format specified.")
        console.print(
            "[bold red]ERROR:[/bold red] Please specify a valid Hugging Face repository in the format 'username/repo_name'.")
        return

    repo_name = model_id.split('/')[-1]
    local_dir = repo_name

    console.print(Panel.fit("[bold green]Starting model compression process[/bold green]", title="Compression Status"))

    try:
        if os.path.exists(local_dir):
            console.print(
                f"[bold yellow]INFO:[/bold yellow] Existing model directory '{local_dir}' found. Attempting to resume download...")
        else:
            console.print(f"[bold blue]INFO:[/bold blue] Creating new directory '{local_dir}' for the model...")
            os.makedirs(local_dir, exist_ok=True)

        console.print(f"[bold blue]INFO:[/bold blue] Downloading model '{model_id}'...")

        try:
            if token:
                lazy_import('huggingface_hub').snapshot_download(
                    repo_id=model_id,
                    local_dir=local_dir,
                    revision="main",
                    token=token,
                    resume_download=True
                )
            else:
                lazy_import('huggingface_hub').snapshot_download(
                    repo_id=model_id,
                    local_dir=local_dir,
                    revision="main",
                    resume_download=True
                )

            console.print(
                f"[bold green]SUCCESS:[/bold green] Finished downloading {model_id} into the directory '{local_dir}'")
        except Exception as e:
            console.print(f"[bold red]ERROR:[/bold red] Failed to download the model. Error: {e}")
            return

        if method == 'llama.cpp':
            current_dir = os.path.basename(os.getcwd())

            if current_dir == "llama.cpp":
                console.print(
                    "[bold blue]INFO:[/bold blue] Current directory is already llama.cpp. Using current directory.")
                llama_cpp_dir = "."
            else:
                llama_cpp_dir = "llama.cpp"

                if not os.path.exists(llama_cpp_dir):
                    console.print("[bold blue]INFO:[/bold blue] Setting up llama.cpp...")
                    git_executable = shutil.which("git")

                    if not git_executable:
                        console.print("[bold red]ERROR:[/bold red] The 'git' executable was not found in your PATH.")
                        return

                    try:
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

                        console.print(
                            f"[bold green]SUCCESS:[/bold green] Finished cloning llama.cpp repository into '{llama_cpp_dir}'")
                    except subprocess.CalledProcessError as e:
                        console.print(
                            f"[bold red]ERROR:[/bold red] Failed to clone the llama.cpp repository. Error: {e}")
                        return
                else:
                    console.print(f"[bold blue]INFO:[/bold blue] llama.cpp directory already exists. Skipping setup.")

            ready_file = os.path.join(llama_cpp_dir, "ready")
            requirements_file = os.path.join(llama_cpp_dir, "requirements.txt")
            venv_dir = os.path.join(llama_cpp_dir, "venv")
            venv_python = os.path.join(venv_dir, "bin", "python")
            convert_script = os.path.join(llama_cpp_dir, 'convert_hf_to_gguf.py')

            if not os.path.exists(ready_file):
                console.print("[bold blue]INFO:[/bold blue] Building llama.cpp...")

                try:
                    make_executable = shutil.which("make")

                    if not make_executable:
                        console.print("[bold red]ERROR:[/bold red] 'make' executable not found in PATH.")
                        return

                    subprocess.run([make_executable], cwd=llama_cpp_dir, check=True)

                    console.print("[bold green]SUCCESS:[/bold green] Finished building llama.cpp")

                    if not os.path.exists(venv_dir):
                        console.print(f"[bold blue]INFO:[/bold blue] Creating virtual environment in '{venv_dir}'...")
                        create_venv(venv_dir)
                    else:
                        console.print(
                            f"[bold blue]INFO:[/bold blue] Virtual environment already exists in '{venv_dir}'")

                    console.print("[bold blue]INFO:[/bold blue] Installing llama.cpp dependencies...")
                    pip_command = [
                        "/bin/bash", "-c",
                        (
                            f"source {shlex.quote(os.path.join(venv_dir, 'bin', 'activate'))} && "
                            f"pip install -r {shlex.quote(requirements_file)}"
                        )
                    ]
                    subprocess.run(pip_command, check=True)
                    console.print("[bold green]SUCCESS:[/bold green] Finished installing llama.cpp dependencies")

                    with open(ready_file, 'w') as f:
                        f.write("Ready")
                except subprocess.CalledProcessError as e:
                    console.print(f"[bold red]ERROR:[/bold red] Failed to build llama.cpp. Error: {e}")
                    return
                except Exception as e:
                    console.print(
                        f"[bold red]ERROR:[/bold red] An error occurred during the setup of llama.cpp. Error: {e}")
                    return
            else:
                console.print("[bold blue]INFO:[/bold blue] llama.cpp is already built and ready.")

            output_file = os.path.join(local_dir, f"{repo_name}-fp16.bin")
            quantized_output_file = os.path.join(local_dir, f"{repo_name}-Q4_K_M.gguf")
            outtype = "f16"

            if not os.path.exists(convert_script):
                console.print(f"[bold red]ERROR:[/bold red] The conversion script '{convert_script}' does not exist.")
                return

            if not os.path.exists(output_file):
                console.print("[bold blue]INFO:[/bold blue] Converting the model...")

                try:
                    venv_activate = os.path.join(venv_dir, "bin", "activate")
                    convert_command = [
                        "/bin/bash", "-c",
                        (
                            f"source {shlex.quote(venv_activate)} && {shlex.quote(venv_python)} "
                            f"{shlex.quote(convert_script)} {shlex.quote(local_dir)} --outfile "
                            f"{shlex.quote(output_file)} --outtype {shlex.quote(outtype)}"
                        )
                    ]
                    console.print(Panel(Syntax(" ".join(convert_command), "bash", theme="monokai", line_numbers=True),
                                        title="Conversion Command", expand=False))
                    subprocess.run(convert_command, check=True)
                    console.print(
                        f"[bold green]SUCCESS:[/bold green] Conversion complete. The model has been compressed and saved as '{output_file}'")
                except subprocess.CalledProcessError as e:
                    console.print(f"[bold red]ERROR:[/bold red] Conversion failed. Error: {e}")
                    return
                except Exception as e:
                    console.print(
                        f"[bold red]ERROR:[/bold red] An error occurred during the model conversion. Error: {e}")
                    return
            else:
                console.print(
                    f"[bold blue]INFO:[/bold blue] The model has already been converted and saved as '{output_file}'.")

            if os.path.exists(output_file):
                console.print("[bold blue]INFO:[/bold blue] Quantizing the model...")

                try:
                    quantize_command = [
                        os.path.join(llama_cpp_dir, 'llama-quantize'),
                        output_file,
                        quantized_output_file,
                        "Q4_K_M"
                    ]
                    subprocess.run(quantize_command, check=True)
                    console.print(
                        f"[bold green]SUCCESS:[/bold green] Quantization complete. The quantized model has been saved as '{quantized_output_file}'.")

                    console.print(f"[bold blue]INFO:[/bold blue] Deleting the original .bin file '{output_file}'...")
                    os.remove(output_file)
                    console.print(f"[bold green]SUCCESS:[/bold green] Deleted the original .bin file '{output_file}'.")
                except subprocess.CalledProcessError as e:
                    console.print(f"[bold red]ERROR:[/bold red] Quantization failed. Error: {e}")
                    return
                except Exception as e:
                    console.print(
                        f"[bold red]ERROR:[/bold red] An error occurred during the quantization process. Error: {e}")
                    return
            else:
                console.print(f"[bold red]ERROR:[/bold red] The original model file '{output_file}' does not exist.")

        else:
            console.print(f"[bold red]ERROR:[/bold red] Unsupported compression method: {method}")

    except KeyboardInterrupt:
        console.print("\n[bold yellow]WARNING:[/bold yellow] Process interrupted by user. Exiting...")
    except Exception as e:
        console.print(f"[bold red]ERROR:[/bold red] An unexpected error occurred: {e}")


def fpk_cli_handle_find(args, session):
    """Handle the 'find' command to search for model files."""
    logger.info("Searching for files...")
    model_files = fpk_find_models()
    if model_files:
        logger.info("Found the following files:")
        for model_file in model_files:
            logger.info(" - %s", model_file)
    else:
        logger.info("No files found.")


def fpk_cli_handle_get_api_key(args, session):
    """Handle the 'get' command to retrieve the API key."""
    logger.info("Retrieving API key...")
    api_key = fpk_get_api_key()
    if api_key:
        logger.info("API Key: %s", api_key)
    else:
        logger.error("No API key found.")


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
    else:
        parser.print_help()
        logger.info("Displayed general help.")


def fpk_cli_handle_list(args, session):
    directories = fpk_list_directories(session)
    if directories:
        table = lazy_import('prettytable').PrettyTable()
        table.field_names = ["Index", "Directory Name"]
        table.align["Index"] = "r"
        table.align["Directory Name"] = "l"

        for index, directory in enumerate(directories.split('\n'), start=1):
            table.add_row([index, directory])

        print(table)
        logger.info("Directories found: %s", directories)
    else:
        logger.error("No directories found.")


# Hooks

def load_and_get_hooks():
    hooks_file_path = os.path.join(flatpack_directory, HOOKS_FILE)
    if os.path.exists(hooks_file_path):
        with open(hooks_file_path, "r") as f:
            return json.load(f).get("hooks", [])
    return []


def add_hook_to_file(hook):
    hooks = load_hooks_from_file()
    hooks.append(hook.dict())
    save_hooks_to_file(hooks)


def delete_hook_from_file(hook_id):
    hooks = load_hooks_from_file()
    hooks = [hook for hook in hooks if hook["hook_id"] != hook_id]
    save_hooks_to_file(hooks)


def load_hooks_from_file():
    hooks_file_path = os.path.join(flatpack_directory, HOOKS_FILE)
    if os.path.exists(hooks_file_path):
        with open(hooks_file_path, "r") as f:
            return json.load(f).get("hooks", [])
    return []


def save_hooks_to_file(hooks):
    hooks_file_path = os.path.join(flatpack_directory, HOOKS_FILE)
    os.makedirs(os.path.dirname(hooks_file_path), exist_ok=True)
    with open(hooks_file_path, "w") as f:
        json.dump({"hooks": hooks}, f, indent=4)


def sync_hooks_to_db_on_startup():
    hooks = load_and_get_hooks()
    for hook in hooks:
        try:
            add_hook_to_database(Hook(**hook))
        except Exception as e:
            logger.warning(
                "Hook %s might already exist in the database. %s",
                hook.get('hook_name', 'unknown'),
                e
            )


def update_hook_in_file(hook_id, updated_hook):
    hooks = load_hooks_from_file()
    for hook in hooks:
        if hook["hook_id"] == hook_id:
            hook.update(updated_hook.dict())
    save_hooks_to_file(hooks)


flatpack_directory = None


def setup_routes(app):
    @app.on_event("startup")
    async def startup_event():
        global SERVER_START_TIME
        SERVER_START_TIME = datetime.now(timezone.utc)
        app.state.csrf_token_base = secrets.token_urlsafe(32)
        logger.info("Server started at %s. Cooldown period: %s", SERVER_START_TIME, COOLDOWN_PERIOD)
        logger.info("CSRF token base generated for this session: %s", app.state.csrf_token_base)
        asyncio.create_task(run_scheduler())

    @app.middleware("http")
    async def csrf_middleware(request: Request, call_next):
        if request.method != "GET" and not any(request.url.path.startswith(path) for path in CSRF_EXEMPT_PATHS):
            await csrf_protect(request)
        return await call_next(request)

    @app.get("/csrf-token")
    async def get_csrf_token(request: Request, response: Response):
        csrf_token = secrets.token_urlsafe(32)
        timestamp = str(int(time.time()))
        token_with_timestamp = f"{timestamp}:{csrf_token}"
        signed_token = request.app.state.signer.sign(token_with_timestamp).decode()

        response.set_cookie(
            key="csrf_token",
            value=signed_token,
            httponly=True,
            samesite="strict",
            secure=False
        )

        return {"csrf_token": csrf_token}

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        favicon_path = Path(flatpack_directory) / "build" / "favicon.ico"
        if favicon_path.exists():
            return FileResponse(favicon_path)
        return Response(status_code=204)

    @app.post("/test-csrf", dependencies=[Depends(csrf_protect)])
    async def test_csrf(request: Request):
        return {"message": "CSRF check passed successfully!"}

    @app.get("/test_db")
    async def test_db():
        try:
            initialize_database_manager(str(flatpack_directory))
            db_manager._execute_query("SELECT 1")

            return {"message": "Database connection successful"}
        except sqlite3.Error as e:
            logger.error("Database connection failed: %s", e)
            return {"message": f"Database connection failed: {e}"}

    @app.post("/api/abort-build", dependencies=[Depends(csrf_protect)])
    async def abort_build(token: str = Depends(authenticate_token)):
        """Abort the current build process."""
        if not build_in_progress:
            return JSONResponse(content={"message": "No build in progress to abort."}, status_code=400)

        await abort_build_process()
        return JSONResponse(content={"message": "Build process aborted successfully."}, status_code=200)

    @app.post("/api/build", dependencies=[Depends(csrf_protect)])
    async def build_flatpack(
            request: Request,
            background_tasks: BackgroundTasks,
            token: str = Depends(authenticate_token)
    ):
        """Trigger the build process for the flatpack."""
        global abort_requested

        if not flatpack_directory:
            raise HTTPException(status_code=500, detail="Flatpack directory is not set")

        if build_in_progress:
            return JSONResponse(
                content={"message": "A build is already in progress."},
                status_code=409
            )

        try:
            abort_requested = False
            background_tasks.add_task(run_build_process, schedule_id=None)
            logger.info("Started build process for flatpack located at %s", flatpack_directory)

            return JSONResponse(
                content={"flatpack": flatpack_directory, "message": "Build process started in background."},
                status_code=200)
        except Exception as e:
            logger.error("Failed to start build process: %s", e)
            return JSONResponse(
                content={"flatpack": flatpack_directory, "message": f"Failed to start build process: {e}"},
                status_code=500)

    @app.get("/api/build-status", dependencies=[Depends(csrf_protect)])
    async def get_build_status(token: str = Depends(authenticate_token)):
        """Get the current build status."""
        if not flatpack_directory:
            raise HTTPException(status_code=500, detail="Flatpack directory is not set")

        status_file = os.path.join(flatpack_directory, 'build', 'build_status.json')

        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                status_data = json.load(f)
            return JSONResponse(content=status_data)
        return JSONResponse(content={"status": "no_builds"})

    @app.post("/api/clear-build-status", dependencies=[Depends(csrf_protect)])
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

    @app.post("/api/comments", dependencies=[Depends(csrf_protect)])
    async def add_comment(comment: Comment, token: str = Depends(authenticate_token)):
        """Add a new comment to the database."""
        ensure_database_initialized()
        try:
            db_manager.add_comment(comment.block_id, comment.selected_text, comment.comment)
            return JSONResponse(content={"message": "Comment added successfully."}, status_code=201)
        except Exception as e:
            logger.error("Error adding comment: %s", str(e), exc_info=True)
            raise HTTPException(status_code=500, detail=f"An error occurred while adding the comment: {str(e)}")

    @app.delete("/api/comments/{comment_id}", dependencies=[Depends(csrf_protect)])
    async def delete_comment(comment_id: int, token: str = Depends(authenticate_token)):
        """Delete a comment from the database by its ID."""
        ensure_database_initialized()
        try:
            if db_manager.delete_comment(comment_id):
                return JSONResponse(content={"message": "Comment deleted successfully."}, status_code=200)
            raise HTTPException(status_code=404, detail="Comment not found")
        except Exception as e:
            logger.error("An error occurred while deleting the comment: %s", e)
            raise HTTPException(status_code=500, detail=f"An error occurred while deleting the comment: {e}")

    @app.get("/api/comments", dependencies=[Depends(csrf_protect)])
    async def get_all_comments(token: str = Depends(authenticate_token)):
        """Retrieve all comments from the database."""
        ensure_database_initialized()
        try:
            return db_manager.get_all_comments()
        except Exception as e:
            logger.error("An error occurred while retrieving comments: %s", e)
            raise HTTPException(status_code=500, detail=f"An error occurred while retrieving comments: {e}")

    @app.post("/api/build", dependencies=[Depends(csrf_protect)])
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
            return JSONResponse(
                content={"flatpack": flatpack_directory, "message": f"Failed to start build process: {e}"},
                status_code=500)

    @app.get("/api/heartbeat", dependencies=[Depends(csrf_protect)])
    async def heartbeat():
        """Endpoint to check the server heartbeat."""
        current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        return JSONResponse(content={"server_time": current_time}, status_code=200)

    @app.post("/api/hooks", dependencies=[Depends(csrf_protect)])
    async def add_hook(hook: Hook, token: str = Depends(authenticate_token)):
        try:
            response = add_hook_to_database(hook)
            if "existing_hook" in response:
                return JSONResponse(content=response, status_code=409)

            hooks = get_all_hooks_from_database()
            save_hooks_to_file(hooks)

            return JSONResponse(content=response, status_code=201)
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error("Failed to add hook: %s", e)
            raise HTTPException(status_code=500, detail="Failed to add hook.")

    @app.delete("/api/hooks/{hook_id}", dependencies=[Depends(csrf_protect)])
    async def delete_hook(hook_id: int, token: str = Depends(authenticate_token)):
        ensure_database_initialized()
        try:
            if db_manager.delete_hook(hook_id):
                hooks = get_all_hooks_from_database()
                save_hooks_to_file(hooks)
                return JSONResponse(content={"message": "Hook deleted successfully."}, status_code=200)
            raise HTTPException(status_code=404, detail="Hook not found")
        except Exception as e:
            logger.error("An error occurred while deleting the hook: %s", e)
            raise HTTPException(status_code=500, detail=f"An error occurred while deleting the hook: {e}")

    @app.get("/api/hooks", response_model=List[Hook], dependencies=[Depends(csrf_protect)])
    async def get_hooks(token: str = Depends(authenticate_token)):
        try:
            hooks_from_file = load_hooks_from_file()
            for hook in hooks_from_file:
                add_hook_to_database(Hook(**hook))

            hooks = get_all_hooks_from_database()
            return JSONResponse(content={"hooks": hooks}, status_code=200)
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error("Failed to retrieve hooks: %s", e)
            raise HTTPException(status_code=500, detail="Failed to retrieve hooks.")

    @app.put("/api/hooks/{hook_id}", dependencies=[Depends(csrf_protect)])
    async def update_hook(hook_id: int, hook: Hook, token: str = Depends(authenticate_token)):
        ensure_database_initialized()
        try:
            success = db_manager.update_hook(hook_id, hook.hook_name, hook.hook_placement, hook.hook_script,
                                             hook.hook_type)
            if success:
                hooks = get_all_hooks_from_database()
                save_hooks_to_file(hooks)
                return JSONResponse(content={"message": "Hook updated successfully."}, status_code=200)
            raise HTTPException(status_code=404, detail="Hook not found or update failed.")
        except Exception as e:
            logger.error("An error occurred while updating the hook: %s", e)
            raise HTTPException(status_code=500, detail=f"An error occurred while updating the hook: {e}")

    @app.get("/api/list_image_files", dependencies=[Depends(csrf_protect)])
    async def list_image_files(token: str = Depends(authenticate_token)):
        global flatpack_directory
        if not flatpack_directory:
            raise HTTPException(status_code=500, detail="Flatpack directory is not set")
        output_folder = Path(flatpack_directory) / "web" / "output"
        allowed_extensions = {'.gif', '.jpeg', '.jpg', '.png'}
        try:
            image_files = [
                {
                    'name': f.name,
                    'created_at': f.stat().st_ctime
                }
                for f in output_folder.iterdir()
                if f.is_file() and f.suffix.lower() in allowed_extensions
            ]
            image_files.sort(key=lambda x: x['created_at'], reverse=True)
            return JSONResponse(content={'files': image_files})
        except Exception as e:
            logger.error("Error listing image files: %s", str(e))
            raise HTTPException(status_code=500, detail=f"Error listing image files: {str(e)}")

    @app.get("/api/load_file")
    async def load_file(
            filename: str,
            token: str = Depends(authenticate_token)
    ):
        if not flatpack_directory:
            raise HTTPException(status_code=500, detail="Flatpack directory is not set")

        sanitized_filename = secure_filename(filename)
        build_dir = Path(flatpack_directory) / 'build'
        file_path = build_dir / sanitized_filename

        try:
            file_path = file_path.resolve()
            build_dir = build_dir.resolve()

            if not file_path.is_relative_to(build_dir):
                raise HTTPException(status_code=403, detail="Access to the requested file is forbidden")
        except Exception as e:
            logger.error("Error validating file path: %s", e)
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                code_blocks = json.load(file)

            if not isinstance(code_blocks, list):
                raise ValueError("Invalid file format: Expected a list of code blocks")

            json_content = json.dumps(code_blocks)
            base64_content = base64.b64encode(json_content.encode('utf-8')).decode('utf-8')

            return JSONResponse(content={"content": base64_content})

        except json.JSONDecodeError:
            logger.error("Error decoding JSON from file: %s", sanitized_filename)
            raise HTTPException(status_code=400, detail="Invalid JSON format in file")
        except ValueError as ve:
            logger.error("Error validating file content: %s", str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error("Error reading file '%s': %s", sanitized_filename, str(e))
            raise HTTPException(status_code=500, detail="Error reading file")

    @app.get("/api/logs", dependencies=[Depends(csrf_protect)])
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

    @app.get("/api/logs/{log_filename}", dependencies=[Depends(csrf_protect)])
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

        sanitized_log_filename = secure_filename(log_filename)
        log_path = logs_directory / sanitized_log_filename

        if log_path.exists() and log_path.is_file():
            try:
                if not log_path.resolve().is_relative_to(logs_directory.resolve()):
                    raise HTTPException(status_code=403, detail="Access to the requested file is forbidden")

                with open(log_path, 'r') as file:
                    content = file.read()
                return JSONResponse(content={"log": content}, status_code=200)
            except Exception as e:
                logger.error("Error reading log file '%s': %s", sanitized_log_filename, e)
                raise HTTPException(status_code=500, detail=f"Error reading log file: {e}")
        else:
            raise HTTPException(status_code=404, detail="Log file not found")

    @app.post("/api/save_file", dependencies=[Depends(csrf_protect)])
    async def save_file(
            request: Request,
            filename: str = Form(...),
            content: str = Form(...),
            token: str = Depends(authenticate_token)
    ):
        if not flatpack_directory:
            raise HTTPException(status_code=500, detail="Flatpack directory is not set")

        sanitized_filename = secure_filename(filename)
        file_path = os.path.join(flatpack_directory, 'build', sanitized_filename)

        if not os.path.commonpath([flatpack_directory, os.path.realpath(file_path)]).startswith(
                os.path.realpath(flatpack_directory)):
            raise HTTPException(status_code=403, detail="Access to the requested file is forbidden")

        try:
            decoded_content = base64.b64decode(content.encode('utf-8')).decode('utf-8')

            code_blocks = json.loads(decoded_content)

            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(code_blocks, file, ensure_ascii=False, indent=4)

            return JSONResponse(content={"message": "File saved successfully!"})

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    @app.get("/api/schedule", dependencies=[Depends(csrf_protect)])
    async def get_schedule(token: str = Depends(authenticate_token)):
        ensure_database_initialized()
        try:
            schedules = db_manager.get_all_schedules()
            return JSONResponse(content={"schedules": schedules}, status_code=200)
        except Exception as e:
            logger.error("An error occurred while retrieving the schedules: %s", e)
            raise HTTPException(status_code=500, detail=f"An error occurred while retrieving the schedules: {e}")

    @app.post("/api/schedule", dependencies=[Depends(csrf_protect)])
    async def save_schedule(request: Request, token: str = Depends(authenticate_token)):
        ensure_database_initialized()
        try:
            data = await request.json()
            schedule_type = data.get('type')
            pattern = data.get('pattern')
            datetimes = data.get('datetimes', [])

            if schedule_type == 'manual':
                datetimes = [datetime.fromisoformat(dt).astimezone(timezone.utc).isoformat() for dt in datetimes]

            new_schedule_id = db_manager.add_schedule(schedule_type, pattern, datetimes)

            return JSONResponse(content={"message": "Schedule saved successfully.", "id": new_schedule_id},
                                status_code=200)
        except Exception as e:
            logger.error("An error occurred while saving the schedule: %s", e)
            raise HTTPException(status_code=500, detail=f"An error occurred while saving the schedule: {e}")

    @app.delete("/api/schedule/{schedule_id}", dependencies=[Depends(csrf_protect)])
    async def delete_schedule_entry(schedule_id: int, datetime_index: Optional[int] = None,
                                    token: str = Depends(authenticate_token)):
        ensure_database_initialized()
        try:
            if datetime_index is not None:
                success = db_manager.delete_schedule_datetime(schedule_id, datetime_index)
                if success:
                    return JSONResponse(content={"message": "Schedule datetime entry deleted successfully."},
                                        status_code=200)
                raise HTTPException(status_code=404, detail="Datetime entry not found")
            success = db_manager.delete_schedule(schedule_id)
            if success:
                return JSONResponse(content={"message": "Entire schedule deleted successfully."}, status_code=200)
            raise HTTPException(status_code=404, detail="Schedule not found")
        except Exception as e:
            logger.error("An error occurred while deleting the schedule entry: %s", e)
            raise HTTPException(status_code=500,
                                detail=f"An error occurred while deleting the schedule entry: {e}")

    @app.get("/api/user_status")
    async def user_status(auth: str = Depends(authenticate_token)):
        """Check if the user is authenticated."""
        return JSONResponse(content={"authenticated": True}, status_code=200)

    @app.post("/api/validate_token")
    async def validate_token(request: Request, api_token: str = Form(...)):
        """Validate the provided API token and create a session."""
        global VALIDATION_ATTEMPTS
        stored_token = get_token()

        if not stored_token:
            return JSONResponse(content={"message": "API token is not set."}, status_code=200)
        if validate_api_token(api_token):
            session_id = create_session(api_token)
            response = JSONResponse(content={"message": "API token is valid.", "session_id": session_id},
                                    status_code=200)
            response.set_cookie(key="session_id", value=session_id, httponly=True, samesite="strict")
            return response
        VALIDATION_ATTEMPTS += 1
        if VALIDATION_ATTEMPTS >= MAX_ATTEMPTS:
            shutdown_server()
        return JSONResponse(content={"message": "Invalid API token."}, status_code=401)

    @app.post("/api/verify", dependencies=[Depends(csrf_protect)])
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

    return app


def fpk_cli_handle_run(args, session):
    """Handle the 'run' command to start the FastAPI server."""
    global console

    if not args.input:
        console.print("Please specify a flatpack for the run command.", style="bold red")
        return

    security_message = create_security_notice()

    console.print(
        Panel(
            security_message,
            title="[bold yellow]SECURITY NOTICE[/bold yellow]",
            border_style="bold yellow",
            expand=False,
            padding=(1, 1),
        )
    )

    while True:
        acknowledgment = console.input("[bold yellow]Do you agree to proceed? (YES/NO):[/bold yellow] ").strip().upper()

        if acknowledgment == "YES":
            break
        if acknowledgment == "NO":
            console.print("")
            console.print("You must agree to proceed. Exiting.", style="bold red")
            return
        else:
            console.print("")
            console.print("Please answer YES or NO.", style="bold yellow")

    directory = Path(args.input).resolve()
    allowed_directory = Path.cwd()

    if not directory.is_dir() or not directory.exists():
        console.print("")
        console.print(f"The flatpack '{directory}' does not exist or is not a directory.", style="bold red")
        return

    if not directory.is_relative_to(allowed_directory):
        console.print("")
        console.print("The specified directory is not within allowed paths.", style="bold red")
        return

    build_dir = directory / 'build'
    web_dir = directory / 'web'

    if not build_dir.exists() or not web_dir.exists():
        console.print("")
        console.print("The 'build' or 'web' directory is missing in the flatpack.", style="bold yellow")
        return

    console.print("")

    if args.share:
        warning_message = create_warning_message()

        console.print(
            Panel(
                warning_message,
                title="[bold red]WARNING[/bold red]",
                border_style="bold red",
                expand=False,
                padding=(1, 1),
            )
        )

        while True:
            user_response = console.input(
                "[bold red]Do you accept these risks? (YES/NO):[/bold red] "
            ).strip().upper()
            if user_response == "YES":
                console.print("")
                try:
                    fpk_check_ngrok_auth()
                except EnvironmentError as e:
                    return
                break
            if user_response == "NO":
                console.print("")
                console.print("[bold red]Sharing aborted. Exiting.[/bold red]")
                return
            else:
                console.print("")
                console.print("[bold red]Please answer YES or NO.[/bold red]")

    secret_key = get_secret_key()
    logger.info("[CSRF] New secret key generated for this session.")
    console.print("[CSRF] New secret key generated for this session.", style="bold blue")

    csrf_token_base = secrets.token_urlsafe(32)
    logger.info("[CSRF] New token generated for this session.")
    console.print("[CSRF] New token generated for this session.", style="bold blue")

    console.print("")

    token = generate_secure_token()
    logger.info("API token generated and displayed to user.")
    console.print(f"Generated API token: {token}", style="bold bright_cyan")

    set_token(token)

    console.print("")

    with console.status("[bold green]Initializing FastAPI server...", spinner="dots") as status:
        app = initialize_fastapi_app(secret_key)
        setup_static_directory(app, str(directory))

    host = "127.0.0.1"

    with console.status("[bold green]Finding available port...", spinner="dots"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind((host, 0))
        port = sock.getsockname()[1]
        sock.close()

    public_url = None

    if args.share:
        with console.status("[bold green]Establishing ngrok ingress...", spinner="dots"):
            ngrok_module = lazy_import('ngrok')

            if args.domain:
                listener = ngrok_module.connect(f"{host}:{port}", authtoken_from_env=True, domain=args.domain)
            else:
                listener = ngrok_module.connect(f"{host}:{port}", authtoken_from_env=True)

            public_url = listener.url()

            logger.info("Ingress established at %s", public_url)
            console.print(f"Ingress established at {public_url}", style="bold green")
            console.print("")

    config = uvicorn.Config(app, host=host, port=port)
    server = uvicorn.Server(config)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(run_scheduler)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user (KeyboardInterrupt).")
        console.print("Process interrupted by user. Exiting.", style="bold yellow")
    finally:
        if args.share and public_url:
            try:
                ngrok_module = lazy_import('ngrok')
                ngrok_module.disconnect(public_url)
                logger.info("Disconnected ngrok ingress.")
                console.print("Disconnected ngrok ingress.", style="bold green")
            except Exception as e:
                logger.error("Failed to disconnect ngrok ingress: %s", str(e))
                console.print(f"Failed to disconnect ngrok ingress: {e}", style="bold red")


def fpk_cli_handle_set_api_key(args, session):
    """Handle the 'set' command to set the API key."""
    logger.info("Setting API key: %s", args.api_key)

    api_key = args.api_key
    config = load_config()
    config['api_key'] = api_key
    save_config(config)

    logger.info("API key set successfully!")

    try:
        test_key = fpk_get_api_key()
        if test_key == api_key:
            logger.info("Verification successful: API key matches.")
        else:
            logger.error("Verification failed: API key does not match.")
    except Exception as e:
        logger.error("Error during API key verification: %s", e)


def fpk_cli_handle_unbox(args, session):
    """Handle the 'unbox' command to unbox a flatpack from GitHub or a local directory.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    if args.input is None:
        console.print("Please specify a flatpack for the unbox command.", style="bold red")
        return

    directory_name = args.input

    existing_dirs = fpk_fetch_github_dirs(session)
    console.print("Running unbox process...", style="bold green")

    if directory_name not in existing_dirs and not args.local:
        console.print("")
        console.print(f"The flatpack '{directory_name}' does not exist.", style="bold red")
        return

    fpk_display_disclaimer(directory_name, local=args.local)

    while True:
        user_response = input().strip().upper()
        if user_response == "YES":
            break
        if user_response == "NO":
            console.print("")
            console.print("Installation aborted by user.", style="bold yellow")
            return
        logger.error("Invalid input from user. Expected 'YES' or 'NO'.")
        console.print("Invalid input. Please type 'YES' to accept or 'NO' to decline.", style="bold red")

    if args.local:
        local_directory_path = Path(directory_name)

        if not local_directory_path.exists() or not local_directory_path.is_dir():
            logger.error("Local directory does not exist: '%s'.", directory_name)
            console.print(f"Local directory does not exist: '{directory_name}'.", style="bold red")
            return

        directory_name = str(local_directory_path.resolve())  # Update directory_name to full path
        toml_path = local_directory_path / 'flatpack.toml'

        if not toml_path.exists():
            logger.error("flatpack.toml not found in the specified directory: '%s'.", directory_name)
            console.print(f"flatpack.toml not found in '{directory_name}'.", style="bold red")
            return

    if directory_name is None or directory_name.strip() == "":
        logger.error("Invalid directory name")
        console.print("Invalid directory name", style="bold red")
        return

    logger.info("Directory name resolved to: '%s'", directory_name)

    console.print("")
    console.print(f"Directory name resolved to: '{directory_name}'", style="bold green")

    try:
        unbox_result = fpk_unbox(directory_name, session, local=args.local)
        if unbox_result:
            logger.info("Unboxed flatpack '%s' successfully.", directory_name)
            console.print(f"Unboxed flatpack '{directory_name}' successfully.", style="bold green")
        else:
            logger.info("Unboxing of flatpack '%s' was aborted.", directory_name)
            console.print(f"Unboxing of flatpack '{directory_name}' was aborted.", style="bold yellow")
    except Exception as e:
        logger.error("Failed to unbox flatpack '%s': %s", directory_name, e)
        console.print(f"Failed to unbox flatpack '{directory_name}': {e}", style="bold red")


def fpk_cli_handle_update(args, session):
    """Handle the 'update' command to update a flatpack's files from the template.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    if not args.flatpack_name:
        logger.error("No flatpack specified for the update command.")
        return

    fpk_update(args.flatpack_name, session)


def fpk_cli_handle_vector_commands(args, session, vm):
    """Handle vector database commands.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
        vm: The Vector Manager instance.
    """
    if args.vector_command == 'add-texts':
        vm.add_texts(args.texts, "manual")
        console.print(f"[bold green]Added {len(args.texts)} texts to the database.[/bold green]")

    elif args.vector_command == 'search-text':
        results = vm.search_vectors(args.query)
        if results:
            console.print("[bold cyan]Search results:[/bold cyan]")
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("ID", style="dim", width=12)
            table.add_column("Text")
            for result in results:
                table.add_row(str(result['id']), result['text'])
            console.print(table)
        else:
            console.print("[yellow]No results found.[/yellow]")

    elif args.vector_command == 'add-pdf':
        console.print(f"[bold blue]Adding PDF: {args.pdf_path}[/bold blue]")
        fpk_cli_handle_add_pdf(args.pdf_path, vm)

    elif args.vector_command == 'add-url':
        console.print(f"[bold blue]Adding URL: {args.url}[/bold blue]")
        fpk_cli_handle_add_url(args.url, vm)

    elif args.vector_command == 'add-wikipedia':
        vm.add_wikipedia_page(args.page_title)
        console.print(
            f"[bold green]Added text from Wikipedia page: '{args.page_title}' to the vector database.[/bold green]")

    else:
        console.print(f"[bold red]Unknown vector command: {args.vector_command}[/bold red]")

    logger.info("Handled vector command: %s", args.vector_command)


def fpk_cli_handle_verify(args, session):
    """Handle the 'verify' command to verify a flatpack.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    directory_name = args.directory
    if not directory_name:
        logger.error("No directory specified for the verify command.")
        return

    logger.info("Verifying flatpack in directory: %s", directory_name)

    try:
        fpk_verify(directory_name)
        logger.info("Verification successful for directory: %s", directory_name)
    except Exception as e:
        logger.error("Verification failed for directory '%s': %s", directory_name, e)


def fpk_cli_handle_version(args, session):
    """Handle the 'version' command to display the version of flatpack.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    print(VERSION)


@safe_exit
def main():
    setup_exception_handling()
    setup_signal_handling()

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


if __name__ == "__main__":
    main()
