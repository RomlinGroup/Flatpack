import os
import sys

from pathlib import Path
from textwrap import dedent

from rich.console import Console

PACKAGE_DIR = Path(sys.modules["flatpack"].__file__).parent
IMPORT_CACHE_FILE = PACKAGE_DIR / ".fpk_import_cache"

console = Console()

if not IMPORT_CACHE_FILE.exists():
    ascii_art = dedent(
        """\
         _____ __    _____ _____ _____ _____ _____ _____ 
        |   __|  |  |  _  |_   _|  _  |  _  |     |  |  |
        |   __|  |__|     | | | |   __|     |   --|    -|
        |__|  |_____|__|__| |_| |__|  |__|__|_____|__|__|                                      
        """
    )

    console.print(f"[bold green]{ascii_art}[/bold green]")
    console.print(
        "[bold green]Initialising Flatpack for the first time. This may take a moment...[/bold green]"
    )

import argparse
import asyncio
import base64
import importlib
import json
import logging
import mimetypes
import re
import secrets
import select
import shlex
import shutil
import signal
import sqlite3
import stat
import string
import subprocess
import tempfile
import threading
import time
import unicodedata

from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from io import BytesIO
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Set, Tuple, Union
from tqdm.auto import tqdm
from zipfile import ZipFile

import httpx
import psutil
import requests
import toml

from itsdangerous import BadSignature, SignatureExpired, TimestampSigner
from pydantic import BaseModel, Field
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from .database_manager import DatabaseManager
from .error_handling import \
    safe_exit, \
    setup_exception_handling, \
    setup_signal_handling
from .package_manager import PackageManager
from .parsers import parse_toml_to_venv_script
from .session_manager import SessionManager
from .vector_manager import VectorManager

if not IMPORT_CACHE_FILE.exists():
    IMPORT_CACHE_FILE.touch()
    console.print("")
    console.print(
        "[bold green]First-time initialisation complete! ✨[/bold green]"
    )
    console.print("")

_cache_lock = threading.RLock()
_runtime_cache = {}


def lazy_import(module_name, package=None, callable_name=None):
    """Dynamically import module or callable with thread-safe runtime caching."""
    global _cache_lock, _runtime_cache
    cache_key = (module_name, package, callable_name)

    with _cache_lock:
        if cache_key in _runtime_cache:
            return _runtime_cache[cache_key]

        try:
            module = importlib.import_module(module_name, package)
            result = getattr(module,
                             callable_name) if callable_name else module
            _runtime_cache[cache_key] = result
            return result
        except (ImportError, AttributeError):
            _runtime_cache[cache_key] = None
            return None


def set_file_limits():
    """Set higher limits for number of open files."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        return True
    except Exception as e:
        print(f"Warning: Could not increase file limits: {e}")
        return False


APIKeyCookie = lazy_import("fastapi.security", callable_name="APIKeyCookie")
BackgroundTasks = lazy_import("fastapi", callable_name="BackgroundTasks")
BaseHTTPMiddleware = lazy_import(
    "starlette.middleware.base",
    callable_name="BaseHTTPMiddleware"
)
Depends = lazy_import("fastapi", callable_name="Depends")
FastAPI = lazy_import("fastapi", callable_name="FastAPI")
Form = lazy_import("fastapi", callable_name="Form")
HTTPException = lazy_import("fastapi", callable_name="HTTPException")
JSONResponse = lazy_import("fastapi.responses", callable_name="JSONResponse")
Request = lazy_import("fastapi", callable_name="Request")
warnings = lazy_import("warnings")

HOME_DIR = Path.home() / ".fpk"

BASE_URL = "https://raw.githubusercontent.com/RomlinGroup/Flatpack/main/warehouse"
CONNECTIONS_FILE = "connections.json"
COOLDOWN_PERIOD = timedelta(minutes=1)
CONFIG_FILE_PATH = HOME_DIR / ".fpk_config.toml"
CSRF_EXEMPT_PATHS = ["/", "/csrf-token", "/favicon.ico", "/static"]
GITHUB_CACHE = HOME_DIR / ".fpk_github.cache"
GITHUB_CACHE_EXPIRY = timedelta(hours=1)
GITHUB_REPO_URL = "https://api.github.com/repos/RomlinGroup/Flatpack"
HOOKS_FILE = "hooks.json"
HOME_DIR.mkdir(exist_ok=True)
MAX_ATTEMPTS = 5
SERVER_START_TIME = None
TEMPLATE_REPO_URL = "https://api.github.com/repos/RomlinGroup/template"
VALIDATION_ATTEMPTS = 0
VERSION = version("flatpack")

abort_requested = False
active_sessions = {}
build_in_progress = False
console = Console()
force_exit_timer = None
nextjs_process = None
shutdown_in_progress = False
shutdown_requested = False


class BuildStatusData(BaseModel):
    status: str = Field(...)
    timestamp: str = Field(...)
    cleanup_success: bool = Field(...)
    initial_status: str = Field(...)
    final_status: str = Field(...)
    memory_released: float = Field(...)
    error: Optional[str] = None


class Comment(BaseModel):
    block_id: str
    selected_text: str
    comment: str


class EndpointFilter(logging.Filter):
    def filter(self, record):
        return all(
            endpoint not in record.getMessage()
            for endpoint in [
                "GET /api/heartbeat",
                "GET /api/build-status",
                "GET /api/list-media-files",
                "GET /api/source-hook-mappings",
                "GET /csrf-token",
                "POST /api/clear-build-status",
            ]
        )


class Hook(BaseModel):
    hook_name: str
    hook_placement: str
    hook_script: str
    hook_type: str
    show_on_frontpage: bool = False


class MappingResults(BaseModel):
    mappings: List[dict]


class Source(BaseModel):
    source_name: str
    source_type: str
    source_details: Optional[dict] = None


class SourceHookMapping(BaseModel):
    sourceId: str
    targetId: str
    sourceType: str
    targetType: str


class SourceUpdate(BaseModel):
    source_name: Optional[str]
    source_type: Optional[str]
    source_details: Optional[Dict[str, str]]


abort_lock = asyncio.Lock()
csrf_cookie = APIKeyCookie(name="csrf_token")

db_manager = None

DirectoryPaths = Tuple[Path, Path, Path, Path]


def initialize_database_manager(flatpack_db_directory):
    """Initializes the database manager, creating the database file and directory if necessary."""
    global db_manager

    if flatpack_db_directory is None:
        raise ValueError("flatpack_db_directory is not initialized.")

    db_path = os.path.join(flatpack_db_directory, "build", "flatpack.db")

    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path))
        console.print(
            f"[bold green]SUCCESS:[/bold green] Created directory for database at path: {os.path.dirname(db_path)}"
        )

    db_manager = DatabaseManager(db_path)
    db_manager.initialize_database()
    console.print(
        f"[bold blue]INFO:[/bold blue] Database manager initialized with database at: {db_path}"
    )


logger = logging.getLogger(__name__)


def setup_logging(log_path: Path) -> logging.Logger:
    """Sets up logging to both console and a rotating file."""
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(logging.WARNING)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    try:
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.WARNING)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        logger.info(
            "Logging initialized: console and file output to %s",
            log_path
        )
        return logger

    except OSError as e:
        logger.addHandler(console_handler)
        logger.error("Failed to setup file logging at %s: %s", log_path, e)
        return logger


try:
    global_log_file_path = HOME_DIR / ".fpk_logger.log"
    logger = setup_logging(global_log_file_path)
    os.chmod(global_log_file_path, 0o600)
except Exception as e:
    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.error(f"Failed to initialize logging: {e}")

schedule_lock = asyncio.Lock()
uvicorn_server = None


def add_hook_to_database(hook: Hook):
    global db_manager
    ensure_database_initialized()
    try:
        if db_manager.hook_exists(hook.hook_name):
            existing_hook = db_manager.get_hook_by_name(hook.hook_name)
            return {
                "message": "Hook with this name already exists.",
                "existing_hook": existing_hook,
                "new_hook": hook.dict(),
            }

        hook_id = db_manager.add_hook(
            hook.hook_name,
            hook.hook_placement,
            hook.hook_script,
            hook.hook_type,
            hook.show_on_frontpage,
        )
        return {"message": "Hook added successfully.", "hook_id": hook_id}
    except Exception as e:
        logger.error("An error occurred while adding the hook: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while adding the hook: {e}"
        )


def authenticate_token(request: Request):
    """Authenticate using either token or session."""
    token = request.headers.get("Authorization")
    session_id = request.cookies.get("session_id")
    stored_token = get_token()

    if session_id and validate_session(session_id):
        return session_id

    if token and token.startswith("Bearer "):
        token = token.split(" ")[1]
        if token == stored_token:
            return token

    raise HTTPException(status_code=401,
                        detail="Invalid or missing authentication")


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
                schedule_id = schedule["id"]
                schedule_type = schedule["type"]
                pattern = schedule["pattern"]
                datetimes = schedule["datetimes"]
                last_run = schedule["last_run"]

                if schedule_type == "recurring":
                    if pattern:
                        cron = lazy_import("croniter").croniter(pattern, now)
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
                            db_manager.update_schedule_last_run(schedule_id,
                                                                now)
                            logger.info(
                                "Executed recurring build for schedule %s",
                                schedule_id
                            )

                elif schedule_type == "manual":
                    if datetimes:
                        executed_datetimes = []

                        for dt in datetimes:
                            scheduled_time = datetime.fromisoformat(
                                dt).replace(
                                tzinfo=timezone.utc
                            )
                            if scheduled_time <= now:
                                await run_build_process(schedule_id)
                                logger.info(
                                    "Executed manual build for schedule %s",
                                    schedule_id
                                )
                                executed_datetimes.append(dt)

                        remaining_datetimes = [
                            dt for dt in datetimes if
                            dt not in executed_datetimes
                        ]

                        if not remaining_datetimes:
                            db_manager.delete_schedule(schedule_id)
                        else:
                            db_manager.update_schedule(
                                schedule_id, schedule_type, pattern,
                                remaining_datetimes
                            )

        except Exception as e:
            logger.error("An error occurred: %s", e)


def check_node_and_run_npm_install(web_dir):
    if web_dir is None or not isinstance(web_dir, (str, os.PathLike)):
        console.print(
            Panel(
                "[bold red]Invalid web directory provided.[/bold red]\n\n"
                "[yellow]Aborting further operations due to this error.[/yellow]",
                title="Error: Invalid Directory",
                expand=False,
            )
        )
        return False

    original_dir = os.getcwd()

    try:
        try:
            node_path = get_executable_path("node")
            npm_path = get_executable_path("npm")

            if node_path is None or npm_path is None:
                raise FileNotFoundError("Node.js or npm not found")

            node_version = subprocess.run(
                [node_path, "--version"],
                check=True,
                capture_output=True,
                text=True
            ).stdout.strip()
            console.print("")
            console.print(f"[green]Node.js version:[/green] {node_version}")
            console.print("")
            console.print(
                "[green]npm is assumed to be installed with Node.js[/green]")
            console.print("")

            web_dir_path = Path(web_dir).resolve()

            if not web_dir_path.exists() or not web_dir_path.is_dir():
                raise FileNotFoundError(
                    f"Web directory not found: {web_dir_path}")

            os.chdir(web_dir_path)

            with console.status(
                    "[bold green]Running npm install...",
                    spinner="dots"
            ):
                subprocess.run(
                    [
                        npm_path,
                        "install"
                    ],
                    check=True,
                    capture_output=True,
                    text=True
                )

            console.print(
                "[bold green]Successfully ran 'npm install'[/bold green]"
            )

        except FileNotFoundError as e:
            console.print(
                Panel(
                    dedent(
                        f"""\
                    [bold red]{str(e)}[/bold red]

                    To resolve this issue:
                    1. Download and install Node.js from [link=https://nodejs.org]https://nodejs.org[/link]
                    2. npm is included with Node.js installation
                    3. After installation, restart your terminal and run this script again

                    [yellow]Aborting further operations due to missing Node.js or npm.[/yellow]"""
                    ),
                    title="Error: Node.js or npm not found",
                    expand=False,
                )
            )
            return False

        except subprocess.CalledProcessError as e:
            console.print(
                Panel(
                    dedent(
                        f"""\
                    [bold red]An error occurred while running a command:[/bold red]

                    {e}

                    Stdout: {e.stdout}
                    Stderr: {e.stderr}
                    [yellow]Aborting further operations due to this error.[/yellow]"""
                    ),
                    title="Command Error",
                    expand=False,
                )
            )
            return False

        except Exception as e:
            console.print(
                Panel(
                    dedent(
                        f"""\
                    [bold red]An unexpected error occurred:[/bold red]

                    {str(e)}

                    [yellow]Aborting further operations due to this error.[/yellow]"""
                    ),
                    title="Unexpected Error",
                    expand=False,
                )
            )
            return False

    finally:
        os.chdir(original_dir)
        console.print("")

    return True


def create_session(token):
    session_id = secrets.token_urlsafe(32)
    expiration = datetime.now() + timedelta(hours=24)
    active_sessions[session_id] = {"token": token, "expiration": expiration}
    return session_id


def create_temp_sh(
        build_dir,
        flatpack_json_path: Path,
        temp_sh_path: Path,
        use_euxo: bool = False,
        hooks: list = None,
):
    if hooks is None:
        hooks = []
    else:
        logging.info("Using %d hooks passed to the function", len(hooks))

    try:
        with flatpack_json_path.open("r", encoding="utf-8") as infile:
            code_blocks = json.load(infile)

        def is_block_disabled(block):
            return block.get("disabled", False)

        def process_hook(hook, hook_index):
            required_fields = [
                "hook_name",
                "hook_type",
                "hook_script",
                "hook_placement",
            ]

            if not all(field in hook for field in required_fields):
                return

            if hook["hook_type"] == "bash":
                outfile.write(f"{hook['hook_script']}\n")
            elif hook["hook_type"] == "python":
                outfile.write(
                    f"\necho -e '\\033[1;33m[BEGIN] Executing hook {hook_index} (Python)\\033[0m'\n"
                )
                outfile.write(
                    send_code_to_python(
                        hook["hook_name"],
                        hook["hook_script"]
                    )
                )
                outfile.write(
                    f"\necho -e '\\033[1;33m[END] Executing hook {hook_index} (Python)\\033[0m'\n"
                )

        last_count = sum(
            1 for block in code_blocks if not is_block_disabled(block)
        )

        temp_sh_path.parent.mkdir(parents=True, exist_ok=True)

        with temp_sh_path.open("w", encoding="utf-8") as outfile:
            header_script = dedent(
                f"""\
                #!/bin/bash
                set -{'eux' if use_euxo else 'eu'}o pipefail
                
                executed_blocks=()

                cd "{build_dir}" || exit 1

                export SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
                echo "Script is running in $SCRIPT_DIR"
                echo "Current working directory: $(pwd)"

                if [ -n "${{VIRTUAL_ENV:-}}" ]; then
                    : # Already activated
                elif [ -f "$SCRIPT_DIR/bin/activate" ]; then
                    . "$SCRIPT_DIR/bin/activate"
                elif [ -f "$SCRIPT_DIR/Scripts/activate" ]; then
                    . "$SCRIPT_DIR/Scripts/activate"
                fi

                VENV_PYTHON=${{VIRTUAL_ENV:+$VIRTUAL_ENV/bin/python}}
                VENV_PYTHON=${{VENV_PYTHON:-$(command -v python3 || command -v python)}}

                [ ! -x "$VENV_PYTHON" ] && echo "Error: Python not found in virtual environment." && exit 1

                PYTHONPATH=${{PYTHONPATH:-}}
                export PYTHONPATH="$(dirname "$SCRIPT_DIR"):$PYTHONPATH"

                echo "Virtual environment is $VENV_PYTHON"
                
                rm -f "$SCRIPT_DIR/python_stdin" "$SCRIPT_DIR/python_stdout" "$SCRIPT_DIR/python_input"
                mkfifo -m 666 "$SCRIPT_DIR/python_stdin" "$SCRIPT_DIR/python_stdout" "$SCRIPT_DIR/python_input"

                "$VENV_PYTHON" -u "$SCRIPT_DIR/python_executor.py" < "$SCRIPT_DIR/python_stdin" > "$SCRIPT_DIR/python_stdout" 2>&1 &
                PYTHON_EXECUTOR_PID=$!

                exec 3> "$SCRIPT_DIR/python_stdin"
                exec 4< "$SCRIPT_DIR/python_stdout"
                exec 5> "$SCRIPT_DIR/python_input"

                datetime=$(date -u +"%Y-%m-%d %H:%M:%S")

                LAST_COUNT={last_count}

                EVAL_BUILD="$(dirname "$SCRIPT_DIR")/build/output/eval_build.json"
                EVAL_DATA="$(dirname "$SCRIPT_DIR")/build/output/eval_data.json"

                echo "SCRIPT_DIR=$SCRIPT_DIR"
                echo "EVAL_DATA=$EVAL_DATA"

                echo '[]' > "$EVAL_DATA"
                touch "$EVAL_DATA"
                """
            )

            outfile.write(header_script)
            outfile.write("\n")

            abort_handler = dedent(
                """\
                ABORT_FILE="$SCRIPT_DIR/abort_build"

                function check_abort() {
                    if [ -f "$ABORT_FILE" ]; then
                        echo "Abort requested, stopping build..."
                        cleanup
                        exit 1
                    fi
                }

                trap 'check_abort' DEBUG
                """
            )

            outfile.write(abort_handler)
            outfile.write("\n")

            log_eval_data_func = dedent(
                """\
                function log_eval_data() {
                    local block_index="$1"

                    echo -e "\033[1;36m[LOG] Running log_eval_data for block $block_index\033[0m"

                    files_since_last="$(find "${SCRIPT_DIR}" \\
                        -type f \\
                        -newer "${EVAL_DATA}" \\
                        -not -path "*/venv/*" \\
                        \\( \\
                            -name '*.gif' -o \\
                            -name '*.jpg' -o \\
                            -name '*.mp3' -o \\
                            -name '*.mp4' -o \\
                            -name '*.png' -o \\
                            -name '*.txt' -o \\
                            -name '*.wav' \\
                        \\) \\
                        2>/dev/null
                    )"

                    if [ -n "$files_since_last" ]; then
                        local eval_data_json="[]"

                        for file in $files_since_last; do
                            local file_basename="/output/$(basename "$file")"
                            local file_mime_type=$(file --mime-type -b "$file")

                            local file_json=$(jq -n \\
                                --arg eval "$block_index" \\
                                --arg file "$file" \\
                                --arg public "$file_basename" \\
                                --arg type "$file_mime_type" \\
                                '{eval: ($eval|tonumber), file: $file, public: $public, type: $type}')

                            eval_data_json=$(echo "$eval_data_json" | jq ". + [$file_json]")
                        done

                        echo "$eval_data_json" | jq -s '.[0] + .[1]' "$EVAL_DATA" > "$EVAL_DATA.tmp" && mv "$EVAL_DATA.tmp" "$EVAL_DATA"
                    fi

                    local eval_count

                    if [ "$block_index" -eq "$LAST_COUNT" ]; then
                        eval_count="null"
                    else
                        eval_count=$((block_index + 1))
                    fi

                    jq -nc --arg curr "$block_index" --arg last "$LAST_COUNT" --arg eval "$eval_count" --arg dt "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" '{\"curr\": ($curr|tonumber), \"last\": ($last|tonumber), \"eval\": (if $eval == \"null\" then null else ($eval|tonumber) end), \"datetime\": $dt }' | jq '.' > "$EVAL_BUILD"
                }
                """
            )
            outfile.write(log_eval_data_func)
            outfile.write("\n")

            cleanup_script = dedent(
                """\
                function cleanup() {
                    if [ -n "$PYTHON_EXECUTOR_PID" ]; then
                        echo "__EXIT_PYTHON_EXECUTOR__" >&3 2>/dev/null || true

                        sleep 0.5

                        if kill -0 $PYTHON_EXECUTOR_PID 2>/dev/null; then
                            kill -TERM $PYTHON_EXECUTOR_PID 2>/dev/null || true
                        fi
                    fi
                
                    local build_dir="$SCRIPT_DIR"
                    local current_pid=$$
                    local parent_pid=$PPID
                
                    children=$(pgrep -P $current_pid)
                    
                    for child in $children; do
                        if [ "$child" != "$parent_pid" ] && [ "$child" != "$current_pid" ]; then
                            kill -TERM $child 2>/dev/null || true
                        fi
                    done
                
                    ps -ef | grep python | grep "$build_dir" | grep -v grep | while read -r pid rest; do
                        if [ "$pid" != "$parent_pid" ] && [ "$pid" != "$current_pid" ]; then
                            kill -TERM $pid 2>/dev/null || true
                        fi
                    done
                }
                
                trap 'cleanup' EXIT INT TERM
                """
            )
            outfile.write(cleanup_script)
            outfile.write("\n")

            helper_functions = dedent(
                """\
                function send_input_to_python() {
                    # echo "Sending input to Python: '$1'"
                    echo "$1" >&5
                }

                function send_code_to_python_and_wait() {
                    lsof -p $PYTHON_EXECUTOR_PID 2>/dev/null | grep FIFO >/dev/null
                    
                    echo "" >&3
                    
                    cat >&3
                    echo '__END_CODE_BLOCK__' >&3
                    output=""
                    
                    while IFS= read -r line <&4; do
                        if [[ ! "$line" == *"EXECUTION_COMPLETE"* ]] && [[ ! "$line" == *"READY_FOR_INPUT"* ]]; then
                            echo "$line"
                        fi

                        if [[ -z "$line" ]]; then
                            continue
                        fi

                        output+="$line"$'\\n'

                        if [[ "$output" == *"READY_FOR_INPUT"* ]]; then
                            read -r user_input < /dev/tty
                            send_input_to_python "$user_input"
                            output=""
                        elif [[ "$output" == *"EXECUTION_FAILED"* ]]; then
                            exit 1
                        elif [[ "$output" == *"EXECUTION_COMPLETE"* ]]; then
                            break
                        elif [[ "$output" == *"Error executing code:"* ]]; then
                            echo "$output" >&2
                            exit 1
                        fi
                    done
                }
                """
            )
            outfile.write(helper_functions)
            outfile.write("\n")

            def send_code_to_python(id, code):
                return dedent(
                    """\
                    send_code_to_python_and_wait << 'EOF_CODE'
                    {}
                    EOF_CODE
                    """
                ).format(code)

            for hook_index, hook in enumerate(hooks):
                if hook.get("hook_placement", "").strip().lower() == "before":
                    process_hook(hook, hook_index)

            for block_index, block in enumerate(code_blocks):
                if is_block_disabled(block):
                    continue

                id = block.get("id")
                code = block.get("code", "")
                language = block.get("type")

                outfile.write('echo -e ""\n')  # Empty line

                outfile.write(
                    f"\necho -e '\\033[1;35m[DEBUG] INDEX {block_index} / ID {id} / LANGUAGE {language}\\033[0m'\n"
                )

                if language == "bash":
                    code_single_line = "".join(code.splitlines())

                    dangerous_patterns = [
                        r"\$\((rm|chmod|chown|sudo|su|eval).*?\)",
                        r"\$\{.*?\}",
                        r"\.\s+\S+",
                        r"/\s*$",
                        r"`.*`",
                        r"^\s*\bbase64\b\s",
                        r"^\s*\bbash\b\s",
                        r"^\s*\bcat\b\s+/etc/passwd",
                        r"^\s*\bcat\b\s+/etc/shadow",
                        r"^\s*\bchroot\b\s",
                        r"^\s*\bdd\b\s",
                        r"^\s*\benv\b\s",
                        r"^\s*\beval\b\s",
                        r"^\s*\bfdisk\b\s",
                        r"^\s*\bfind\b\s+(?!.*-exec).*$",
                        r"^\s*\bfish\b\s",
                        r"^\s*\bkillall\b\s",
                        r"^\s*\bksh\b\s",
                        r"^\s*\blvcreate\b\s",
                        r"^\s*\blvextend\b\s",
                        r"^\s*\blvreduce\b\s",
                        r"^\s*\blvremove\b\s",
                        r"^\s*\bmkfs\b\s",
                        r"^\s*\bmount\b\s",
                        r"^\s*\bnc\b\s",
                        r"^\s*\bncat\b\s",
                        r"^\s*\bparted\b\s",
                        r"^\s*\bperl\b\s+(-c)?",
                        r"^\s*\bpgrep\b\s",
                        r"^\s*\bphp\b\s+(-c)?",
                        r"^\s*\bpkill\b\s",
                        r"^\s*\bpv\b\s",
                        r"^\s*\bpython\b\s+(-c)?",
                        r"^\s*\breboot\b\s",
                        r"^\s*\brm\b\s+.*--no-preserve-root",
                        r"^\s*\brm\b\s+.*-r",
                        r"^\s*\bruby\b\s+(-c)?",
                        r"^\s*\bshutdown\b\s",
                        r"^\s*\bsource\b\s",
                        r"^\s*\bsu\b\s",
                        r"^\s*\bsudo\b\s",
                        r"^\s*\bsystemctl\b\s+(?!start|stop|restart|status)",
                        r"^\s*\btrap\b\s",
                        r"^\s*\bumount\b\s",
                        r"^\s*\buseradd\b\s",
                        r"^\s*\buserdel\b\s",
                        r"^\s*\busermod\b\s",
                        r"^\s*\bvgchange\b\s",
                        r"^\s*\bvgcreate\b\s",
                        r"^\s*\bvgremove\b\s",
                        r"^\s*\bzsh\b\s"
                    ]

                    outfile.write(
                        f"\necho -e '\\033[1;33m[BEGIN] Executing block {block_index} (Bash)\\033[0m'\n"
                    )

                    outfile.write(f'block_start_time=$(date +%s.%N)\n')

                    outfile.write(f"{code}\n")

                    for pattern in dangerous_patterns:
                        if re.search(pattern, code_single_line):
                            outfile.write(
                                f"echo -e \"\\033[31m[ERROR] Commands matching '{pattern}' are not allowed.\\033[0m\"\n"
                            )
                            outfile.write("exit 1\n")

                    if re.search(r"pip\s+", code_single_line):
                        outfile.write(
                            f"echo -e \"\\033[31m[ERROR] 'pip' commands are not allowed in code blocks. Please use the flatpack.toml file.\\033[0m\"\n"
                        )
                        outfile.write("exit 1\n")

                    outfile.write(
                        f"\necho -e '\\033[1;33m[END] Executing block {block_index} (Bash)\\033[0m'\n"
                    )

                    outfile.write(f'executed_blocks+=("{id}")\n')

                    outfile.write(
                        f'block_end_time=$(date +%s.%N)\n'
                    )
                    outfile.write(
                        f'total_duration_ms=$(echo "($block_end_time - $block_start_time) * 1000" | bc)\n'
                    )
                    outfile.write(
                        f'total_execution_times="${{total_execution_times:+$total_execution_times }}$total_duration_ms"\n'
                    )
                    outfile.write(
                        'echo -e "\\033[1;32m[DEBUG] Block duration: $total_duration_ms milliseconds\\033[0m"\n'
                    )

                elif language == "python":
                    code_lines = code.splitlines()

                    dangerous_python_functions = [
                        "__builtins__",
                        "__import__",
                        "compile",
                        "eval",
                        "exec",
                        "execfile",
                        "fcntl",
                        "imp",
                        "importlib",
                        "importlib.import_module",
                        "importlib.util",
                        "os.chmod",
                        "os.chown",
                        "os.chroot",
                        "os.execv",
                        "os.execve",
                        "os.link",
                        "os.mknod",
                        "os.path.join",
                        "os.popen",
                        "os.putenv",
                        "os.removedirs",
                        "os.rename",
                        "os.replace",
                        "os.setpgrp",
                        "os.setregid",
                        "os.setresuid",
                        "os.setreuid",
                        "os.setsid",
                        "os.setuid",
                        "os.symlink",
                        "os.system",
                        "platform.os",
                        "platform.popen",
                        "platform.processor",
                        "platform.python_branch",
                        "platform.python_build",
                        "platform.python_compiler",
                        "platform.python_implementation",
                        "platform.python_revision",
                        "platform.python_version",
                        "platform.python_version_tuple",
                        "platform.release",
                        "platform.subprocess",
                        "platform.system",
                        "platform.system_alias",
                        "platform.uname",
                        "platform.version",
                        "platform.win32_edition",
                        "platform.win32_is_iot",
                        "platform.win32_ver",
                        "pty",
                        "resource",
                        "shutil.copy2",
                        "shutil.copymode",
                        "shutil.copystat",
                        "shutil.copytree",
                        "shutil.make_archive",
                        "shutil.rmtree",
                        "subprocess.Popen",
                        "subprocess.call",
                        "subprocess.check_call",
                        "subprocess.check_output",
                        "subprocess.getoutput",
                        "subprocess.getstatusoutput",
                        "tempfile.mkdtemp",
                        "tempfile.SpooledTemporaryFile",
                        "tempfile.TemporaryFile",
                        "winreg"
                    ]

                    outfile.write(
                        f"\necho -e '\\033[1;33m[BEGIN] Executing block {block_index} (Python)\\033[0m'\n"
                    )

                    outfile.write(f'block_start_time=$(date +%s.%N)\n')

                    for line_number, line in enumerate(code_lines, 1):
                        in_comment = False
                        in_string = False

                        for i, char in enumerate(line):
                            if char == "#" and not in_string:
                                in_comment = True
                            elif char in ("'", "\"") and not in_comment:
                                in_string = not in_string
                            elif char == "\\" and in_string:
                                i += 1

                            if not in_string and not in_comment:
                                for func in dangerous_python_functions:
                                    if re.search(
                                            r"\b" + re.escape(func) + r"\s*\(",
                                            line[i:]):
                                        outfile.write(
                                            f"echo -e \"\\033[31m[ERROR] Line {line_number}: Potentially dangerous function '{func}' is not allowed.\\033[0m\"\n"
                                        )
                                        outfile.write("exit 1\n")

                    outfile.write(send_code_to_python(id, code))
                    outfile.write(
                        f"\necho -e '\\033[1;33m[END] Executing block {block_index} (Python)\\033[0m'\n"
                    )

                    outfile.write(f'executed_blocks+=("{id}")\n')

                    outfile.write(
                        f'block_end_time=$(date +%s.%N)\n'
                    )
                    outfile.write(
                        f'total_duration_ms=$(echo "($block_end_time - $block_start_time) * 1000" | bc)\n'
                    )
                    outfile.write(
                        f'total_execution_times="${{total_execution_times:+$total_execution_times }}$total_duration_ms"\n'
                    )
                    outfile.write(
                        'echo -e "\\033[1;32m[DEBUG] Block duration: $total_duration_ms milliseconds\\033[0m"\n'
                    )

                outfile.write(f'log_eval_data "{block_index}"\n')

            outfile.write('echo -e ""\n')  # Empty line

            outfile.write(dedent(
                """\
                echo -e "\\033[1;32m[BEGIN] Executed blocks\\033[0m"
                executed_blocks_counter=0
                if [ ${#executed_blocks[@]:-0} -gt 0 ]; then
                    for block_id in "${executed_blocks[@]}"; do
                        echo "($executed_blocks_counter) $block_id"
                        ((executed_blocks_counter++))
                    done
                fi
                echo -e "\\033[1;32mTotal: $executed_blocks_counter\\033[0m"
                echo -e "\\033[1;32m[END] Executed blocks\\033[0m"
                """
            ))

            for hook_index, hook in enumerate(hooks):
                if hook.get("hook_placement", "").strip().lower() == "after":
                    process_hook(hook, hook_index)

            temp_sh_path.chmod(0o755)

    except Exception as e:
        logging.error(
            "An error occurred while creating temp script: %s", e,
            exc_info=True
        )
        raise


def create_venv(venv_dir: str):
    """Create a virtual environment in the specified directory using the current Python version."""
    python_executable = sys.executable

    try:
        subprocess.run(
            [python_executable, "-m", "venv", venv_dir],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(
            "Virtual environment created successfully in '%s' using %s.",
            venv_dir,
            python_executable,
        )
        console.print(
            f"[bold green]SUCCESS:[/bold green] Virtual environment created successfully in '{venv_dir}' using {python_executable}."
        )
    except subprocess.CalledProcessError as e:
        logger.error("Failed to create virtual environment: %s", e.stderr)
        console.print(
            f"[bold red]ERROR:[/bold red] Failed to create virtual environment: {e.stderr}"
        )
    except Exception as e:
        logger.error(
            "An unexpected error occurred while creating virtual environment: %s",
            e
        )
        console.print(
            f"[bold red]ERROR:[/bold red] An unexpected error occurred while creating virtual environment: {e}"
        )


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
        unsigned_token = request.app.state.signer.unsign(
            csrf_token_cookie, max_age=3600
        )
        timestamp, token = unsigned_token.decode().split(":")

        if not secrets.compare_digest(token, csrf_token_header):
            raise HTTPException(status_code=403, detail="CSRF token invalid")

    except (SignatureExpired, BadSignature):
        raise HTTPException(
            status_code=403,
            detail="CSRF token expired or invalid"
        )


def ensure_database_initialized():
    global db_manager, flatpack_directory

    if db_manager is None:

        if flatpack_directory is None:
            raise ValueError("flatpack_directory is not set")

        db_path = os.path.join(flatpack_directory, "build", "flatpack.db")
        logger.info("Initializing database at %s", db_path)
        db_manager = DatabaseManager(db_path)
        db_manager.initialize_database()
        logger.info("Database initialized successfully")


def generate_secure_token(length=8):
    """Generate a secure token of the specified length.

    Args:
        length (int): The length of the token to generate. Default is 8.

    Returns:
        str: A securely generated token.
    """
    alphabet = string.ascii_letters + string.digits + "-._~"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def get_all_hooks_from_database():
    global db_manager
    ensure_database_initialized()
    try:
        return db_manager.get_all_hooks()
    except Exception as e:
        logger.error("An error occurred while fetching hooks: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while fetching hooks: {e}"
        )


def get_executable_path(executable):
    """Securely get the full path of an executable."""
    if sys.platform.startswith("win"):
        system_root = os.environ.get("SystemRoot", "C:\\Windows")
        where_cmd = os.path.join(system_root, "System32", "where.exe")
        try:
            result = subprocess.run(
                [where_cmd, executable], check=True, capture_output=True,
                text=True
            )
            paths = result.stdout.strip().split("\n")
            return paths[0] if paths else None
        except subprocess.CalledProcessError:
            return None
    else:
        try:
            which_cmd = "/usr/bin/which"
            if not os.path.exists(which_cmd):
                which_cmd = "/bin/which"

            result = subprocess.run(
                [which_cmd, executable], check=True, capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None


def get_process_tree(pid: int) -> Set[int]:
    """Recursively get all child process IDs in the process tree."""
    try:
        process = psutil.Process(pid)
        children = process.children(recursive=True)
        return {pid} | {child.pid for child in children}
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return set()


def get_secret_key():
    return secrets.token_urlsafe(32)


def get_token() -> Optional[str]:
    """Retrieve the token from the configuration file."""
    config = load_config()
    return config.get("token")


def initialize_fastapi_app(secret_key):
    """Initialize and configure a FastAPI application."""
    CORSMiddleware = lazy_import(
        "fastapi.middleware.cors",
        callable_name="CORSMiddleware"
    )
    SessionMiddleware = lazy_import(
        "starlette.middleware.sessions",
        callable_name="SessionMiddleware"
    )

    fastapi_app = FastAPI(openapi_url=None)

    fastapi_app.add_middleware(SessionMiddleware, secret_key=secret_key)
    fastapi_app.state.signer = TimestampSigner(secret_key)

    uvicorn_logger = logging.getLogger("uvicorn.access")
    uvicorn_logger.addFilter(EndpointFilter())

    origins = ["*"]

    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    setup_routes(fastapi_app)

    return fastapi_app


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
        logger.warning("Configuration file does not exist: %s",
                       CONFIG_FILE_PATH)
        return {}

    try:
        with open(CONFIG_FILE_PATH, "r") as config_file:
            config = toml.load(config_file)
            return config
    except Exception as e:
        error_message = f"Error loading config: {e}"
        logger.error("Error loading config: %s", e)
        return {}


async def run_build_process(schedule_id=None):
    global abort_requested, build_in_progress

    build_in_progress = True
    logger.info("Running build process...")

    try:
        await update_build_status("in_progress", schedule_id)

        steps = [
            ("Preparing build environment", 1),
            ("Compiling source code", 1),
            ("Running tests", 1),
            ("Packaging application", 1),
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

        await update_build_status("in_progress: Running build script",
                                  schedule_id)
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
    filename = unicodedata.normalize("NFKD", filename)
    filename = filename.encode("ascii", "ignore").decode("ascii")
    filename = re.sub(r"[^\w\.-]", "_", filename)
    filename = filename.strip("._")
    return filename


def set_token(token: str):
    try:
        config = load_config()
        config["token"] = token
        save_config(config)
        logger.info("Token set successfully.")
    except Exception as e:
        logger.error("Failed to set token: %s", str(e))


def setup_static_directory(fastapi_app, directory: str):
    """Setup the static directory for serving static files, excluding 'output' folder for unauthenticated users."""
    StaticFiles = lazy_import(
        "fastapi.staticfiles",
        callable_name="StaticFiles"
    )

    global flatpack_directory
    flatpack_directory = os.path.abspath(directory)

    if os.path.exists(flatpack_directory) and os.path.isdir(
            flatpack_directory):
        static_dir = os.path.join(flatpack_directory, "web")

        @fastapi_app.middleware("http")
        async def block_output_directory(request: Request, call_next):
            if request.url.path.startswith("/output/"):
                session_id = request.cookies.get("session_id")
                if session_id and is_user_logged_in(session_id):
                    return await call_next(request)
                return JSONResponse(
                    status_code=403,
                    content={
                        "detail": "Access to the /output directory is forbidden for unauthenticated users"
                    },
                )
            return await call_next(request)

        fastapi_app.mount(
            "/",
            StaticFiles(
                directory=static_dir,
                follow_symlink=True,
                html=True
            ),
            name="static"
        )

        logger.info("Static files will be served from: %s", static_dir)
    else:
        logger.error(
            "The directory '%s' does not exist or is not a directory.",
            flatpack_directory,
        )
        raise ValueError(
            f"The directory '{directory}' does not exist or is not a directory."
        )


def shutdown_server():
    """Shutdown the FastAPI server and Next.js process."""
    global nextjs_process

    logging.getLogger("uvicorn.error").info(
        "Shutting down the server after maximum validation attempts."
    )

    if nextjs_process:
        try:
            nextjs_process.terminate()
            nextjs_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            nextjs_process.kill()

    os._exit(0)


def strip_html(script):
    BeautifulSoup = lazy_import("bs4", callable_name="BeautifulSoup")
    MarkupResemblesLocatorWarning = lazy_import(
        "bs4", callable_name="MarkupResemblesLocatorWarning"
    )
    warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
    soup = BeautifulSoup(script, "html.parser")
    return soup.get_text()


def sync_connections_from_file():
    """Load connections from connections.json file and sync them to the database."""
    global flatpack_directory

    try:
        connections_file = os.path.join(flatpack_directory, CONNECTIONS_FILE)
        if not os.path.exists(connections_file):
            logger.info("No connections.json file found for initial sync")
            return

        with open(connections_file, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "connections" not in data:
            logger.warning(
                "Invalid format in connections.json - expected {'connections': [...]}"
            )
            return

        connections = []
        for conn in data["connections"]:
            try:
                mapping = SourceHookMapping(
                    sourceId=conn["source_id"],
                    targetId=conn["target_id"],
                    sourceType=conn["source_type"],
                    targetType=conn["target_type"],
                )
                connections.append(mapping)
            except Exception as e:
                logger.warning(
                    "Invalid connection entry in connections.json: %s", e)
                continue

        if connections:
            save_connections_to_file_and_db(connections)
            logger.info(
                "Successfully synced %d connections from file to database",
                len(connections),
            )
    except Exception as e:
        logger.error("Error syncing connections from file: %s", e)


def sync_sources_from_file():
    """Load sources from sources.json file and sync them to the database."""
    global flatpack_directory

    try:
        sources_file = os.path.join(flatpack_directory, "sources.json")
        if not os.path.exists(sources_file):
            logger.info("No sources.json file found for initial sync")
            return

        with open(sources_file, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "sources" not in data:
            logger.warning(
                "Invalid format in sources.json - expected {'sources': [...]}"
            )
            return

        sources = []
        for source in data["sources"]:
            try:
                source_obj = Source(
                    source_name=source["source_name"],
                    source_type=source["source_type"],
                    source_details=source.get("source_details"),
                )
                sources.append(source_obj)
            except Exception as e:
                logger.warning("Invalid source entry in sources.json: %s", e)
                continue

        if sources:
            ensure_database_initialized()
            for source in sources:
                try:
                    db_manager.add_source(
                        source.source_name, source.source_type,
                        source.source_details
                    )
                except Exception as e:
                    logger.warning(
                        "Source %s might already exist in the database: %s",
                        source.source_name,
                        e,
                    )

            logger.info(
                "Successfully synced %d sources from file to database",
                len(sources)
            )
    except Exception as e:
        logger.error("Error syncing sources from file: %s", e)


def unescape_content_parts(content: str) -> str:
    """Unescape special characters within content parts."""
    parts = content.split("part_")
    unescaped_content = parts[0]
    for part in parts[1:]:
        if part.startswith('bash """') or part.startswith('python """'):
            type_and_content = part.split('"""', 1)
            if len(type_and_content) > 1:
                type_and_header, code = type_and_content
                code, footer = code.rsplit('"""', 1)
                unescaped_content += f'part_{type_and_header}"""{unescape_special_chars(code)}"""{footer}'
            else:
                unescaped_content += f"part_{part}"
        else:
            unescaped_content += f"part_{part}"
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
        "schedule_id": schedule_id,
    }
    if error:
        status_data["error"] = str(error)

    status_file = os.path.join(flatpack_directory, "build",
                               "build_status.json")

    try:
        os.makedirs(os.path.dirname(status_file), exist_ok=True)
        with open(status_file, "w") as f:
            json.dump(status_data, f)
        logging.info("Updated build status: %s", status)
    except Exception as e:
        logging.error("Failed to update build status: %s", e)


def validate_api_token(api_token: str) -> bool:
    """Validate the API token."""
    return api_token == get_token()


def validate_file_path(path, is_input=True, allowed_dir=None):
    """Validate the file path to prevent directory traversal attacks."""
    absolute_path = os.path.abspath(path)

    if allowed_dir:
        allowed_dir_absolute = os.path.abspath(allowed_dir)

        if not absolute_path.startswith(allowed_dir_absolute):
            raise ValueError(
                f"Path '{path}' is outside the allowed directory '{allowed_dir}'."
            )

    if is_input:
        if not os.path.exists(absolute_path):
            raise FileNotFoundError(
                f"The path '{absolute_path}' does not exist."
            )

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


async def fpk_build(directory: Union[str, None], use_euxo: bool = False):
    """Asynchronous function to build a flatpack with connection validation."""
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
        logger.error(
            "No cached flatpack found, and no valid directory provided."
        )
        raise ValueError(
            "No cached flatpack found, and no valid directory provided."
        )

    flatpack_dir = Path(last_unboxed_flatpack)

    if not flatpack_dir.exists():
        logger.error(
            "The flatpack directory '%s' does not exist.",
            flatpack_dir
        )
        raise ValueError(
            f"The flatpack directory '{flatpack_dir}' does not exist."
        )

    build_dir = flatpack_dir / "build"

    if not build_dir.exists():
        logger.error("The build directory is missing. Did you unbox?")
        raise ValueError("The build directory is missing. Did you unbox?")

    sync_hooks_to_db_on_startup()
    sync_connections_from_file()

    console = Console()

    for directory in (build_dir, flatpack_dir):
        connections_file = directory / "connections.json"

        if connections_file.exists():
            break
    else:
        logger.error(
            "connections.json not found in either %s or %s",
            build_dir,
            flatpack_dir
        )
        raise FileNotFoundError(
            f"connections.json not found in {build_dir} or {flatpack_dir}"
        )

    try:
        with open(connections_file, "r") as f:
            connections_data = json.load(f)

        connections = connections_data.get("connections", [])
        hooks = load_and_get_hooks()
        hook_names = {hook["hook_name"] for hook in hooks}

        if connections:
            table = Table(title="Connections")
            table.add_column("Source ID", style="cyan")
            table.add_column("Source type", style="blue")
            table.add_column("Target ID", style="green")

            invalid_connections = []

            for connection in connections:
                source_id = connection.get("source_id")
                source_type = connection.get("source_type", "N/A")
                target_id = connection.get("target_id")

                matched_hook = None

                for hook_name in hook_names:
                    if target_id and target_id.startswith(hook_name + "-"):
                        matched_hook = hook_name
                        break

                if not matched_hook:
                    invalid_connections.append(connection)
                    table.add_row(
                        source_id,
                        source_type,
                        f"[red]{target_id}[/red]"
                    )
                else:
                    table.add_row(source_id, source_type, target_id)

            console.print(table)
            console.print("")

            if invalid_connections:
                console.print(
                    "[red]Found invalid connections referencing non-existent hooks:[/red]"
                )

                for conn in invalid_connections:
                    console.print(
                        f"[red] - Source: {conn.get('source_id')}, Target: {conn.get('target_id')}[/red]"
                    )
                raise ValueError(
                    "Invalid connections found. Build process canceled."
                )

        logger.info("Successfully validated %d connections", len(connections))
    except json.JSONDecodeError:
        logger.error("Invalid JSON format in connections.json")
        raise ValueError("Invalid JSON format in connections.json")

    flatpack_json_path = build_dir / "flatpack.json"

    if not flatpack_json_path.exists() or not flatpack_json_path.is_file():
        logger.error(
            "flatpack.json not found in %s. Build process canceled.",
            build_dir
        )
        raise FileNotFoundError(
            f"flatpack.json not found in {build_dir}. Build process canceled."
        )

    temp_sh_path = build_dir / "temp.sh"

    if temp_sh_path.exists():
        temp_sh_path.unlink()

    create_temp_sh(
        build_dir,
        flatpack_json_path,
        temp_sh_path,
        use_euxo=use_euxo,
        hooks=hooks
    )

    lines = temp_sh_path.read_text().splitlines()

    start = 0

    while start < len(lines) and not lines[start].strip():
        start += 1

    end = len(lines)

    while end > start and not lines[end - 1].strip():
        end -= 1

    temp_sh_path.write_text("\n".join(lines[start:end]), encoding="utf-8")

    building_script_path = build_dir / "build.sh"

    if not building_script_path.exists() or not building_script_path.is_file():
        logger.error("Building script not found in %s", build_dir)
        raise FileNotFoundError(f"Building script not found in {build_dir}.")

    log_dir = build_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = f"build_{datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')}.log"
    build_log_file_path = log_dir / log_filename

    stdbuf_cmd = f"stdbuf -i0 -o0 -e0 python3 -u -c {shlex.quote('import sys; sys.stdout = sys.stderr')}"
    bash_cmd = f"bash {shlex.quote(str(building_script_path))}"
    tee_cmd = f"tee {shlex.quote(str(build_log_file_path))}"

    full_command = f"{stdbuf_cmd} && {bash_cmd} | {tee_cmd}"

    process = subprocess.Popen(
        full_command,
        bufsize=1,
        shell=True,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
        universal_newlines=True
    )

    process.wait()

    web_dir = flatpack_dir / "web"

    if not web_dir.exists():
        logger.error("The web directory '%s' does not exist.", web_dir)
        raise FileNotFoundError(
            f"The web directory '{web_dir}' does not exist."
        )

    output_dir = build_dir / "output"
    eval_build_path = output_dir / "eval_build.json"
    eval_data_path = output_dir / "eval_data.json"

    if not eval_data_path.exists():
        logger.error(
            "The 'eval_data.json' file does not exist in '%s'.",
            output_dir
        )
        raise FileNotFoundError(
            f"The 'eval_data.json' file does not exist in '{output_dir}'."
        )

    with eval_data_path.open("r") as file:
        eval_data = json.load(file)
        for item in eval_data:
            original_path = Path(item["file"])
            relative_path = None
            for parent in original_path.parents:
                if parent.parts[-1] == "build":
                    relative_path = original_path.relative_to(parent)
                    break

            if relative_path:
                source_file = build_dir / relative_path

                if source_file.exists():
                    allowed_mimetypes = [
                        "audio/wav",
                        "audio/x-wav",
                        "image/jpeg",
                        "image/png",
                        "text/plain",
                        "video/mp4",
                    ]
                    mime_type, _ = mimetypes.guess_type(source_file)
                    if mime_type in allowed_mimetypes:
                        dest_path = output_dir / source_file.name
                        shutil.copy2(source_file, dest_path)
                        logger.info("Copied %s to %s", source_file, dest_path)
                else:
                    logger.error("File %s not found.", source_file)
            else:
                logger.error(
                    "Could not determine relative path for %s",
                    original_path
                )


def fpk_cache_unbox(directory_name: str):
    """Cache the last unboxed flatpack's directory name to a file.

    Args:
        directory_name (str): The name of the directory to cache.

    Returns:
        None
    """
    cache_file_path = HOME_DIR / ".fpk_unbox.cache"

    try:
        with open(cache_file_path, "w") as f:
            f.write(directory_name)

        logger.info(
            "Cached the directory name '%s' to %s", directory_name,
            cache_file_path
        )

    except IOError as e:
        logger.error("Failed to cache the directory name '%s': %s",
                     directory_name, e)


def fpk_check_ngrok_auth():
    """
    Check if the NGROK_AUTHTOKEN environment variable is set.

    Raises:
        EnvironmentError: If the NGROK_AUTHTOKEN is not set.
    """
    ngrok_auth_token = os.environ.get("NGROK_AUTHTOKEN")

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
    if not re.match(r"^[a-z0-9-]+$", flatpack_name):
        raise ValueError(
            "Invalid name format. Only lowercase letters, numbers, and hyphens are allowed."
        )

    flatpack_name = flatpack_name.lower().replace(" ", "-")
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
            if item in [".deepsource.toml", ".gitignore", "LICENSE"]:
                continue
            s = os.path.join(template_dir, item)
            d = os.path.join(flatpack_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

        logger.info("Copied template files to flatpack directory: %s",
                    flatpack_dir)

        files_to_edit = [
            (
                os.path.join(flatpack_dir, "README.md"),
                r"# template",
                f"# {flatpack_name}",
            ),
            (
                os.path.join(flatpack_dir, "flatpack.toml"),
                r"{{model_name}}",
                flatpack_name,
            ),
            (
                os.path.join(flatpack_dir, "build.sh"),
                r"export DEFAULT_REPO_NAME=template",
                f"export DEFAULT_REPO_NAME={flatpack_name}",
            ),
            (
                os.path.join(flatpack_dir, "build.sh"),
                r"export FLATPACK_NAME=template",
                f"export FLATPACK_NAME={flatpack_name}",
            ),
        ]

        for file_path, pattern, replacement in files_to_edit:
            with open(file_path, "r") as file:
                filedata = file.read()
            newdata = re.sub(pattern, replacement, filedata)
            with open(file_path, "w") as file:
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
        local,
    )

    disclaimer_message = disclaimer_template.format(
        please_note=please_note_content
    ).strip()
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

        default_branch = repo_info["default_branch"]

        zip_url = f"{repo_url}/zipball/{default_branch}"

        zip_response = requests.get(zip_url)
        zip_response.raise_for_status()

        with ZipFile(BytesIO(zip_response.content)) as zip_ref:
            top_level_dir = zip_ref.namelist()[0].split("/")[0]
            zip_ref.extractall(dest_dir)

        extracted_dir = os.path.join(dest_dir, top_level_dir)
        os.rename(extracted_dir, template_dir)

        files_to_remove = [
            "app.css",
            "app.js",
            "index.html",
            "package.json",
            "robotomono.woff2",
        ]

        for file in files_to_remove:
            file_path = os.path.join(template_dir, file)
            if os.path.exists(file_path):
                os.remove(file_path)
            else:
                print(f"File not found: {file}")

        logger.info(
            "Downloaded and extracted template from %s to %s", repo_url,
            dest_dir
        )

        return template_dir
    except requests.RequestException as e:
        error_message = f"Failed to download template from {repo_url}: {e}"
        logger.error("%s", error_message)
        raise RuntimeError(error_message)
    except OSError as e:
        error_message = (
            f"Failed to extract template or remove index.html in {dest_dir}: {e}"
        )
        logger.error("%s", error_message)
        raise RuntimeError(error_message)


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
        with open(GITHUB_CACHE, "r") as f:
            cache_data = json.load(f)

        cache_time = datetime.fromisoformat(cache_data["timestamp"])

        if datetime.now() - cache_time < GITHUB_CACHE_EXPIRY:
            return cache_data["directories"]

    try:
        response = session.get(f"{GITHUB_REPO_URL}/contents/warehouse")
        response.raise_for_status()
        json_data = response.json()

        if isinstance(json_data, list):
            directories = [
                item["name"]
                for item in json_data
                if (
                        isinstance(item, dict)
                        and item.get("type") == "dir"
                        and item.get("name", "").lower()
                )
            ]

            directories = sorted(directories)

            os.makedirs(HOME_DIR, exist_ok=True)

            with open(GITHUB_CACHE, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "directories": directories,
                    }, f
                )

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

    model_file_formats = [
        ".caffemodel",
        ".ckpt",
        ".gguf",
        ".h5",
        ".mar",
        ".mlmodel",
        ".model",
        ".onnx",
        ".params",
        ".pb",
        ".pkl",
        ".pickle",
        ".pt",
        ".pth",
        ".sav",
        ".tflite",
        ".weights",
    ]
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
        api_key = config.get("api_key")

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
            with cache_file_path.open("r") as cache_file:
                last_flatpack = cache_file.read().strip()
                logger.info(
                    "Last unboxed flatpack directory retrieved: %s",
                    last_flatpack
                )
                return last_flatpack
        else:
            logger.warning("Cache file does not exist: %s", cache_file_path)
    except OSError as e:
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
    data_dir = getattr(args, "data_dir", ".")

    logger.info("Initializing Vector Manager and data directory: %s", data_dir)

    return VectorManager(model_id="all-MiniLM-L6-v2", directory=data_dir)


def fpk_is_raspberry_pi() -> bool:
    """Check if we're running on a Raspberry Pi.

    Returns:
        bool: True if running on a Raspberry Pi, False otherwise.
    """
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Hardware") and "BCM" in line:
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


def setup_nextjs_project(app_dir: Path) -> bool:
    """Set up a new Next.js project in the specified directory."""
    if app_dir.exists():
        shutil.rmtree(app_dir)

    create_next_app_cmd = [
        "npx",
        "create-next-app@latest",
        str(app_dir),
        "--import-alias",
        "@/*",
        "--tailwind",
        "--typescript",
        "--no-app",
        "--no-eslint",
        "--no-experimental-app",
        "--no-src-dir",
        "--no-turbopack",
        "--use-npm"
    ]

    try:
        with console.status(
                "[bold green]Setting up a new Next.js project...",
                spinner="dots"
        ):
            subprocess.run(
                create_next_app_cmd,
                input='y\n',
                text=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        console.print(
            "[bold green]Successfully set up a new Next.js project[/bold green]"
        )
        console.print("")
        return True
    except subprocess.CalledProcessError as e:
        logger.error("Failed to set up Next.js project: %s", e)
        return False


def copy_files_to_web_dir(
        files_to_download: Dict[str,
        List[str]],
        web_dir: Path,
        session: httpx.Client
) -> bool:
    """Download and copy template files to the web directory."""
    for file in files_to_download["web"]:
        try:
            file_url = f"{TEMPLATE_REPO_URL}/contents/{file}"
            response = session.get(file_url)
            response.raise_for_status()

            file_content = response.json()["content"]
            file_decoded = base64.b64decode(file_content)
            file_path = web_dir / file

            mode = "wb" if file.endswith(".woff2") else "w"
            encoding = None if file.endswith(".woff2") else "utf-8"

            with open(file_path, mode, encoding=encoding) as f:
                if mode == "wb":
                    f.write(file_decoded)
                else:
                    f.write(file_decoded.decode("utf-8"))
        except Exception as e:
            logger.error("Failed to download or save %s: %s", file, e)
            return False
    return True


def cleanup_flatpack_dir(flatpack_dir: Path) -> None:
    """Safely clean up the flatpack directory if it exists."""
    if flatpack_dir.exists():
        try:
            shutil.rmtree(
                flatpack_dir,
                ignore_errors=True
            )
        except Exception as e:
            logger.error(
                "Failed to clean up directory %s: %s",
                flatpack_dir, e
            )


def setup_flatpack_directory(
        directory_name: str,
        session: httpx.Client
) -> Tuple[bool, Path]:
    """Set up initial flatpack directory structure."""
    flatpack_dir = Path.cwd() / directory_name

    if flatpack_dir.exists():
        if (flatpack_dir / "build").exists():
            logger.error(
                "Directory '%s' already contains a build directory",
                directory_name
            )
            return False, None

    try:
        flatpack_dir.mkdir(parents=True, exist_ok=True)

        web_dir = flatpack_dir / "web"
        build_dir = flatpack_dir / "build"
        app_dir = web_dir / "app"
        output_dir = build_dir / "output"

        for dir_path in [web_dir, build_dir, output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info("Created directory: %s", dir_path)

        eval_data_path = output_dir / "eval_data.json"

        if not eval_data_path.exists():
            with open(eval_data_path, "w") as f:
                json.dump([], f)

        files_to_download = {
            "web": [
                "app.css",
                "app.js",
                "index.html",
                "package.json",
                "robotomono.woff2",
            ]
        }

        if not copy_files_to_web_dir(
                files_to_download,
                web_dir,
                session
        ):
            cleanup_flatpack_dir(flatpack_dir)
            return False, None

        if not check_node_and_run_npm_install(web_dir):
            cleanup_flatpack_dir(flatpack_dir)
            return False, None

        if not setup_nextjs_project(app_dir):
            cleanup_flatpack_dir(flatpack_dir)
            return False, None

        for subdir in ["public", "pages"]:
            dir_path = app_dir / subdir
            dir_path.mkdir(parents=True, exist_ok=True)

        app_public_dir = app_dir / "public"
        app_output_dir = app_public_dir / "output"
        web_output_dir = web_dir / "output"

        if app_output_dir.exists() and not app_output_dir.is_symlink():
            app_output_dir.unlink(missing_ok=True)

        if web_output_dir.exists() and not web_output_dir.is_symlink():
            web_output_dir.unlink(missing_ok=True)

        if not app_output_dir.exists():
            app_output_dir.symlink_to(
                output_dir.resolve(),
                target_is_directory=True
            )

            logger.info(
                "Created symlink: %s -> %s",
                app_output_dir,
                output_dir
            )

        if not web_output_dir.exists():
            web_output_dir.symlink_to(
                output_dir.resolve(),
                target_is_directory=True
            )

            logger.info(
                "Created symlink: %s -> %s",
                web_output_dir,
                output_dir
            )

        return True, flatpack_dir

    except Exception as e:
        logger.error("Failed to set up flatpack directory: %s", e)
        cleanup_flatpack_dir(flatpack_dir)
        return False, None


def unpack_content(
        source: Union[Path, str],
        dest_dir: Path,
        package_manager: PackageManager,
        is_fpk: bool = False
) -> bool:
    """Common unpacking procedure for all sources."""
    try:
        source_path = Path(source).resolve()
        dest_dir = dest_dir.resolve()

        if is_fpk:
            package_manager.unpack(
                str(source_path),
                str(dest_dir)
            )
        else:
            dest_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for item in source_path.iterdir():
                dest_item = dest_dir / item.name

                if not dest_item.exists():
                    if item.is_dir():
                        shutil.copytree(
                            item,
                            dest_item
                        )
                    else:
                        shutil.copy2(
                            item,
                            dest_item
                        )

        return True
    except Exception as e:
        logger.error("Failed to unpack content: %s", e)
        return False


def finalize_setup(flatpack_dir: Path) -> bool:
    """Common post-setup procedure."""
    try:
        pkgdir = sys.modules['flatpack'].__path__[0]
        executor_src = Path(pkgdir) / 'python_executor.py'
        executor_dest = flatpack_dir / 'build' / 'python_executor.py'
        shutil.copy2(executor_src, executor_dest)

        initialize_database_manager(str(flatpack_dir))

        bash_script_path = flatpack_dir / "flatpack.sh"
        toml_path = flatpack_dir / "flatpack.toml"

        if toml_path.exists():
            bash_script_content = parse_toml_to_venv_script(
                str(toml_path),
                env_name=str(flatpack_dir)
            )
            bash_script_path.write_text(bash_script_content)
            safe_script_path = shlex.quote(str(bash_script_path.resolve()))
            subprocess.run(['/bin/bash', safe_script_path], check=True)

        web_dir = flatpack_dir / "web"
        build_dir = flatpack_dir / "build"

        for file in ["app/pages/index.tsx"]:
            source = flatpack_dir / file
            destination = web_dir / file

            if source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)

                if destination.exists():
                    destination.unlink()

                shutil.copy2(source, destination)

        global flatpack_directory
        flatpack_directory = str(flatpack_dir)

        sync_hooks_to_db_on_startup()
        sync_connections_from_file()
        sync_sources_from_file()

        fpk_cache_unbox(str(flatpack_dir))
        return True
    except Exception as e:
        logger.error("Failed during final setup: %s", e)
        return False


def unbox_from_online(directory_name: str, session: httpx.Client) -> bool:
    """Unbox from an online source using standardized procedure."""
    package_manager = PackageManager()
    success, flatpack_dir = setup_flatpack_directory(
        directory_name,
        session
    )

    if not success:
        return False

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            fpk_url = f"{BASE_URL}/{directory_name}/{directory_name}.fpk"
            temp_path = Path(temp_dir) / f"{directory_name}.fpk"

            download_response = session.get(fpk_url)
            download_response.raise_for_status()
            temp_path.write_bytes(download_response.content)

            if not unpack_content(
                    temp_path,
                    flatpack_dir,
                    package_manager,
                    is_fpk=True
            ):
                cleanup_flatpack_dir(flatpack_dir)
                return False

            if not finalize_setup(flatpack_dir):
                cleanup_flatpack_dir(flatpack_dir)
                return False

            return True
    except Exception as e:
        logger.error("Failed during online unboxing: %s", e)
        cleanup_flatpack_dir(flatpack_dir)
        return False


def unbox_from_local_fpk(
        directory_name: str,
        fpk_path: Path,
        session: httpx.Client
) -> bool:
    """Unbox from a local .fpk file using standardized procedure."""
    package_manager = PackageManager()
    success, flatpack_dir = setup_flatpack_directory(directory_name, session)

    if not success:
        return False

    try:
        if not unpack_content(
                fpk_path,
                flatpack_dir,
                package_manager,
                is_fpk=True
        ):
            cleanup_flatpack_dir(flatpack_dir)
            return False

        if not finalize_setup(flatpack_dir):
            cleanup_flatpack_dir(flatpack_dir)
            return False

        return True
    except Exception as e:
        logger.error("Failed to unbox local .fpk: %s", e)
        cleanup_flatpack_dir(flatpack_dir)
        return False


def unbox_from_local_directory(
        directory_name: str,
        session: httpx.Client
) -> bool:
    """Unbox from a local directory using standardized procedure."""
    package_manager = PackageManager()
    success, flatpack_dir = setup_flatpack_directory(
        directory_name,
        session
    )

    if not success:
        return False

    try:
        source_dir = Path.cwd() / directory_name

        if not unpack_content(
                source_dir,
                flatpack_dir,
                package_manager,
                is_fpk=False
        ):
            cleanup_flatpack_dir(flatpack_dir)
            return False

        if not finalize_setup(flatpack_dir):
            cleanup_flatpack_dir(flatpack_dir)
            return False

        return True
    except Exception as e:
        logger.error(
            "Failed to unbox local directory: %s", e
        )
        cleanup_flatpack_dir(flatpack_dir)
        return False


def fpk_unbox(
        directory_name: str,
        session: httpx.Client,
        local: bool = False,
        fpk_path: Union[str, Path, None] = None
) -> bool:
    """Main entry point for unboxing flatpacks."""
    if not directory_name or not isinstance(
            directory_name, str
    ) or not fpk_valid_directory_name(
        directory_name
    ):
        logger.error("Invalid directory name: %s", directory_name)
        return False

    if not Path(CONFIG_FILE_PATH).exists():
        logger.info(
            "Config file not found. Creating initial configuration."
        )
        save_config({})

    if fpk_path is not None:
        fpk_file = Path(fpk_path)

        if not fpk_file.is_file() or fpk_file.suffix != ".fpk":
            logger.error("Invalid .fpk file: %s", fpk_file)
            return False

        success = unbox_from_local_fpk(
            directory_name,
            fpk_file,
            session
        )
    elif local:
        success = unbox_from_local_directory(
            directory_name,
            session
        )
    else:
        success = unbox_from_online(
            directory_name,
            session
        )

    if not success:
        return False

    flatpack_dir = Path.cwd() / directory_name
    build_dir = flatpack_dir / "build"
    web_dir = flatpack_dir / "web"
    bash_script_path = flatpack_dir / "flatpack.sh"
    toml_path = flatpack_dir / "flatpack.toml"

    if not toml_path.exists():
        logger.error(
            "flatpack.toml not found in %s",
            toml_path.parent
        )
        return False

    try:
        bash_script_content = parse_toml_to_venv_script(
            str(toml_path),
            env_name=str(flatpack_dir)
        )
        bash_script_path.write_text(bash_script_content)

        safe_script_path = shlex.quote(
            str(bash_script_path.resolve())
        )

        subprocess.run(
            ['/bin/bash', safe_script_path],
            check=True
        )

        for file in ["app/pages/index.tsx"]:
            source = build_dir / file
            destination = web_dir / file

            if source.exists():
                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                if destination.exists():
                    destination.unlink()

                shutil.copy2(
                    source,
                    destination
                )

        pkgdir = sys.modules['flatpack'].__path__[0]
        executor_src = Path(pkgdir) / 'python_executor.py'
        executor_dest = build_dir / 'python_executor.py'
        shutil.copy2(executor_src, executor_dest)

        flatpack_dir_str = str(flatpack_dir.resolve())
        initialize_database_manager(flatpack_dir_str)

        sync_hooks_to_db_on_startup()
        sync_connections_from_file()
        sync_sources_from_file()

        fpk_cache_unbox(flatpack_dir_str)
        return True

    except Exception as e:
        logger.error("Failed during final initialization steps: %s", e)
        return False


def fpk_valid_directory_name(name: str) -> bool:
    """Validate that the directory name contains only alphanumeric characters, dashes, underscores, and slashes.

    Args:
        name (str): The directory name to validate.

    Returns:
        bool: True if the name is valid, False otherwise.
    """
    pattern = r"^[\w\-/]+$"
    is_valid = re.match(pattern, name) is not None
    return is_valid


def fpk_update(
        flatpack_name: str,
        session: requests.Session,
        branch: str = "main"
):
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
        "build": ["device.sh"],
        "web": ["app.css", "app.js", "index.html", "package.json",
                "robotomono.woff2"],
    }

    binary_extensions = [".sh", ".woff2"]

    flatpack_dir = Path.cwd() / flatpack_name

    if not flatpack_dir.exists() or not flatpack_dir.is_dir():
        console.print(
            f"[bold red]Error:[/bold red] The flatpack '{flatpack_name}' does not exist or is not a directory."
        )
        return

    web_dir = flatpack_dir / "web"
    build_dir = flatpack_dir / "build"

    if not web_dir.exists():
        console.print(
            f"[bold red]Error:[/bold red] No web directory found for flatpack '{flatpack_name}'. Aborting update."
        )
        return

    if not build_dir.exists():
        build_dir.mkdir(parents=True, exist_ok=True)
        console.print(
            f"Created [bold cyan]build[/bold cyan] directory for flatpack '{flatpack_name}'."
        )

    for dir_name, files in files_to_update.items():
        target_dir = web_dir if dir_name == "web" else build_dir

        for file in files:
            file_url = f"{TEMPLATE_REPO_URL}/contents/{file}?ref={branch}"
            local_file_path = target_dir / file
            console.print(f"Updating [bold cyan]{file}[/bold cyan]...")

            try:
                response = session.get(file_url)
                response.raise_for_status()
                file_data = response.json()

                if "content" in file_data:
                    file_content = base64.b64decode(file_data["content"])
                    if any(file.endswith(ext) for ext in binary_extensions):
                        with open(local_file_path, "wb") as local_file:
                            local_file.write(file_content)
                    else:
                        content = file_content.decode("utf-8")

                        with open(
                                local_file_path,
                                "w", encoding="utf-8"
                        ) as local_file:

                            local_file.write(content)

                    if local_file_path.exists():
                        console.print(
                            f"[bold green]Replaced[/bold green] existing {file} in flatpack '{flatpack_name}/{dir_name}'"
                        )
                    else:
                        console.print(
                            f"[bold green]Added[/bold green] new {file} to flatpack '{flatpack_name}/{dir_name}'"
                        )
                else:
                    console.print(
                        f"[bold red]Error:[/bold red] Failed to retrieve content for {file}"
                    )

            except requests.RequestException as e:
                console.print(
                    f"[bold red]Error:[/bold red] Failed to update {file}: {str(e)}"
                )
            except UnicodeDecodeError as e:
                console.print(
                    f"[bold red]Error:[/bold red] Failed to decode content for {file}: {str(e)}"
                )

    console.print(
        f"[bold green]Flatpack '{flatpack_name}' update completed.[/bold green]"
    )


def fpk_verify(directory: Union[str, None]):
    """Verify a flatpack.

    Args:
        directory (Union[str, None]): The directory to use for verification.
            If None, a cached directory will be used if available.

    Returns:
        None
    """
    console.print(
        "[yellow]Flatpack verification functionality is not yet implemented.[/yellow]"
    )

    cache_file_path = HOME_DIR / ".fpk_unbox.cache"
    last_unboxed_flatpack = None

    if directory and fpk_valid_directory_name(directory):
        last_unboxed_flatpack = directory
    elif cache_file_path.exists():
        last_unboxed_flatpack = cache_file_path.read_text().strip()
    else:
        console.print("[red]No valid flatpack directory found.[/red]")
        return

    verification_script_path = Path(
        last_unboxed_flatpack) / "build" / "build.sh"

    if not verification_script_path.exists() or not verification_script_path.is_file():
        console.print(
            f"[red]Verification script not found in {last_unboxed_flatpack}.[/red]"
        )
        return


def setup_arg_parser():
    """
    Set up the argument parser for the Flatpack command line interface.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description="Flatpack command line interface")

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands"
    )

    # General commands
    parser_find = subparsers.add_parser(
        "find", help="Find model files in the current directory"
    )

    parser_find.set_defaults(func=fpk_cli_handle_find)

    parser_help = subparsers.add_parser(
        "help",
        help="Display help for commands"
    )

    parser_help.set_defaults(func=fpk_cli_handle_help)

    parser_list = subparsers.add_parser(
        "list",
        help="List available flatpack directories"
    )

    parser_list.set_defaults(func=fpk_cli_handle_list)

    parser_version = subparsers.add_parser(
        "version",
        help="Display the version of flatpack"
    )

    parser_version.set_defaults(func=fpk_cli_handle_version)

    # API Key management
    parser_api_key = subparsers.add_parser(
        "api-key",
        help="API key management commands"
    )

    api_key_subparsers = parser_api_key.add_subparsers(dest="api_key_command")

    # Get API key
    parser_get_api = api_key_subparsers.add_parser(
        "get",
        help="Get the current API key"
    )

    parser_get_api.set_defaults(func=fpk_cli_handle_get_api_key)

    # Set API key
    parser_set_api = api_key_subparsers.add_parser(
        "set",
        help="Set the API key"
    )

    parser_set_api.add_argument(
        "api_key",
        type=str,
        help="API key to set"
    )

    parser_set_api.set_defaults(func=fpk_cli_handle_set_api_key)

    # Build commands
    parser_build = subparsers.add_parser(
        "build",
        help="Build a flatpack"
    )

    parser_build.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="The directory of the flatpack to build",
    )

    parser_build.add_argument(
        "--use-euxo",
        action="store_true",
        help="Use 'set -euxo pipefail' in the shell script (default is 'set -euo pipefail')",
    )

    parser_build.set_defaults(func=fpk_cli_handle_build)

    # Create flatpack
    parser_create = subparsers.add_parser(
        "create",
        help="Create a new flatpack"
    )

    parser_create.add_argument(
        "input",
        nargs="?",
        default=None,
        help="The name of the flatpack to create"
    )

    parser_create.set_defaults(func=fpk_cli_handle_create)

    # Model compression
    parser_compress = subparsers.add_parser(
        "compress",
        help="Compress a model for deployment"
    )

    parser_compress.add_argument(
        "model_id",
        type=str,
        help="The name of the Hugging Face repository (format: username/repo_name)",
    )

    parser_compress.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token for private repositories",
    )

    parser_compress.add_argument(
        "--method",
        type=str,
        default="llama.cpp",
        choices=["llama.cpp"],
        help="Compression method to use (default: llama.cpp)",
    )

    parser_compress.set_defaults(func=fpk_cli_handle_compress)

    # Run server
    parser_run = subparsers.add_parser(
        "run",
        help="Run the FastAPI server"
    )

    parser_run.add_argument(
        "input",
        nargs="?",
        default=None,
        help="The name of the flatpack to run"
    )

    parser_run.add_argument(
        "--share",
        action="store_true",
        help="Share using ngrok"
    )

    parser_run.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Custom ngrok domain"
    )

    parser_run.set_defaults(func=fpk_cli_handle_run)

    # Unbox commands
    parser_unbox = subparsers.add_parser(
        "unbox",
        help="Unbox a flatpack from a local directory, .fpk file, or warehouse"
    )

    parser_unbox.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to local directory, .fpk file, or flatpack name"
    )

    parser_unbox.add_argument(
        "--warehouse",
        action="store_true",
        help="Fetch flatpack from the online warehouse"
    )

    parser_unbox.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing build directory if it exists"
    )

    parser_unbox.set_defaults(func=fpk_cli_handle_unbox)

    # Update flatpack
    parser_update = subparsers.add_parser(
        "update",
        help="Update a flatpack from the template"
    )

    parser_update.add_argument(
        "flatpack_name",
        help="The name of the flatpack to update"
    )

    parser_update.set_defaults(func=fpk_cli_handle_update)

    # Vector database management
    parser_vector = subparsers.add_parser(
        "vector",
        help="Vector database management"
    )

    vector_subparsers = parser_vector.add_subparsers(dest="vector_command")

    # Add PDF
    parser_add_pdf = vector_subparsers.add_parser(
        "add-pdf",
        help="Add text from a PDF file to the vector database"
    )

    parser_add_pdf.add_argument(
        "pdf_path",
        help="Path to the PDF file to add"
    )

    parser_add_pdf.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory path for storing the vector database and metadata files",
    )

    parser_add_pdf.set_defaults(func=fpk_cli_handle_vector_commands)

    # Add texts
    parser_add_text = vector_subparsers.add_parser(
        "add-texts", help="Add new texts to generate embeddings and store them"
    )

    parser_add_text.add_argument(
        "texts",
        nargs="+",
        help="Texts to add"
    )

    parser_add_text.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory path for storing the vector database and metadata files",
    )

    parser_add_text.set_defaults(func=fpk_cli_handle_vector_commands)

    # Add URL
    parser_add_url = vector_subparsers.add_parser(
        "add-url",
        help="Add text from a URL to the vector database"
    )

    parser_add_url.add_argument(
        "url",
        help="URL to add"
    )

    parser_add_url.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory path for storing the vector database and metadata files",
    )

    parser_add_url.set_defaults(func=fpk_cli_handle_vector_commands)

    # Add Wikipedia
    parser_add_wikipedia_page = vector_subparsers.add_parser(
        "add-wikipedia",
        help="Add text from a Wikipedia page to the vector database"
    )

    parser_add_wikipedia_page.add_argument(
        "page_title",
        help="The title of the Wikipedia page to add"
    )

    parser_add_wikipedia_page.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory path for storing the vector database and metadata files",
    )

    parser_add_wikipedia_page.set_defaults(func=fpk_cli_handle_vector_commands)

    # Search text
    parser_search_text = vector_subparsers.add_parser(
        "search-text",
        help="Search for texts similar to the given query"
    )

    parser_search_text.add_argument(
        "query",
        help="Text query to search for"
    )

    parser_search_text.add_argument(
        "--data-dir",
        type=str,
        default=".",
        help="Directory path for storing the vector database and metadata files",
    )

    parser_search_text.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )

    parser_search_text.add_argument(
        "--recency-weight",
        type=float,
        default=0.5,
        help="Weight for recency in search results (0.0 to 1.0, default: 0.5)",
    )

    parser_search_text.set_defaults(func=fpk_cli_handle_vector_commands)

    # Verify commands
    parser_verify = subparsers.add_parser(
        "verify",
        help="Verify a flatpack"
    )

    parser_verify.add_argument(
        "directory",
        nargs="?",
        default=None,
        help="The directory of the flatpack to verify",
    )

    parser_verify.set_defaults(func=fpk_cli_handle_verify)

    # Packing
    parser_pack = subparsers.add_parser(
        "pack",
        help="Pack a directory into a .fpk package"
    )

    parser_pack.add_argument(
        "input_path",
        help="Path to input directory"
    )

    parser_pack.add_argument(
        "-f", "--force",
        action="store_true",
        help="Overwrite existing .fpk file if it exists"
    )

    parser_pack.set_defaults(func=fpk_cli_handle_pack)

    # Unpack
    parser_unpack = subparsers.add_parser(
        "unpack",
        help="Unpack a .fpk package"
    )

    parser_unpack.add_argument(
        "input_path",
        help="Path to .fpk package"
    )

    parser_unpack.add_argument(
        "-o", "--output-path",
        help="Output directory path (defaults to input path without .fpk extension)",
        required=False
    )

    parser_unpack.set_defaults(func=fpk_cli_handle_unpack)

    # Sign
    parser_sign = subparsers.add_parser(
        "sign",
        help="Sign a .fpk package"
    )

    parser_sign.add_argument(
        "input_path",
        help="Path to .fpk package"
    )

    parser_sign.add_argument(
        "output_path",
        help="Path for signed .fpk package"
    )

    parser_sign.add_argument(
        "private_key_path",
        help="Path to private key file"
    )

    parser_sign.add_argument(
        "--hash-size",
        type=int,
        choices=[256, 384, 512],
        default=256,
        help="Hash size for signing"
    )

    parser_sign.add_argument(
        "--passphrase",
        help="Passphrase for encrypted private key (required if key is encrypted)"
    )

    parser_sign.set_defaults(func=fpk_cli_handle_sign)

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
        logger.error(
            "PDF file does not exist: '%s'",
            pdf_path
        )
        return

    try:
        vm.add_pdf(pdf_path)
        logger.info(
            "Added text from PDF: '%s' to the vector database.",
            pdf_path
        )
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
            logger.info(
                "Added text from URL: '%s' to the vector database.",
                url
            )
        else:
            logger.error(
                "URL is not accessible: '%s'. HTTP Status Code: %d",
                url,
                response.status_code,
            )
    except requests.RequestException as e:
        logger.error("Failed to access URL: '%s'. Error: %s", url, e)


def get_python_processes() -> List[Dict[str, any]]:
    """List Python processes spawned by our program."""
    python_processes = []

    current_pid = os.getpid()
    parent_pid = os.getppid()

    logger.info("Current process: %s, Parent process: %s", current_pid,
                parent_pid)
    logger.info("Current process tree: %s", get_process_tree(current_pid))

    def is_descendant_of_current_process(proc):
        """Check if process is a descendant of our program."""
        try:
            while proc is not None and proc.pid != 1:
                if proc.pid in (current_pid, parent_pid):
                    logger.info(
                        "Found descendant process %s, full tree: %s",
                        proc.pid,
                        get_process_tree(proc.pid),
                    )
                    return True
                proc = proc.parent()
            return False
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if (
                    "python" in proc.info["name"].lower()
                    and proc.info["pid"] != current_pid
                    and is_descendant_of_current_process(proc)
            ):
                cmd = (
                    " ".join(proc.info["cmdline"])
                    if proc.info["cmdline"]
                    else "Unknown"
                )
                python_processes.append(
                    {"pid": proc.info["pid"], "command": cmd, "process": proc}
                )
                logger.info("Adding process %s to termination list",
                            proc.info["pid"])
                logger.info("Process details: %s", cmd)
        except (
                psutil.NoSuchProcess, psutil.AccessDenied,
                psutil.ZombieProcess):
            pass
    return python_processes


def terminate_python_processes(processes: List[Dict[str, any]]) -> None:
    """Safely terminate the given Python processes."""
    for proc_info in processes:
        pid = proc_info["pid"]
        try:
            process = proc_info["process"]
            logger.info("Terminating PID %s...", pid)

            process.terminate()
            try:
                process.wait(timeout=3)
                logger.info("PID %s terminated successfully", pid)
            except psutil.TimeoutExpired:
                logger.info("PID %s didn't respond to SIGTERM, using SIGKILL",
                            pid)
                process.kill()
                logger.info("PID %s killed successfully", pid)
        except psutil.NoSuchProcess:
            logger.info("PID %s no longer exists", pid)
        except Exception as e:
            logger.error("Error terminating PID %s: %s", pid, e)


def fpk_cli_handle_build(args, session):
    """
    Handle the build command for the flatpack CLI with enhanced process management.
    Args:
        args: The command-line arguments.
        session: The HTTP session.
    Returns:
        None
    """
    directory_name = args.directory

    if directory_name is None:
        logger.info(
            "No directory name provided. Using cached directory if available.")

    try:
        asyncio.run(fpk_build(directory_name, use_euxo=args.use_euxo))
    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            error_msg = "Build process was interrupted by user."
        else:
            error_msg = f"An error occurred during the build process: {e}"

        logger.info(error_msg)

        processes = get_python_processes()

        if processes:
            logger.info("Detected running Python processes")

            for proc in processes:
                logger.info("PID: %s - Command: %s", proc["pid"],
                            proc["command"])

            terminate_python_processes(processes)
        else:
            logger.info("No other Python processes found running")

        if not isinstance(e, KeyboardInterrupt):
            sys.exit(1)


def fpk_cli_handle_create(args, session):
    if not args.input:
        console.print(
            "[bold red]Error:[/bold red] No flatpack name specified.")
        return

    flatpack_name = args.input
    if not fpk_valid_directory_name(flatpack_name):
        console.print(
            f"[bold red]Error:[/bold red] Invalid flatpack name: '{flatpack_name}'. Only lowercase letters, numbers, and hyphens are allowed."
        )
        return

    try:
        with console.status(
                f"[bold blue]Creating flatpack '{flatpack_name}'..."):
            fpk_create(flatpack_name)

        console.print(
            f"[bold green]Success:[/bold green] Flatpack '{flatpack_name}' created successfully."
        )
    except ValueError as e:
        console.print(f"[bold yellow]Warning:[/bold yellow] {str(e)}")
    except Exception as e:
        console.print(
            f"[bold red]Error:[/bold red] An unexpected error occurred: {str(e)}"
        )


def is_running_in_docker():
    """Check if code is running inside Docker container."""
    try:
        with open("/proc/1/cgroup", "r") as f:
            for line in f:
                if "docker" in line:
                    return True
        if os.path.exists("/.dockerenv"):
            return True
        return False
    except:
        return False


def download_file_with_progress(
        url: str,
        local_path: Path,
        token: str = None,
        chunk_size: int = 8192
):
    """Download a file with progress tracking using tqdm."""
    headers = {'Authorization': f'Bearer {token}'} if token else {}

    try:
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = chunk_size
        desc = os.path.basename(url)

        with open(local_path, 'wb') as f:
            with tqdm(
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc,
                    dynamic_ncols=True
            ) as pbar:
                try:
                    for chunk in response.iter_content(chunk_size=block_size):
                        size = f.write(chunk)
                        pbar.update(size)
                        if os.path.exists("abort_build"):
                            raise KeyboardInterrupt()
                finally:
                    pbar.close()
    except (requests.exceptions.RequestException, IOError) as e:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Download interrupted by user.[/yellow]")
        raise


def download_model_files(model_id: str, local_dir: Path, token: str = None):
    """Download model files from Hugging Face."""
    api_url = f"https://huggingface.co/api/models/{model_id}/tree/main"
    headers = {'Authorization': f'Bearer {token}'} if token else {}

    try:
        response = requests.get(api_url, headers=headers)
        response.raise_for_status()
        files = response.json()

        tracking_file = local_dir / ".download_tracking"
        downloaded_files = set()

        if tracking_file.exists():
            downloaded_files = set(tracking_file.read_text().splitlines())

        for file in files:
            if file['type'] == 'file':
                rel_path = file['path']
                if rel_path in downloaded_files:
                    continue

                file_url = f"https://huggingface.co/{model_id}/resolve/main/{rel_path}"
                local_path = local_dir / rel_path

                local_path.parent.mkdir(parents=True, exist_ok=True)

                download_file_with_progress(file_url, local_path, token)

                downloaded_files.add(rel_path)
                tracking_file.write_text('\n'.join(downloaded_files))

        return True

    except KeyboardInterrupt:
        return False

    except requests.exceptions.RequestException as e:
        return False


def fpk_cli_handle_compress(args, session: httpx.Client):
    """
    Handle the model compression command for the Flatpack CLI.
    """
    if is_running_in_docker():
        console.print(
            Panel(
                "[bold red]Error:[/bold red] Model compression cannot be run inside Docker containers.\n"
                "Please run compression operations on your host machine.",
                title="Docker Environment Detected",
                border_style="red",
            )
        )
        return

    model_id = args.model_id
    token = args.token
    method = getattr(args, "method", "llama.cpp")

    if not re.match(r"^[\w-]+/[\w.-]+$", model_id):
        logger.error("Invalid Hugging Face repository format specified.")
        console.print(
            "[bold red]ERROR:[/bold red] Please specify a valid Hugging Face repository in the format 'username/repo_name'."
        )
        return

    repo_name = model_id.split("/")[-1]
    local_dir = Path(repo_name)

    console.print(
        Panel.fit(
            "[bold green]Starting model compression process[/bold green]",
            title="Compression Status",
        )
    )

    try:
        if local_dir.exists():
            console.print(
                f"[bold yellow]INFO:[/bold yellow] Existing model directory '{local_dir}' found. Attempting to resume download..."
            )
        else:
            console.print(
                f"[bold blue]INFO:[/bold blue] Creating new directory '{local_dir}' for the model..."
            )
            local_dir.mkdir(parents=True, exist_ok=True)

        success = download_model_files(model_id, local_dir, token)
        if not success:
            if os.path.exists("abort_build"):
                console.print(
                    "\n[bold yellow]WARNING:[/bold yellow] Process interrupted by user. Exiting..."
                )
                return
            console.print(
                f"[bold red]ERROR:[/bold red] Failed to download the model."
            )
            return

        if method == "llama.cpp":
            current_dir = os.path.basename(os.getcwd())

            if current_dir == "llama.cpp":
                console.print(
                    "[bold blue]INFO:[/bold blue] Current directory is already llama.cpp. Using current directory."
                )
                llama_cpp_dir = "."
            else:
                llama_cpp_dir = "llama.cpp"

                if not os.path.exists(llama_cpp_dir):
                    console.print(
                        "[bold blue]INFO:[/bold blue] Setting up llama.cpp..."
                    )
                    git_executable = shutil.which("git")

                    if not git_executable:
                        console.print(
                            "[bold red]ERROR:[/bold red] The 'git' executable was not found in your PATH."
                        )
                        return

                    try:
                        subprocess.run(
                            [
                                git_executable,
                                "clone",
                                "--depth",
                                "1",
                                "https://github.com/ggerganov/llama.cpp",
                                llama_cpp_dir,
                            ],
                            check=True,
                        )

                        console.print(
                            f"[bold green]SUCCESS:[/bold green] Finished cloning llama.cpp repository into '{llama_cpp_dir}'"
                        )
                    except subprocess.CalledProcessError as e:
                        console.print(
                            f"[bold red]ERROR:[/bold red] Failed to clone the llama.cpp repository. Error: {e}"
                        )
                        return
                else:
                    console.print(
                        f"[bold blue]INFO:[/bold blue] llama.cpp directory already exists. Skipping setup."
                    )

            ready_file = os.path.join(llama_cpp_dir, "ready")
            requirements_file = os.path.join(
                llama_cpp_dir,
                "requirements.txt"
            )
            venv_dir = os.path.join(llama_cpp_dir, "venv")
            venv_python = os.path.join(venv_dir, "bin", "python")
            convert_script = os.path.join(llama_cpp_dir,
                                          "convert_hf_to_gguf.py")

            if not os.path.exists(ready_file):
                console.print(
                    "[bold blue]INFO:[/bold blue] Building llama.cpp...")

                try:
                    cmake_executable = shutil.which("cmake")

                    if not cmake_executable:
                        console.print(
                            "[bold red]ERROR:[/bold red] 'cmake' executable not found in PATH."
                        )
                        return

                    subprocess.run(
                        [cmake_executable, "-B", "build"], cwd=llama_cpp_dir,
                        check=True
                    )

                    subprocess.run(
                        [cmake_executable, "--build", "build", "--config",
                         "Release"],
                        cwd=llama_cpp_dir,
                        check=True,
                    )

                    console.print(
                        "[bold green]SUCCESS:[/bold green] Finished building llama.cpp"
                    )

                    if not os.path.exists(venv_dir):
                        console.print(
                            f"[bold blue]INFO:[/bold blue] Creating virtual environment in '{venv_dir}'..."
                        )
                        create_venv(venv_dir)
                    else:
                        console.print(
                            f"[bold blue]INFO:[/bold blue] Virtual environment already exists in '{venv_dir}'"
                        )

                    console.print(
                        "[bold blue]INFO:[/bold blue] Installing llama.cpp dependencies..."
                    )
                    pip_command = [
                        "/bin/bash",
                        "-c",
                        (
                            f"source {shlex.quote(os.path.join(venv_dir, 'bin', 'activate'))} && "
                            f"pip install -r {shlex.quote(requirements_file)}"
                        ),
                    ]
                    subprocess.run(pip_command, check=True)
                    console.print(
                        "[bold green]SUCCESS:[/bold green] Finished installing llama.cpp dependencies"
                    )

                    with open(ready_file, "w") as f:
                        f.write("Ready")

                except subprocess.CalledProcessError as e:
                    console.print(
                        f"[bold red]ERROR:[/bold red] Failed to build llama.cpp. Error: {e}"
                    )
                    return
                except Exception as e:
                    console.print(
                        f"[bold red]ERROR:[/bold red] An error occurred during the setup of llama.cpp. Error: {e}"
                    )
                    return
            else:
                console.print(
                    "[bold blue]INFO:[/bold blue] llama.cpp is already built and ready."
                )

            output_file = os.path.join(local_dir, f"{repo_name}-fp16.bin")
            quantized_output_file = os.path.join(local_dir,
                                                 f"{repo_name}-Q4_K_M.gguf")
            outtype = "f16"

            if not os.path.exists(convert_script):
                console.print(
                    f"[bold red]ERROR:[/bold red] The conversion script '{convert_script}' does not exist."
                )
                return

            if not os.path.exists(output_file):
                console.print(
                    "[bold blue]INFO:[/bold blue] Converting the model...")

                try:
                    venv_activate = os.path.join(venv_dir, "bin", "activate")

                    convert_command = [
                        "/bin/bash",
                        "-c",
                        (
                            f"source {shlex.quote(venv_activate)} && {shlex.quote(venv_python)} "
                            f"{shlex.quote(convert_script)} {shlex.quote(str(local_dir))} --outfile "
                            f"{shlex.quote(str(output_file))} --outtype {shlex.quote(outtype)}"
                        ),
                    ]

                    console.print(
                        Panel(
                            Syntax(
                                " ".join(convert_command),
                                "bash",
                                theme="monokai",
                                line_numbers=True,
                            ),
                            title="Conversion Command",
                            expand=False,
                        )
                    )

                    subprocess.run(convert_command, check=True)
                    console.print(
                        f"[bold green]SUCCESS:[/bold green] Conversion complete. The model has been compressed and saved as '{output_file}'"
                    )
                except subprocess.CalledProcessError as e:
                    console.print(
                        f"[bold red]ERROR:[/bold red] Conversion failed. Error: {e}"
                    )
                    return
                except Exception as e:
                    console.print(
                        f"[bold red]ERROR:[/bold red] An error occurred during the model conversion. Error: {e}"
                    )
                    return
            else:
                console.print(
                    f"[bold blue]INFO:[/bold blue] The model has already been converted and saved as '{output_file}'."
                )

            if os.path.exists(output_file):
                console.print(
                    "[bold blue]INFO:[/bold blue] Quantizing the model...")

                try:
                    quantize_command = [
                        os.path.join(llama_cpp_dir,
                                     "build/bin/llama-quantize"),
                        output_file,
                        quantized_output_file,
                        "Q4_K_M",
                    ]
                    subprocess.run(quantize_command, check=True)
                    console.print(
                        f"[bold green]SUCCESS:[/bold green] Quantization complete. The quantized model has been saved as '{quantized_output_file}'."
                    )

                    console.print(
                        f"[bold blue]INFO:[/bold blue] Deleting the original .bin file '{output_file}'..."
                    )
                    os.remove(output_file)
                    console.print(
                        f"[bold green]SUCCESS:[/bold green] Deleted the original .bin file '{output_file}'."
                    )
                except subprocess.CalledProcessError as e:
                    console.print(
                        f"[bold red]ERROR:[/bold red] Quantization failed. Error: {e}"
                    )
                    return
                except Exception as e:
                    console.print(
                        f"[bold red]ERROR:[/bold red] An error occurred during the quantization process. Error: {e}"
                    )
                    return
            else:
                console.print(
                    f"[bold red]ERROR:[/bold red] The original model file '{output_file}' does not exist."
                )

        else:
            console.print(
                f"[bold red]ERROR:[/bold red] Unsupported compression method: {method}"
            )

    except KeyboardInterrupt:
        console.print(
            "\n[bold yellow]WARNING:[/bold yellow] Process interrupted by user. Exiting..."
        )
    except Exception as e:
        console.print(
            f"[bold red]ERROR:[/bold red] An unexpected error occurred: {e}")


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
        table = Table(title="Warehouse")

        table.add_column(
            "Index",
            justify="right",
            style="cyan",
            no_wrap=True
        )

        table.add_column(
            "Flatpack Name",
            justify="left",
            style="green"
        )

        for index, directory in enumerate(directories.split("\n"), start=1):
            table.add_row(
                str(index),
                directory
            )

        console.print(table)

        logger.info("Flatpacks found: %s", directories)
    else:
        console.print("[red]No flatpacks found.[/red]")
        logger.error("No flatpacks found.")


# Connections
def save_connections_to_file_and_db(mappings: List[SourceHookMapping]) -> None:
    """
    Save the source-hook mappings to both the database and the connections file.

    Args:
        mappings (List[SourceHookMapping]): List of mapping objects containing source and target information
    """
    global flatpack_directory

    if not flatpack_directory:
        logger.error("Flatpack directory is not set")
        raise ValueError("Flatpack directory is not set")

    connections_file = os.path.join(flatpack_directory, CONNECTIONS_FILE)

    os.makedirs(os.path.dirname(connections_file), exist_ok=True)

    connections_data = {
        "connections": [
            {
                "source_id": mapping.sourceId,
                "target_id": mapping.targetId,
                "source_type": mapping.sourceType,
                "target_type": mapping.targetType,
            }
            for mapping in mappings
        ]
    }

    try:
        with open(connections_file, "w") as f:
            json.dump(connections_data, f, indent=4)
        logger.info("Saved connections to file: %s", connections_file)
    except Exception as e:
        logger.error("Failed to save connections to file: %s", e)
        raise

    ensure_database_initialized()

    try:
        if not db_manager.delete_all_source_hook_mappings():
            raise Exception("Failed to clear existing mappings from database")

        for mapping in mappings:
            db_manager.add_source_hook_mapping(
                source_id=mapping.sourceId,
                target_id=mapping.targetId,
                source_type=mapping.sourceType,
                target_type=mapping.targetType,
            )
        logger.info("Saved connections to database")
    except Exception as e:
        logger.error("Failed to save connections to database: %s", e)
        raise


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
                hook.get("hook_name", "unknown"),
                e,
            )


def update_hook_in_file(hook_id, updated_hook):
    hooks = load_hooks_from_file()
    for hook in hooks:
        if hook["hook_id"] == hook_id:
            hook.update(updated_hook.dict())
    save_hooks_to_file(hooks)


def setup_routes(fastapi_app):
    FileResponse = lazy_import(
        "fastapi.responses",
        callable_name="FileResponse"
    )

    Response = lazy_import(
        "fastapi.responses",
        callable_name="Response"
    )

    @fastapi_app.on_event("startup")
    async def startup_event():
        """Handle startup tasks, including initializing server state and scheduling."""
        global SERVER_START_TIME

        SERVER_START_TIME = datetime.now(timezone.utc)

        fastapi_app.state.csrf_token_base = secrets.token_urlsafe(32)

        logger.info(
            "Server started at %s. Cooldown period: %s",
            SERVER_START_TIME,
            COOLDOWN_PERIOD,
        )
        logger.info(
            "CSRF token base generated for this session: %s",
            fastapi_app.state.csrf_token_base,
        )
        asyncio.create_task(run_scheduler())

    @fastapi_app.on_event("shutdown")
    async def shutdown_event():
        """Handle shutdown tasks, including disconnecting ngrok ingress."""
        if hasattr(fastapi_app.state, "ngrok_listener"):
            try:
                ngrok_module = lazy_import("ngrok")
                await asyncio.wait_for(
                    asyncio.to_thread(
                        ngrok_module.disconnect,
                        fastapi_app.state.ngrok_listener.url()
                    ),
                    timeout=5.0,
                )
                logger.info("Disconnected ngrok ingress.")
                console.print(
                    "Disconnected ngrok ingress.",
                    style="bold green"
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Timed out while trying to disconnect ngrok ingress.")
                console.print(
                    "Timed out while trying to disconnect ngrok ingress.",
                    style="bold red",
                )
            except Exception as e:
                logger.error("Failed to disconnect ngrok ingress: %s", e)
                console.print(
                    f"Failed to disconnect ngrok ingress: {e}",
                    style="bold red"
                )
            finally:
                fastapi_app.state.ngrok_listener = None

    @fastapi_app.middleware("http")
    async def csrf_middleware(request: Request, call_next):
        """Enforces CSRF protection on non-GET requests, excluding exempt paths."""
        if request.method != "GET" and not any(
                request.url.path.startswith(path) for path in CSRF_EXEMPT_PATHS
        ):
            await csrf_protect(request)
        return await call_next(request)

    @fastapi_app.get("/csrf-token")
    async def get_csrf_token(request: Request, response: Response):
        """Generate and return a CSRF token, setting it as an HTTP-only cookie."""
        csrf_token = secrets.token_urlsafe(32)
        timestamp = str(int(time.time()))
        token_with_timestamp = f"{timestamp}:{csrf_token}"
        signed_token = request.app.state.signer.sign(
            token_with_timestamp).decode()

        response.set_cookie(
            key="csrf_token",
            value=signed_token,
            httponly=True,
            samesite="strict",
            secure=False,
        )

        return {"csrf_token": csrf_token}

    @fastapi_app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        """Serve the favicon if it exists."""
        favicon_path = Path(flatpack_directory) / "build" / "favicon.ico"
        if favicon_path.exists():
            return FileResponse(favicon_path)
        return Response(status_code=204)

    @fastapi_app.post("/test-csrf", dependencies=[Depends(csrf_protect)])
    async def test_csrf(request: Request):
        """Test the CSRF protection mechanism."""
        return {"message": "CSRF check passed successfully!"}

    @fastapi_app.get("/test-db")
    async def test_db():
        """Test the database connection."""
        try:
            initialize_database_manager(str(flatpack_directory))
            db_manager._execute_query("SELECT 1")

            return {"message": "Database connection successful"}
        except sqlite3.Error as e:
            logger.error("Database connection failed: %s", e)
            return {"message": f"Database connection failed: {e}"}

    @fastapi_app.post("/api/abort-build", dependencies=[Depends(csrf_protect)])
    async def abort_build(
            request: Request,
            token: str = Depends(authenticate_token)
    ) -> JSONResponse:
        global abort_requested, build_in_progress, flatpack_directory

        async with abort_lock:
            try:
                if not flatpack_directory:
                    logger.warning(
                        "Abort request received with no active build directory")
                    return JSONResponse(
                        content={"message": "No active build directory."},
                        status_code=400)

                build_dir = Path(flatpack_directory) / "build"
                abort_file = build_dir / "abort_build"
                abort_file.touch()

                abort_requested = True
                build_in_progress = False

                build_script_pid = None
                build_processes = []
                for proc in psutil.process_iter(
                        ['pid', 'cmdline', 'name', 'ppid']):
                    try:
                        if str(build_dir) in ' '.join(proc.cmdline()):
                            if 'python' in proc.name().lower():
                                build_processes.append(proc)

                                if proc.ppid() == os.getpid():
                                    build_script_pid = proc.pid
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                cleanup_success = True

                if build_script_pid:
                    try:
                        os.write(3, b"__EXIT_PYTHON_EXECUTOR__\n")
                        await asyncio.sleep(0.5)
                    except OSError:
                        pass

                for proc in build_processes:
                    try:
                        if proc.is_running():
                            proc.terminate()
                    except psutil.NoSuchProcess:
                        pass

                temp_files = ["temp.sh", "python_stdin",
                              "python_stdout", "python_input", "abort_build"]

                for temp_file in temp_files:
                    try:
                        temp_path = build_dir / temp_file
                        if temp_path.exists():
                            temp_path.unlink()
                    except Exception as e:
                        logger.warning("Could not remove temp file %s: %s",
                                       temp_file, e)
                        cleanup_success = False

                status_data = BuildStatusData(
                    status="aborted",
                    timestamp=datetime.now().isoformat(),
                    cleanup_success=cleanup_success,
                    initial_status="running",
                    final_status="aborted",
                    memory_released=0,
                    error=None
                )

                status_file = build_dir / "build_status.json"
                os.makedirs(status_file.parent, exist_ok=True)
                with open(status_file, "w") as f:
                    json.dump(status_data.dict(), f)

                return JSONResponse(
                    content={
                        "message": "Build process aborted successfully",
                        "cleanup_success": cleanup_success,
                    },
                    status_code=200
                )

            except Exception as e:
                logger.error("Error in abort_build: %s", str(e), exc_info=True)
                abort_requested = False
                build_in_progress = False

                try:
                    if abort_file.exists():
                        abort_file.unlink()
                except Exception as cleanup_error:
                    logger.error("Emergency cleanup failed: %s", cleanup_error)

                return JSONResponse(
                    content={"message": f"Error during abort: {str(e)}",
                             "error": str(e)},
                    status_code=500
                )

    @fastapi_app.post("/api/build", dependencies=[Depends(csrf_protect)])
    async def build_flatpack(
            request: Request,
            background_tasks: BackgroundTasks,
            token: str = Depends(authenticate_token),
    ):
        """Trigger the build process for the flatpack."""
        global abort_requested

        if not flatpack_directory:
            raise HTTPException(status_code=500,
                                detail="Flatpack directory is not set")

        if build_in_progress:
            return JSONResponse(
                content={"message": "A build is already in progress."},
                status_code=409
            )

        try:
            abort_requested = False
            background_tasks.add_task(run_build_process, schedule_id=None)
            logger.info(
                "Started build process for flatpack located at %s",
                flatpack_directory
            )

            return JSONResponse(
                content={
                    "flatpack": flatpack_directory,
                    "message": "Build process started in background.",
                },
                status_code=200,
            )
        except Exception as e:
            logger.error("Failed to start build process: %s", e)
            return JSONResponse(
                content={
                    "flatpack": flatpack_directory,
                    "message": f"Failed to start build process: {e}",
                },
                status_code=500,
            )

    @fastapi_app.get("/api/build-status", dependencies=[Depends(csrf_protect)])
    async def get_build_status(token: str = Depends(authenticate_token)):
        """Get the current build status."""
        if not flatpack_directory:
            raise HTTPException(status_code=500,
                                detail="Flatpack directory is not set")

        status_file = os.path.join(flatpack_directory, "build",
                                   "build_status.json")

        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                status_data = json.load(f)
            return JSONResponse(content=status_data)
        return JSONResponse(content={"status": "no_builds"})

    @fastapi_app.post("/api/clear-build-status",
                      dependencies=[Depends(csrf_protect)])
    async def clear_build_status(token: str = Depends(authenticate_token)):
        """Clear the current build status."""
        if not flatpack_directory:
            raise HTTPException(status_code=500,
                                detail="Flatpack directory is not set")

        status_file = os.path.join(flatpack_directory, "build",
                                   "build_status.json")

        try:
            if os.path.exists(status_file):
                os.remove(status_file)
            return JSONResponse(
                content={"message": "Build status cleared successfully."}
            )
        except Exception as e:
            logger.error("Error clearing build status: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Error clearing build status: {e}"
            )

    @fastapi_app.post("/api/comments", dependencies=[Depends(csrf_protect)])
    async def add_comment(comment: Comment,
                          token: str = Depends(authenticate_token)):
        """Add a new comment to the database."""
        ensure_database_initialized()
        try:
            db_manager.add_comment(
                comment.block_id, comment.selected_text, comment.comment
            )
            return JSONResponse(
                content={"message": "Comment added successfully."},
                status_code=201
            )
        except Exception as e:
            logger.error("Error adding comment: %s", str(e), exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while adding the comment: {str(e)}",
            )

    @fastapi_app.delete(
        "/api/comments/{comment_id}", dependencies=[Depends(csrf_protect)]
    )
    async def delete_comment(comment_id: int,
                             token: str = Depends(authenticate_token)):
        """Delete a comment from the database by its ID."""
        ensure_database_initialized()
        try:
            if db_manager.delete_comment(comment_id):
                return JSONResponse(
                    content={"message": "Comment deleted successfully."},
                    status_code=200,
                )
            raise HTTPException(status_code=404, detail="Comment not found")
        except Exception as e:
            logger.error("An error occurred while deleting the comment: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while deleting the comment: {e}",
            )

    @fastapi_app.get("/api/comments", dependencies=[Depends(csrf_protect)])
    async def get_all_comments(token: str = Depends(authenticate_token)):
        """Retrieve all comments from the database."""
        ensure_database_initialized()
        try:
            return db_manager.get_all_comments()
        except Exception as e:
            logger.error("An error occurred while retrieving comments: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while retrieving comments: {e}",
            )

    @fastapi_app.get("/api/heartbeat")
    async def heartbeat():
        """Lightweight heartbeat endpoint for maintaining connection."""
        return JSONResponse(content={"timestamp": int(time.time())},
                            status_code=200)

    @fastapi_app.get(
        "/api/hook-mappings/{target_id}", dependencies=[Depends(csrf_protect)]
    )
    async def get_hook_mappings(
            target_id: str, token: str = Depends(authenticate_token)
    ):
        """Get all source mappings for a specific hook."""
        ensure_database_initialized()
        try:
            mappings = db_manager.get_mappings_by_target(target_id)
            return JSONResponse(content={"mappings": mappings})
        except Exception as e:
            logger.error("Error retrieving hook mappings: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while retrieving the mappings: {e}",
            )

    @fastapi_app.post("/api/hooks", dependencies=[Depends(csrf_protect)])
    async def add_hook(hook: Hook, token: str = Depends(authenticate_token)):
        """Add a new hook to the database."""
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

    @fastapi_app.delete("/api/hooks/{hook_id}",
                        dependencies=[Depends(csrf_protect)])
    async def delete_hook(hook_id: int,
                          token: str = Depends(authenticate_token)):
        """Delete a specific hook by its ID."""
        ensure_database_initialized()
        try:
            if db_manager.delete_hook(hook_id):
                hooks = get_all_hooks_from_database()
                save_hooks_to_file(hooks)
                return JSONResponse(
                    content={"message": "Hook deleted successfully."},
                    status_code=200
                )
            raise HTTPException(status_code=404, detail="Hook not found")
        except Exception as e:
            logger.error("An error occurred while deleting the hook: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while deleting the hook: {e}",
            )

    @fastapi_app.get(
        "/api/hooks", response_model=List[Hook],
        dependencies=[Depends(csrf_protect)]
    )
    async def get_hooks(token: str = Depends(authenticate_token)):
        """Retrieve all hooks from the database and synchronize with the hooks file."""
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
            raise HTTPException(status_code=500,
                                detail="Failed to retrieve hooks.")

    @fastapi_app.put("/api/hooks/{hook_id}",
                     dependencies=[Depends(csrf_protect)])
    async def update_hook(
            hook_id: int, hook: Hook, token: str = Depends(authenticate_token)
    ):
        """Update an existing hook by its ID."""
        ensure_database_initialized()
        try:
            success = db_manager.update_hook(
                hook_id,
                hook.hook_name,
                hook.hook_placement,
                hook.hook_script,
                hook.hook_type,
                hook.show_on_frontpage,
            )

            if success:
                hooks = get_all_hooks_from_database()
                save_hooks_to_file(hooks)
                return JSONResponse(
                    content={"message": "Hook updated successfully."},
                    status_code=200
                )

            raise HTTPException(
                status_code=404, detail="Hook not found or update failed."
            )
        except Exception as e:
            logger.error("An error occurred while updating the hook: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while updating the hook: {e}",
            )

    @fastapi_app.get("/api/list-media-files",
                     dependencies=[Depends(csrf_protect)])
    async def list_media_files(token: str = Depends(authenticate_token)):
        """List all media files from the output directory."""
        global flatpack_directory
        if not flatpack_directory:
            raise HTTPException(status_code=500,
                                detail="Flatpack directory is not set")
        output_folder = Path(flatpack_directory) / "web" / "output"
        allowed_extensions = {".gif", ".jpg", ".mp3", ".mp4", ".png", ".wav"}
        try:
            media_files = [
                {"name": f.name, "created_at": f.stat().st_ctime}
                for f in output_folder.iterdir()
                if f.is_file() and f.suffix.lower() in allowed_extensions
            ]
            media_files.sort(key=lambda x: x["created_at"], reverse=True)
            return JSONResponse(content={"files": media_files})
        except Exception as e:
            logger.error("Error listing media files: %s", str(e))
            raise HTTPException(
                status_code=500, detail=f"Error listing media files: {str(e)}"
            )

    @fastapi_app.get("/api/load-file")
    async def load_file(filename: str,
                        token: str = Depends(authenticate_token)):
        """Load a file from the flatpack build directory."""
        if not flatpack_directory:
            raise HTTPException(status_code=500,
                                detail="Flatpack directory is not set")

        sanitized_filename = secure_filename(filename)
        build_dir = Path(flatpack_directory) / "build"
        file_path = build_dir / sanitized_filename

        try:
            file_path = file_path.resolve()
            build_dir = build_dir.resolve()

            if not file_path.is_relative_to(build_dir):
                raise HTTPException(
                    status_code=403,
                    detail="Access to the requested file is forbidden"
                )
        except Exception as e:
            logger.error("Error validating file path: %s", e)
            raise HTTPException(status_code=400, detail="Invalid file path")

        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                code_blocks = json.load(file)

            if not isinstance(code_blocks, list):
                raise ValueError(
                    "Invalid file format: Expected a list of code blocks")

            json_content = json.dumps(code_blocks)
            base64_content = base64.b64encode(
                json_content.encode("utf-8")).decode(
                "utf-8"
            )

            return JSONResponse(content={"content": base64_content})

        except json.JSONDecodeError:
            logger.error("Error decoding JSON from file: %s",
                         sanitized_filename)
            raise HTTPException(status_code=400,
                                detail="Invalid JSON format in file")
        except ValueError as ve:
            logger.error("Error validating file content: %s", str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error("Error reading file '%s': %s", sanitized_filename,
                         str(e))
            raise HTTPException(status_code=500, detail="Error reading file")

    @fastapi_app.get("/api/logs", dependencies=[Depends(csrf_protect)])
    async def get_all_logs(request: Request,
                           token: str = Depends(authenticate_token)):
        """Get a list of all available logs ordered by date."""
        global flatpack_directory

        if flatpack_directory:
            logs_directory = Path(flatpack_directory) / "build" / "logs"
        else:
            cache_file_path = HOME_DIR / ".fpk_unbox.cache"
            if cache_file_path.exists():
                last_unboxed_flatpack = cache_file_path.read_text().strip()
                logs_directory = Path(last_unboxed_flatpack) / "build" / "logs"
            else:
                raise HTTPException(
                    status_code=500,
                    detail="No cached flatpack directory found"
                )

        try:
            if not logs_directory.exists():
                logs_directory.mkdir(parents=True, exist_ok=True)
                if not logs_directory.exists():
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to create logs directory"
                    )

            log_files = sorted(
                [
                    f
                    for f in os.listdir(logs_directory)
                    if f.startswith("build_") and f.endswith(".log")
                ],
                key=lambda x: datetime.strptime(x,
                                                "build_%Y_%m_%d_%H_%M_%S.log"),
                reverse=True,
            )

            return JSONResponse(content={"logs": log_files}, status_code=200)

        except HTTPException:
            raise

        except Exception as e:
            logger.error("Failed to list log files: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Failed to list log files: {e}"
            )

    @fastapi_app.get("/api/logs/{log_filename}",
                     dependencies=[Depends(csrf_protect)])
    async def get_log_file(
            request: Request, log_filename: str,
            token: str = Depends(authenticate_token)
    ):
        """Get the content of a specific log file."""
        global flatpack_directory

        if flatpack_directory:
            logs_directory = Path(flatpack_directory) / "build" / "logs"
        else:
            cache_file_path = HOME_DIR / ".fpk_unbox.cache"
            if cache_file_path.exists():
                last_unboxed_flatpack = cache_file_path.read_text().strip()
                logs_directory = Path(last_unboxed_flatpack) / "build" / "logs"
            else:
                raise HTTPException(
                    status_code=500,
                    detail="No cached flatpack directory found"
                )

        sanitized_log_filename = secure_filename(log_filename)
        log_path = logs_directory / sanitized_log_filename

        if log_path.exists() and log_path.is_file():
            try:
                if not log_path.resolve().is_relative_to(
                        logs_directory.resolve()):
                    raise HTTPException(
                        status_code=403,
                        detail="Access to the requested file is forbidden",
                    )

                with open(log_path, "r") as file:
                    content = file.read()
                return JSONResponse(content={"log": content}, status_code=200)
            except Exception as e:
                logger.error(
                    "Error reading log file '%s': %s", sanitized_log_filename,
                    e
                )
                raise HTTPException(
                    status_code=500, detail=f"Error reading log file: {e}"
                )
        else:
            raise HTTPException(status_code=404, detail="Log file not found")

    @fastapi_app.get("/api/nextjs-url")
    async def get_nextjs_url(request: Request):
        """Get the Next.js URL when ngrok is active."""
        if hasattr(request.app.state, "nextjs_url"):
            return JSONResponse(
                content={"nextjs_url": request.app.state.nextjs_url}
            )
        return JSONResponse(
            content={"error": "Next.js URL not available"},
            status_code=404
        )

    @fastapi_app.post("/api/save-file", dependencies=[Depends(csrf_protect)])
    async def save_file(
            request: Request,
            filename: str = Form(...),
            content: str = Form(...),
            token: str = Depends(authenticate_token),
    ):
        """Save a file to the flatpack build directory."""

        if not flatpack_directory:
            raise HTTPException(status_code=500,
                                detail="Flatpack directory is not set")

        sanitized_filename = secure_filename(filename)
        file_path = os.path.join(flatpack_directory, "build",
                                 sanitized_filename)

        if not os.path.commonpath(
                [flatpack_directory, os.path.realpath(file_path)]
        ).startswith(os.path.realpath(flatpack_directory)):
            raise HTTPException(
                status_code=403,
                detail="Access to the requested file is forbidden"
            )

        try:
            decoded_content = base64.b64decode(content.encode("utf-8")).decode(
                "utf-8")

            code_blocks = json.loads(decoded_content)

            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(code_blocks, file, ensure_ascii=False, indent=4)

            return JSONResponse(
                content={"message": "File saved successfully!"})

        except Exception as e:
            raise HTTPException(status_code=500,
                                detail=f"Failed to save file: {e}")

    @fastapi_app.get("/api/schedule", dependencies=[Depends(csrf_protect)])
    async def get_schedule(token: str = Depends(authenticate_token)):
        """Retrieve all schedules from the database."""
        ensure_database_initialized()
        try:
            schedules = db_manager.get_all_schedules()
            return JSONResponse(content={"schedules": schedules},
                                status_code=200)
        except Exception as e:
            logger.error(
                "An error occurred while retrieving the schedules: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while retrieving the schedules: {e}",
            )

    @fastapi_app.post("/api/schedule", dependencies=[Depends(csrf_protect)])
    async def save_schedule(request: Request,
                            token: str = Depends(authenticate_token)):
        """Save a new schedule to the database."""
        ensure_database_initialized()
        try:
            data = await request.json()
            schedule_type = data.get("type")
            pattern = data.get("pattern")
            datetimes = data.get("datetimes", [])

            if schedule_type == "manual":
                datetimes = [
                    datetime.fromisoformat(dt).astimezone(
                        timezone.utc).isoformat()
                    for dt in datetimes
                ]

            new_schedule_id = db_manager.add_schedule(schedule_type, pattern,
                                                      datetimes)

            return JSONResponse(
                content={
                    "message": "Schedule saved successfully.",
                    "id": new_schedule_id,
                },
                status_code=200,
            )
        except Exception as e:
            logger.error("An error occurred while saving the schedule: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while saving the schedule: {e}",
            )

    @fastapi_app.delete(
        "/api/schedule/{schedule_id}", dependencies=[Depends(csrf_protect)]
    )
    async def delete_schedule_entry(
            schedule_id: int,
            datetime_index: Optional[int] = None,
            token: str = Depends(authenticate_token),
    ):
        """Delete a schedule or a specific datetime entry from the database."""
        ensure_database_initialized()
        try:
            if datetime_index is not None:
                success = db_manager.delete_schedule_datetime(
                    schedule_id, datetime_index
                )
                if success:
                    return JSONResponse(
                        content={
                            "message": "Schedule datetime entry deleted successfully."
                        },
                        status_code=200,
                    )
                raise HTTPException(status_code=404,
                                    detail="Datetime entry not found")
            success = db_manager.delete_schedule(schedule_id)
            if success:
                return JSONResponse(
                    content={
                        "message": "Entire schedule deleted successfully."},
                    status_code=200,
                )
            raise HTTPException(status_code=404, detail="Schedule not found")
        except Exception as e:
            logger.error(
                "An error occurred while deleting the schedule entry: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while deleting the schedule entry: {e}",
            )

    @fastapi_app.post(
        "/api/source-hook-mappings",
        response_model=MappingResults
    )
    async def add_source_hook_mappings(
            mappings: List[SourceHookMapping],
            token: str = Depends(authenticate_token)
    ) -> JSONResponse:
        """Add new source-hook mappings to the database."""
        try:
            logger.info("Raw request data: %s", mappings)
            logger.info("Data type: %s", type(mappings))
            for mapping in mappings:
                logger.info("Mapping data: %s", mapping.dict())

            if not mappings:
                return JSONResponse(content={"mappings": []})

            results = []
            valid_mappings = []

            for mapping in mappings:
                try:
                    if not mapping.sourceId or not mapping.targetId:
                        logger.error(
                            "Invalid mapping data: sourceId=%s, targetId=%s",
                            mapping.sourceId,
                            mapping.targetId,
                        )
                        raise ValueError(
                            "Missing required sourceId or targetId")

                    valid_mappings.append(mapping)
                    results.append(
                        {
                            "source_id": mapping.sourceId,
                            "target_id": mapping.targetId,
                            "source_type": mapping.sourceType,
                            "target_type": mapping.targetType,
                        }
                    )
                except ValueError as e:
                    logger.error("Validation error for mapping: %s", str(e))
                    continue
                except Exception as e:
                    logger.error("Unexpected error processing mapping: %s",
                                 str(e))
                    continue

            if valid_mappings:
                save_connections_to_file_and_db(valid_mappings)

            return JSONResponse(content={"mappings": results})
        except Exception as e:
            logger.error("Error in add_source_hook_mappings: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @fastapi_app.delete(
        "/api/source-hook-mappings/{mapping_id}",
        dependencies=[Depends(csrf_protect)]
    )
    async def delete_source_hook_mapping(
            mapping_id: int, token: str = Depends(authenticate_token)
    ) -> JSONResponse:
        """Delete a specific source-hook mapping by its ID."""
        try:
            ensure_database_initialized()

            mapping = db_manager.get_source_hook_mapping(mapping_id)

            if not mapping:
                raise HTTPException(status_code=404,
                                    detail="Mapping not found")

            success = db_manager.delete_source_hook_mapping(mapping_id)

            if not success:
                raise HTTPException(status_code=500,
                                    detail="Failed to delete mapping")

            remaining_mappings = db_manager.get_all_source_hook_mappings()

            converted_mappings = [
                SourceHookMapping(
                    sourceId=m["source_id"],
                    targetId=m["target_id"],
                    sourceType=m["source_type"],
                    targetType=m["target_type"],
                )
                for m in remaining_mappings
            ]

            save_connections_to_file_and_db(converted_mappings)

            return JSONResponse(
                content={"message": "Mapping deleted successfully"},
                status_code=200
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error deleting source-hook mapping: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @fastapi_app.get("/api/source-hook-mappings")
    async def get_all_source_hook_mappings(
            token: str = Depends(authenticate_token),
    ) -> JSONResponse:
        """Fetch all source-hook mappings from the database."""
        try:
            ensure_database_initialized()
            mappings = db_manager.get_all_source_hook_mappings()
            return JSONResponse(
                content={"mappings": mappings if mappings else []})
        except Exception as e:
            logger.error("Error retrieving source-hook mappings: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @fastapi_app.post("/api/sources", dependencies=[Depends(csrf_protect)])
    async def add_source(source: Source,
                         token: str = Depends(authenticate_token)):
        """Add a new source to the database."""
        ensure_database_initialized()
        try:
            source_id = db_manager.add_source(
                source.source_name, source.source_type, source.source_details
            )
            return JSONResponse(
                content={"message": "Source added successfully.",
                         "id": source_id},
                status_code=201,
            )
        except Exception as e:
            logger.error("Error adding source: %s", str(e))
            raise HTTPException(
                status_code=500, detail=f"Error adding source: {str(e)}"
            )

    @fastapi_app.get("/api/sources", dependencies=[Depends(csrf_protect)])
    async def get_all_sources(token: str = Depends(authenticate_token)):
        """Retrieve all sources from the database."""
        ensure_database_initialized()
        try:
            sources = db_manager.get_all_sources()
            return JSONResponse(content={"sources": sources}, status_code=200)
        except Exception as e:
            logger.error("Error retrieving sources: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Error retrieving sources: {e}"
            )

    @fastapi_app.get("/api/sources/{source_id}",
                     dependencies=[Depends(csrf_protect)])
    async def get_source_by_id(
            source_id: int, token: str = Depends(authenticate_token)
    ):
        """Retrieve a source by its ID."""
        ensure_database_initialized()
        try:
            source = db_manager.get_source_by_id(source_id)
            if source:
                return JSONResponse(content={"source": source},
                                    status_code=200)
            raise HTTPException(status_code=404, detail="Source not found")
        except Exception as e:
            logger.error("Error retrieving source by ID: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Error retrieving source by ID: {e}"
            )

    @fastapi_app.put("/api/sources/{source_id}",
                     dependencies=[Depends(csrf_protect)])
    async def update_source(
            source_id: int, source: SourceUpdate,
            token: str = Depends(authenticate_token)
    ):
        """Update an existing source."""
        ensure_database_initialized()
        try:
            success = db_manager.update_source(
                source_id,
                source_name=source.source_name,
                source_type=source.source_type,
                source_details=source.source_details,
            )
            if success:
                return JSONResponse(
                    content={"message": "Source updated successfully."},
                    status_code=200
                )
            raise HTTPException(status_code=404, detail="Source not found")
        except Exception as e:
            logger.error("Error updating source: %s", str(e))
            raise HTTPException(
                status_code=500, detail=f"Error updating source: {str(e)}"
            )

    @fastapi_app.delete(
        "/api/sources/{source_id}", dependencies=[Depends(csrf_protect)]
    )
    async def delete_source(source_id: int,
                            token: str = Depends(authenticate_token)):
        """Delete a source from the database by its ID."""
        ensure_database_initialized()
        try:
            if db_manager.delete_source(source_id):
                return JSONResponse(
                    content={"message": "Source deleted successfully."},
                    status_code=200
                )
            raise HTTPException(status_code=404, detail="Source not found")
        except Exception as e:
            logger.error("Error deleting source: %s", e)
            raise HTTPException(status_code=500,
                                detail=f"Error deleting source: {e}")

    @fastapi_app.get("/api/user-status")
    async def user_status(auth: str = Depends(authenticate_token)):
        """Check if the user is authenticated."""
        return JSONResponse(content={"authenticated": True}, status_code=200)

    @fastapi_app.post("/api/validate-token")
    async def validate_token(request: Request, api_token: str = Form(...)):
        """Validate the provided API token and create a session."""
        global VALIDATION_ATTEMPTS
        stored_token = get_token()

        if not stored_token:
            return JSONResponse(
                content={"message": "API token is not set."}, status_code=200
            )
        if validate_api_token(api_token):
            session_id = create_session(api_token)
            response = JSONResponse(
                content={"message": "API token is valid.",
                         "session_id": session_id},
                status_code=200,
            )
            response.set_cookie(
                key="session_id", value=session_id, httponly=True,
                samesite="strict"
            )
            return response
        VALIDATION_ATTEMPTS += 1
        if VALIDATION_ATTEMPTS >= MAX_ATTEMPTS:
            shutdown_server()
        return JSONResponse(content={"message": "Invalid API token."},
                            status_code=401)

    @fastapi_app.post("/api/verify", dependencies=[Depends(csrf_protect)])
    async def verify_flatpack(
            request: Request, token: str = Depends(authenticate_token)
    ):
        """Trigger the verification process for the flatpack."""
        if not flatpack_directory:
            raise HTTPException(status_code=500,
                                detail="Flatpack directory is not set")

        try:
            fpk_verify(flatpack_directory)
            return JSONResponse(
                content={
                    "message": "Verification process completed successfully."},
                status_code=200,
            )
        except Exception as e:
            logger.error("Verification process failed: %s", e)
            return JSONResponse(
                content={"message": f"Verification process failed: {e}"},
                status_code=500,
            )

    return fastapi_app


def fpk_cli_handle_run(args, session):
    ngrok = lazy_import("ngrok")

    global abort_requested, build_in_progress, console, flatpack_directory, nextjs_process

    if not args.input:
        console.print(
            "Please specify a flatpack for the run command.", style="bold red"
        )
        return

    app = None
    server = None
    nextjs_process = None

    host = "0.0.0.0"
    fastapi_port = 8000
    nextjs_port = 3000

    console.print(
        Panel(
            create_security_notice(),
            title="[bold yellow]SECURITY NOTICE[/bold yellow]",
            border_style="bold yellow",
            expand=False,
            padding=(1, 1),
        )
    )

    while True:
        acknowledgment = (
            console.input(
                "[bold yellow]Do you agree to proceed? (YES/NO):[/bold yellow] "
            )
            .strip()
            .upper()
        )
        if acknowledgment == "YES":
            break
        if acknowledgment == "NO":
            console.print("")
            console.print("You must agree to proceed. Exiting.",
                          style="bold red")
            return
        else:
            console.print("")
            console.print("Please answer YES or NO.", style="bold yellow")

    directory = Path(args.input).resolve()
    flatpack_directory = directory
    allowed_directory = Path.cwd()

    if not directory.is_dir() or not directory.exists():
        console.print("")
        console.print(
            f"The flatpack '{directory}' does not exist or is not a directory.",
            style="bold red",
        )
        return

    if not directory.is_relative_to(allowed_directory):
        console.print("")
        console.print(
            "The specified directory is not within allowed paths.",
            style="bold red"
        )
        return

    build_dir = directory / "build"
    web_dir = directory / "web"

    if not build_dir.exists() or not web_dir.exists():
        console.print("")
        console.print(
            "The 'build' or 'web' directory is missing in the flatpack.",
            style="bold yellow",
        )
        return

    console.print("")

    if args.share:
        console.print(
            Panel(
                create_warning_message(),
                title="[bold red]WARNING[/bold red]",
                border_style="bold red",
                expand=False,
                padding=(1, 1),
            )
        )

        while True:
            user_response = (
                console.input(
                    "[bold red]Do you accept these risks? (YES/NO):[/bold red] "
                )
                .strip()
                .upper()
            )

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
    console.print(
        "[CSRF] New secret key generated for this session.",
        style="bold blue"
    )

    csrf_token_base = secrets.token_urlsafe(32)
    logger.info("[CSRF] New token generated for this session.")
    console.print(
        "[CSRF] New token generated for this session.",
        style="bold blue"
    )

    console.print("")

    token = generate_secure_token()
    logger.info("API token generated and displayed to user.")
    console.print(f"Generated API token: {token}", style="bold bright_cyan")

    set_token(token)
    console.print("")

    with console.status(
            "[bold green]Initializing FastAPI server...",
            spinner="dots"
    ):
        app = initialize_fastapi_app(secret_key)
        setup_static_directory(app, str(directory))

    if args.share:
        with console.status(
                "[bold green]Establishing ngrok ingress...", spinner="dots"
        ):
            try:
                if args.domain:
                    fastapi_listener = ngrok.connect(
                        f"{host}:{fastapi_port}",
                        authtoken_from_env=True,
                        domain=args.domain,
                    )
                else:
                    fastapi_listener = ngrok.connect(
                        f"{host}:{fastapi_port}", authtoken_from_env=True
                    )

                fastapi_url = fastapi_listener.url()
                app.state.ngrok_listener = fastapi_listener
                app.state.public_url = fastapi_url

                nextjs_domain = f"next-{args.domain}" if args.domain else None

                if nextjs_domain:
                    nextjs_listener = ngrok.connect(
                        f"{host}:{nextjs_port}",
                        authtoken_from_env=True,
                        domain=nextjs_domain,
                    )
                else:
                    nextjs_listener = ngrok.connect(
                        f"{host}:{nextjs_port}", authtoken_from_env=True
                    )

                nextjs_url = nextjs_listener.url()
                app.state.nextjs_listener = nextjs_listener
                app.state.nextjs_url = nextjs_url

                logger.info("FastAPI ingress established at %s", fastapi_url)
                logger.info("Next.js ingress established at %s", nextjs_url)
                console.print(
                    f"FastAPI ingress established at {fastapi_url}",
                    style="bold green"
                )
                console.print(
                    f"Next.js ingress established at {nextjs_url}",
                    style="bold green"
                )
                console.print("")

            except ValueError as e:
                error_message = str(e)
                if "ERR_NGROK_319" in error_message:
                    console.print(
                        "[bold red]Error: Custom domain not reserved[/bold red]"
                    )
                    console.print(
                        "You must reserve the custom domain in your ngrok dashboard before using it."
                    )
                    console.print(
                        "Please visit: https://dashboard.ngrok.com/domains/new"
                    )
                    console.print(
                        "After reserving the domain, try running the command again."
                    )
                else:
                    console.print(
                        f"[bold red]Error establishing ngrok ingress: {error_message}[/bold red]"
                    )
                return

    uvicorn = lazy_import("uvicorn")

    if not uvicorn:
        console.print(
            "Failed to load uvicorn. Please check your installation.",
            style="bold red"
        )
        return

    config = uvicorn.Config(
        app,
        host=host,
        http="auto",
        lifespan="on",
        loop="asyncio",
        port=fastapi_port,
        timeout_graceful_shutdown=30,
        timeout_keep_alive=0,
    )

    server = uvicorn.Server(config)
    background_tasks = BackgroundTasks()
    scheduler_task = background_tasks.add_task(run_scheduler)

    app_dir = web_dir / "app"

    if app_dir.exists():
        try:
            npm_path = get_executable_path("npm")

            if not npm_path:
                logger.error("npm executable not found in PATH")
                return None

            nextjs_process = subprocess.Popen(
                [npm_path, "run", "dev"],
                bufsize=1,
                cwd=str(app_dir),
                stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE,
                text=True,
                universal_newlines=True,
            )

            def log_process_output(process, name):
                while True:
                    try:
                        line = process.stdout.readline()

                        if not line and process.poll() is not None:
                            break

                        if line:
                            console.print(f"[{name}] {line.strip()}")
                    except Exception:
                        break

            threading.Thread(
                target=log_process_output,
                args=(nextjs_process, "Next.js"),
                daemon=True
            ).start()

            console.print(
                "[bold green]Started Next.js development server[/bold green]"
            )
            console.print("")

            build_output = build_dir / "output"

            if not build_output.exists():
                console.print(
                    f"[bold red]Error: Source directory {build_output} does not exist[/bold red]"
                )
                return

        except Exception as e:
            console.print(
                f"[bold red]Failed to start Next.js server: {e}[/bold red]"
            )
            return

    async def cleanup():
        try:
            if hasattr(app.state, "ngrok_listener"):
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            ngrok.disconnect, app.state.ngrok_listener.url()
                        ),
                        timeout=5.0,
                    )
                except:
                    pass

            if nextjs_process:
                try:
                    nextjs_process.terminate()
                    await asyncio.sleep(1)
                    if nextjs_process.poll() is None:
                        nextjs_process.kill()
                except:
                    pass

            if server:
                server.should_exit = True

        except Exception as e:
            logger.error("Error during cleanup: %s", e)

        try:
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except:
                    pass

            gone, alive = psutil.wait_procs(children, timeout=3)
            for process in alive:
                try:
                    process.kill()
                except:
                    pass

        except Exception:
            pass

        finally:
            if nextjs_process and nextjs_process.poll() is None:
                os._exit(1)

    def signal_handler(sig, frame):
        if nextjs_process:
            try:
                nextjs_process.terminate()
                time.sleep(1)
                if nextjs_process.poll() is None:
                    nextjs_process.kill()
            except:
                pass

        if hasattr(app.state, "ngrok_listener"):
            try:
                ngrok.disconnect(app.state.ngrok_listener.url())
            except:
                pass

        if server:
            server.should_exit = True

        try:
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except:
                    pass
            psutil.wait_procs(children, timeout=3)
        except:
            pass

        os._exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        server.run()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
    except Exception as e:
        logger.error("Server error: %s", e)
        console.print(f"Server error: {e}", style="bold red")
        signal_handler(signal.SIGTERM, None)

    return 0


def fpk_cli_handle_set_api_key(args, session):
    """Handle the 'set' command to set the API key."""
    logger.info("Setting API key: %s", args.api_key)

    api_key = args.api_key
    config = load_config()
    config["api_key"] = api_key
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
    if args.input is None:
        console.print(
            "Please specify a flatpack for the unbox command.",
            style="bold red"
        )
        return

    overwrite = getattr(args, 'overwrite', False)
    directory_name = args.input
    input_path = Path(args.input)

    if input_path.suffix == ".fpk":
        if not input_path.exists() or not input_path.is_file():
            console.print(
                f"The .fpk file '{input_path}' does not exist.",
                style="bold red"
            )
            return

        directory_name = input_path.stem
        fpk_file = input_path
        local = True
        warehouse = False

    elif args.warehouse:
        existing_dirs = fpk_fetch_github_dirs(session)

        if directory_name not in existing_dirs:
            console.print(
                f"The flatpack '{directory_name}' does not exist in the warehouse.",
                style="bold red"
            )
            return

        fpk_file = None
        local = False
        warehouse = True

    else:
        if not fpk_valid_directory_name(directory_name):
            console.print(
                f"Invalid directory name: '{directory_name}'.",
                style="bold red"
            )
            return

        if not input_path.exists() or not input_path.is_dir():
            console.print(
                f"The directory '{directory_name}' does not exist.",
                style="bold red"
            )
            return

        fpk_file = None
        local = True
        warehouse = False

    target_dir = Path(directory_name)

    if target_dir.exists():
        build_dir = target_dir / "build"
        web_dir = target_dir / "web"
        dirs_exist = []

        if build_dir.exists():
            dirs_exist.append("build")
        if web_dir.exists():
            dirs_exist.append("web")

        if dirs_exist:
            if not overwrite:
                console.print(
                    f"Cannot unbox: '{directory_name}' already contains {' and '.join(dirs_exist)} director{'y' if len(dirs_exist) == 1 else 'ies'}.",
                    style="bold red"
                )
                console.print(
                    "Use --overwrite to force unboxing and replace existing contents.",
                    style="bold yellow"
                )
                return
            else:
                try:
                    if build_dir.exists():
                        shutil.rmtree(build_dir)
                    if web_dir.exists():
                        shutil.rmtree(web_dir)
                except Exception as e:
                    console.print(
                        f"Error removing existing directories: {e}",
                        style="bold red"
                    )
                    return

    if not overwrite:
        fpk_display_disclaimer(directory_name, local=local)

        while True:
            user_response = input().strip().upper()

            if user_response == "YES":
                console.print("")
                break

            if user_response == "NO":
                console.print(
                    "Installation aborted by user.",
                    style="bold yellow"
                )
                return

            console.print(
                "Invalid input. Please type 'YES' to accept or 'NO' to decline.",
                style="bold red"
            )

    console.print(
        f"Starting to unbox flatpack '{directory_name}'...",
        style="bold blue"
    )

    try:
        unbox_result = fpk_unbox(
            directory_name,
            session,
            local=local,
            fpk_path=fpk_file
        )

        if unbox_result:
            console.print(
                f"Unboxed flatpack '{directory_name}' successfully.",
                style="bold green"
            )
        else:
            console.print(
                f"Unboxing of flatpack '{directory_name}' failed or was aborted.",
                style="bold yellow"
            )
    except Exception as e:
        console.print(
            f"Failed to unbox flatpack '{directory_name}': {e}",
            style="bold red"
        )


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
    if args.vector_command == "add-texts":
        vm.add_texts(args.texts, "manual")
        console.print(
            f"[bold green]Added {len(args.texts)} texts to the database.[/bold green]"
        )

    elif args.vector_command == "search-text":
        recency_weight = getattr(args, "recency_weight", 0.5)
        results = vm.search_vectors(args.query, recency_weight=recency_weight)

        if hasattr(args, "json") and args.json:
            json_results = [{"id": r["id"], "text": r["text"]} for r in
                            results]
            print(json.dumps(json_results))
        else:
            if results:
                console.print("[bold cyan]Search results:[/bold cyan]")

                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("ID", style="dim", width=12)
                table.add_column("Text")

                for result in results:
                    table.add_row(str(result["id"]), result["text"])

                console.print(table)
            else:
                console.print("[yellow]No results found.[/yellow]")

    elif args.vector_command == "add-pdf":
        console.print(f"[bold blue]Adding PDF: {args.pdf_path}[/bold blue]")
        fpk_cli_handle_add_pdf(args.pdf_path, vm)

    elif args.vector_command == "add-url":
        console.print(f"[bold blue]Adding URL: {args.url}[/bold blue]")
        fpk_cli_handle_add_url(args.url, vm)

    elif args.vector_command == "add-wikipedia":
        vm.add_wikipedia_page(args.page_title)
        console.print(
            f"[bold green]Added text from Wikipedia page: '{args.page_title}' to the vector database.[/bold green]"
        )

    else:
        console.print(
            f"[bold red]Unknown vector command: {args.vector_command}[/bold red]"
        )

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
        logger.info("Verification successful for directory: %s",
                    directory_name)
    except Exception as e:
        logger.error("Verification failed for directory '%s': %s",
                     directory_name, e)


def fpk_cli_handle_version(args, session):
    """Handle the 'version' command to display the version of flatpack.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    print(VERSION)


def fpk_cli_handle_pack(args, session):
    try:
        package_manager = PackageManager()
        package_manager.pack(args.input_path, overwrite=args.force)

        print(
            f"Successfully compressed {args.input_path} to {args.input_path}.fpk"
        )
    except FileExistsError:
        print(
            f"The package '{args.input_path}.fpk' already exists. Use --force to overwrite."
        )
    except Exception as e:
        print(f"Error: {str(e)}")


def fpk_cli_handle_unpack(args, session):
    """Handle the unpack command."""
    try:
        package_manager = PackageManager()

        output_path = getattr(args, 'output_path', None)
        package_manager.unpack(args.input_path, output_path)

        final_output_path = output_path if output_path else \
            os.path.splitext(args.input_path)[0]

        console.print(
            f"[green]Successfully decompressed {args.input_path} to {final_output_path}[/green]"
        )
    except Exception as e:
        console.print(f"[red]Error decompressing file: {e}[/red]")
        raise


def fpk_cli_handle_sign(args, session):
    try:
        package_manager = PackageManager()

        if not args.input_path or not args.output_path or not args.private_key_path:
            console.print("[red]Error: Missing required arguments.[/red]")
            return

        package_manager.sign(
            args.input_path,
            args.output_path,
            args.private_key_path,
            hash_size=args.hash_size,
            passphrase=args.passphrase
        )

        console.print(
            f"[green]Successfully signed {args.input_path} to {args.output_path}[/green]"
        )
    except Exception as e:
        console.print(f"[red]Error signing file: {e}[/red]")


@safe_exit
def main():
    setup_exception_handling()
    setup_signal_handling()
    set_file_limits()

    with SessionManager() as session:
        parser = setup_arg_parser()
        args = parser.parse_args()

        vm = None

        if args.command == "vector":
            vm = fpk_initialize_vector_manager(args)

        try:
            if hasattr(args, "func"):
                if args.command == "vector" and "vector_command" in args:
                    args.func(args, session, vm)
                else:
                    args.func(args, session)
            else:
                parser.print_help()
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
