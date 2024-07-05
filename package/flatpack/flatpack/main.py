import argparse
import ast
import atexit
import logging
import os
import re
import secrets
import shlex
import shutil
import signal
import stat
import string
import subprocess
import sys

from datetime import datetime
from importlib.metadata import version
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional, Union
from zipfile import ZipFile

import httpx
import requests
import toml
import uvicorn

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download

from .agent_manager import AgentManager
from .parsers import parse_toml_to_venv_script
from .session_manager import SessionManager
from .vector_manager import VectorManager

HOME_DIR = Path.home() / ".fpk"
HOME_DIR.mkdir(exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/romlingroup/flatpack/main/warehouse"
CONFIG_FILE_PATH = HOME_DIR / ".fpk_config.toml"
GITHUB_REPO_URL = "https://api.github.com/repos/romlingroup/flatpack"
KEY_FILE_PATH = HOME_DIR / ".fpk_encryption_key"
VERSION = version("flatpack")

MAX_ATTEMPTS = 5
VALIDATION_ATTEMPTS = 0


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
global_log_file_path = HOME_DIR / "fpk_local_only.log"
logger = setup_logging(global_log_file_path)
os.chmod(global_log_file_path, 0o600)

uvicorn_server = None


def handle_termination_signal(signal_number, frame):
    """Handle termination signals for graceful shutdown."""
    logger.info("Received termination signal (%s), shutting down...", signal_number)
    sys.exit(0)


signal.signal(signal.SIGINT, handle_termination_signal)
signal.signal(signal.SIGTERM, handle_termination_signal)


def create_temp_sh(custom_sh_path: Path, temp_sh_path: Path):
    # print(f"[INFO] custom_sh_path: {custom_sh_path}.")
    # print(f"[INFO] temp_sh_path: {temp_sh_path}.")

    try:
        with custom_sh_path.open('r') as infile:
            script = infile.read()

        last_count = script.count('part_bash """') + script.count('part_python """')

        parts = []
        lines = script.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('part_bash """') or line.startswith('part_python """'):
                start_line = i
                i += 1
                while i < len(lines) and not lines[i].strip().endswith('"""'):
                    i += 1
                end_line = i
                if i < len(lines):
                    end_line += 1
                parts.append('\n'.join(lines[start_line:end_line]).strip())
            i += 1

        # print(f"Extracted parts:\n{parts}")

        temp_sh_path.parent.mkdir(parents=True, exist_ok=True)

        context_python_script = Path("/tmp/context_python_script.py")
        exec_python_script = Path("/tmp/exec_python_script.py")

        context_python_script.write_text('')
        exec_python_script.write_text('')

        with temp_sh_path.open('w') as outfile:
            outfile.write("#!/bin/bash\n")
            outfile.write("set -euo pipefail\n")

            outfile.write(f"CONTEXT_PYTHON_SCRIPT=\"{context_python_script}\"\n")
            outfile.write("EVAL_BUILD=\"$SCRIPT_DIR/eval_build.json\"\n")
            outfile.write(f"EXEC_PYTHON_SCRIPT=\"{exec_python_script}\"\n")
            outfile.write("CURR=0\n")

            outfile.write("trap 'rm -f \"$CONTEXT_PYTHON_SCRIPT\" \"$EXEC_PYTHON_SCRIPT\"' EXIT\n")
            outfile.write("rm -f \"$CONTEXT_PYTHON_SCRIPT\" \"$EXEC_PYTHON_SCRIPT\"\n")
            outfile.write("touch \"$CONTEXT_PYTHON_SCRIPT\" \"$EXEC_PYTHON_SCRIPT\"\n")

            outfile.write("datetime=$(date -u +\"%Y-%m-%d %H:%M:%S\")\n")

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

                language = 'bash' if 'part_bash' in header else 'python' if 'part_python' in header else None
                code = '\n'.join(code_lines).strip().replace('\\"', '"')

                # print(f"Header: {header}")
                # print(f"Language: {language}")
                # print(f"Code:\n{code}")

                if language == 'bash':

                    outfile.write(f"{code}\n")
                    outfile.write("((CURR++))\n")

                elif language == 'python':

                    context_code = "\n".join(
                        line for line in code_lines if not line.strip().startswith(('print(', 'subprocess.run')))
                    execution_code = "\n".join(
                        line for line in code_lines if line.strip().startswith(('print(', 'subprocess.run')))

                    if context_code:
                        outfile.write(f"echo \"{context_code}\" >> \"$CONTEXT_PYTHON_SCRIPT\"\n")

                    outfile.write("echo \"try:\" > \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("sed 's/^/    /' \"$CONTEXT_PYTHON_SCRIPT\" >> \"$EXEC_PYTHON_SCRIPT\"\n")

                    if execution_code:
                        outfile.write(f"echo \"{execution_code}\" | sed 's/^/    /' >> \"$EXEC_PYTHON_SCRIPT\"\n")

                    outfile.write("echo \"except Exception as e:\" >> \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("echo \"    print(e)\" >> \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("echo \"    import sys; sys.exit(1)\" >> \"$EXEC_PYTHON_SCRIPT\"\n")
                    outfile.write("$VENV_PYTHON \"$EXEC_PYTHON_SCRIPT\"\n")

                    outfile.write("((CURR++))\n")

                else:
                    print(f"Skipping part with unsupported language: {language}")
                    continue

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

        print(f"[INFO] Temp script generated successfully at {temp_sh_path}")
        logger.info("Temp script generated successfully at %s", temp_sh_path)

    except Exception as e:
        print(f"[ERROR] An error occurred while creating temp script: {e}")
        logger.error("An error occurred while creating temp script: %s", e)


def fpk_build(directory: Union[str, None]):
    """Build a flatpack.

    Args:
        directory (Union[str, None]): The directory to use for building the flatpack. If None, a cached directory will be used if available.

    Returns:
        None
    """
    cache_file_path = HOME_DIR / ".fpk_unbox.cache"

    print(f"[INFO] Looking for cached flatpack in {cache_file_path}.")
    logger.info("Looking for cached flatpack in %s", cache_file_path)

    last_unboxed_flatpack = None

    if directory and fpk_valid_directory_name(directory):
        print(f"[INFO] Using provided directory: {directory}")
        logger.info("Using provided directory: %s", directory)
        last_unboxed_flatpack = directory
    elif cache_file_path.exists():
        print(f"[INFO] Found cached flatpack in {cache_file_path}.")
        logger.info("Found cached flatpack in %s", cache_file_path)
        last_unboxed_flatpack = cache_file_path.read_text().strip()
    else:
        print("[ERROR] No cached flatpack found, and no valid directory provided.")
        logger.error("No cached flatpack found, and no valid directory provided.")
        return

    if not last_unboxed_flatpack:
        print("[ERROR] No valid flatpack directory found.")
        logger.error("No valid flatpack directory found.")
        return

    custom_sh_path = Path(last_unboxed_flatpack) / 'build' / 'custom.sh'
    temp_sh_path = Path(last_unboxed_flatpack) / 'build' / 'temp.sh'

    if not custom_sh_path.exists() or not custom_sh_path.is_file():
        print(f"[ERROR] custom.sh not found in {last_unboxed_flatpack}. Build process canceled.")
        logger.error("custom.sh not found in %s. Build process canceled.", last_unboxed_flatpack)
        return
    else:
        create_temp_sh(custom_sh_path, temp_sh_path)

    building_script_path = Path(last_unboxed_flatpack) / 'build' / 'build.sh'

    log_dir = Path(last_unboxed_flatpack) / 'build' / 'logs'
    log_file_time = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
    log_filename = f"build_{log_file_time}.log"

    build_log_file_path = log_dir / log_filename
    build_log_file_path.parent.mkdir(parents=True, exist_ok=True)

    if not building_script_path.exists() or not building_script_path.is_file():
        print(f"[ERROR] Building script not found in {last_unboxed_flatpack}.")
        logger.error("Building script not found in %s", last_unboxed_flatpack)
        return

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
                print(line, end='')
                logger.info(line.strip())
                log_file.write(line)

            process.wait()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] An error occurred while executing the build script: {e}")
        logger.error("An error occurred while executing the build script: %s", e)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        logger.error("An unexpected error occurred: %s", e)


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
        print(f"[INFO] Cached the directory name '{directory_name}' to {cache_file_path}.")

        logger.info(
            "Cached the directory name '%s' to %s",
            directory_name,
            cache_file_path
        )

    except IOError as e:
        print(f"[ERROR] Failed to cache the directory name '{directory_name}': {e}")
        logger.error("Failed to cache the directory name '%s': %s", directory_name, e)


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
        print(f"[ERROR] {message}")
        logger.error("%s", message)
        sys.exit(1)
    else:
        message = "NGROK_AUTHTOKEN is set."
        print(f"[INFO] {message}")
        logger.info("%s", message)


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
        print(f"[ERROR] Invalid color '{color}' provided. Returning the original text.")
        logger.error("Invalid color '%s' provided. Returning the original text.", color)
        return text

    return colors[color] + text + colors["default"]


def fpk_create(flatpack_name, repo_url="https://github.com/RomlinGroup/template"):
    """Create a new flatpack from a template repository.

    Args:
        flatpack_name (str): The name of the flatpack to create.
        repo_url (str, optional): The URL of the template repository. Defaults to "https://github.com/RomlinGroup/template".

    Raises:
        ValueError: If the flatpack name format is invalid.
    """
    if not re.match(r'^[a-z0-9-]+$', flatpack_name):
        error_message = "Invalid name format. Only lowercase letters, numbers, and hyphens are allowed."
        print(f"[ERROR] {error_message}")
        logger.error("%s", error_message)
        raise ValueError(error_message)

    flatpack_name = flatpack_name.lower().replace(' ', '-')
    current_dir = os.getcwd()

    try:
        template_dir = fpk_download_and_extract_template(repo_url, current_dir)
    except Exception as e:
        error_message = f"Failed to download and extract template: {e}"
        print(f"[ERROR] {error_message}")
        logger.error("Failed to download and extract template: %s", e)
        return

    flatpack_dir = os.path.join(current_dir, flatpack_name)

    try:
        os.makedirs(flatpack_dir, exist_ok=True)
        print(f"[INFO] Created flatpack directory: {flatpack_dir}")
        logger.info("Created flatpack directory: %s", flatpack_dir)
    except OSError as e:
        error_message = f"Failed to create flatpack directory: {e}"
        print(f"[ERROR] {error_message}")
        logger.error("Failed to create flatpack directory: %s", e)
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
        print(f"[INFO] Copied template files to flatpack directory: {flatpack_dir}")
        logger.info("Copied template files to flatpack directory: %s", flatpack_dir)
    except OSError as e:
        error_message = f"Failed to copy template files: {e}"
        print(f"[ERROR] {error_message}")
        logger.error("Failed to copy template files: %s", e)
        return

    files_to_edit = [
        (os.path.join(flatpack_dir, "README.md"), r"# template", f"# {flatpack_name}"),
        (os.path.join(flatpack_dir, "flatpack.toml"), r"{{model_name}}", flatpack_name),
        (os.path.join(flatpack_dir, "build.sh"), r"export DEFAULT_REPO_NAME=template",
         f"export DEFAULT_REPO_NAME={flatpack_name}"),
        (os.path.join(flatpack_dir, "build.sh"), r"export FLATPACK_NAME=template",
         f"export FLATPACK_NAME={flatpack_name}")
    ]

    try:
        for file_path, pattern, replacement in files_to_edit:
            with open(file_path, 'r') as file:
                filedata = file.read()
            newdata = re.sub(pattern, replacement, filedata)
            with open(file_path, 'w') as file:
                file.write(newdata)
        print(f"[INFO] Edited template files for flatpack: {flatpack_name}")
        logger.info("Edited template files for flatpack: %s", flatpack_name)
    except OSError as e:
        error_message = f"Failed to edit template files: {e}"
        print(f"[ERROR] {error_message}")
        logger.error("Failed to edit template files: %s", e)
        return

    try:
        shutil.rmtree(template_dir)
        print(f"[INFO] Removed temporary template directory: {template_dir}")
        logger.info("Removed temporary template directory: %s", template_dir)
    except OSError as e:
        error_message = f"Failed to remove temporary template directory: {e}"
        print(f"[ERROR] {error_message}")
        logger.error("Failed to remove temporary template directory: %s", e)
        return

    print(f"[INFO] Successfully created {flatpack_name}.")
    logger.info("Successfully created %s", flatpack_name)


def fpk_display_disclaimer(directory_name: str, local: bool):
    """Display a disclaimer message with details about a specific flatpack.

    Args:
        directory_name (str): Name of the flatpack directory.
        local (bool): Indicates if the flatpack is local.
    """
    disclaimer_template = """
-----------------------------------------------------
STOP AND READ BEFORE YOU PROCEED ✋
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

    disclaimer_message = disclaimer_template.format(please_note=please_note_colored)
    print(disclaimer_message)
    logger.info(
        "Displayed disclaimer for flatpack '%s' with local set to %s.",
        directory_name,
        local
    )


def fpk_download_and_extract_template(repo_url, dest_dir):
    """Download and extract a template repository.

    Args:
        repo_url (str): The URL of the template repository.
        dest_dir (str): The destination directory to extract the template into.

    Returns:
        str: The path to the extracted template directory.

    Raises:
        RuntimeError: If downloading or extracting the template fails.
    """
    template_dir = os.path.join(dest_dir, "template-main")
    try:
        response = requests.get(f"{repo_url}/archive/refs/heads/main.zip")
        response.raise_for_status()
        with ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(dest_dir)

        print(f"[INFO] Downloaded and extracted template from {repo_url} to {dest_dir}")
        logger.info(
            "Downloaded and extracted template from %s to %s",
            repo_url,
            dest_dir
        )

        return template_dir
    except requests.RequestException as e:
        error_message = f"Failed to download template from {repo_url}: {e}"

        print(f"[ERROR] {error_message}")
        logger.error("%s", error_message)

        raise RuntimeError(error_message)
    except (OSError, IOError) as e:
        error_message = f"Failed to extract template to {dest_dir}: {e}"

        print(f"[ERROR] {error_message}")
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

        print(f"[ERROR] {message}")
        logger.error("%s", message)

        return None

    toml_url = f"{BASE_URL}/{directory_name}/flatpack.toml"
    try:
        response = session.get(toml_url)
        response.raise_for_status()

        print(f"[INFO] Successfully fetched TOML from {toml_url}")
        logger.info("Successfully fetched TOML from %s", toml_url)

        return response.text
    except httpx.HTTPStatusError as e:
        message = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"

        print(f"[ERROR] {message}")
        logger.error("%s", message)

        return None
    except httpx.RequestError as e:
        message = f"Network error occurred: {e}"

        print(f"[ERROR] {message}")
        logger.error("%s", message)

        return None
    except Exception as e:
        message = f"An unexpected error occurred: {e}"

        print(f"[ERROR] {message}")
        logger.error("%s", message)

        return None


def fpk_fetch_github_dirs(session: httpx.Client) -> List[str]:
    """Fetch a list of directory names from the GitHub repository.

    Args:
        session (httpx.Client): HTTP client session for making requests.

    Returns:
        List[str]: List of directory names.
    """
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
            logger.info("Fetched directory names from GitHub: %s", directories)
            return sorted(directories)

        message = f"Unexpected response format from GitHub: {json_data}"

        print(f"[ERROR] {message}")
        logger.error("%s", message)

        return []
    except httpx.HTTPError as e:
        message = f"Unable to connect to GitHub: {e}"

        print(f"[ERROR] {message}")
        logger.error("%s", message)

        sys.exit(1)
    except (ValueError, KeyError) as e:
        message = f"Error processing the response from GitHub: {e}"

        print(f"[ERROR] {message}")
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

        print(f"[INFO] Found {len(model_files)} model file(s).")
        logger.info("Total number of model files found: %d", len(model_files))

    except Exception as e:
        error_message = f"An error occurred while searching for model files: {e}"

        print(f"[ERROR] {error_message}")
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
            print("[INFO] API key retrieved successfully.")
            logger.info("API key retrieved successfully.")
        else:
            print("[INFO] API key not found in the configuration.")
            logger.info("API key not found in the configuration.")
        return api_key
    except Exception as e:
        error_message = f"An error occurred while retrieving the API key: {e}"
        print(f"[ERROR] {error_message}")
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
                print(
                    "[INFO] Last unboxed flatpack directory retrieved: %s",
                    last_flatpack
                )
                logger.info(
                    "Last unboxed flatpack directory retrieved: %s",
                    last_flatpack
                )
                return last_flatpack
        else:
            print("[WARNING] Cache file does not exist: %s", cache_file_path)
            logger.warning("Cache file does not exist: %s", cache_file_path)
    except (OSError, IOError) as e:
        error_message = f"An error occurred while accessing the cache file: {e}"
        print("[ERROR] %s", error_message)
        logger.error("%s", error_message)
    return None


def fpk_is_raspberry_pi() -> bool:
    """Check if we're running on a Raspberry Pi.

    Returns:
        bool: True if running on a Raspberry Pi, False otherwise.
    """
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('Hardware') and 'BCM' in line:
                    print("[INFO] Running on a Raspberry Pi.")
                    logger.info("Running on a Raspberry Pi.")
                    return True
    except IOError as e:
        print("[WARNING] Could not access /proc/cpuinfo:", e)
        logger.warning("Could not access /proc/cpuinfo: %s", e)
    print("[INFO] Not running on a Raspberry Pi.")
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
        print("[ERROR] %s", error_message)
        logger.error(error_message)
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
                message = f"Deleted {filename}."
                print("[INFO] %s", message)
                logger.info(message)
    except Exception as e:
        error_message = f"Exception during safe_cleanup: {e}"
        print("[ERROR] %s", error_message)
        logger.error(error_message)


def load_config():
    """Load the configuration from the file.

    Returns:
        dict: The loaded configuration.
    """
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"[WARNING] Configuration file does not exist: {CONFIG_FILE_PATH}")
        logger.warning("Configuration file does not exist: %s", CONFIG_FILE_PATH)
        return {}

    try:
        with open(CONFIG_FILE_PATH, 'r') as config_file:
            config = toml.load(config_file)
            print(f"[INFO] Configuration loaded successfully from {CONFIG_FILE_PATH}")
            logger.info("Configuration loaded successfully from %s", CONFIG_FILE_PATH)
            return config
    except Exception as e:
        error_message = f"Error loading config: {e}"
        print(f"[ERROR] {error_message}")
        logger.error("Error loading config: %s", e)
        return {}


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
        print(f"[INFO] Configuration saved successfully to {CONFIG_FILE_PATH}")
        logger.info("Configuration saved successfully to %s", CONFIG_FILE_PATH)
    except Exception as e:
        error_message = f"Error saving config: {e}"
        print(f"[ERROR] {error_message}")
        logger.error("Error saving config: %s", error_message)


def fpk_set_secure_file_permissions(file_path):
    """Set secure file permissions for the specified file.

    Args:
        file_path (str): Path to the file for which to set secure permissions.
    """
    try:
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
        print(f"[INFO] Set secure file permissions for {file_path}")
        logger.info("Set secure file permissions for %s", file_path)
    except OSError as e:
        error_message = f"Failed to set secure file permissions for {file_path}: {e}"
        print(f"[ERROR] {error_message}")
        logger.error("Failed to set secure file permissions for %s: %s", file_path, e)


def fpk_unbox(directory_name: str, session, local: bool = False):
    """Unbox a flatpack from GitHub or a local directory.

    Args:
        directory_name (str): Name of the flatpack directory.
        session (httpx.Client): HTTP client session for making requests.
        local (bool): Indicates if the flatpack is local. Defaults to False.

    Returns:
        None
    """
    if not fpk_valid_directory_name(directory_name):
        message = f"Invalid directory name: '{directory_name}'."
        print(f"[ERROR] {message}")
        logger.error("Invalid directory name: '%s'", directory_name)
        return

    flatpack_dir = Path.cwd() / directory_name

    if flatpack_dir.exists() and not local:
        message = "Flatpack directory already exists."
        print(f"[ERROR] {message}")
        logger.error("%s", message)
        return
    build_dir = flatpack_dir / "build"

    if build_dir.exists():
        message = "Build directory already exists."
        print(f"[ERROR] {message}")
        logger.error("%s", message)
        return

    flatpack_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    temp_toml_path = build_dir / 'temp_flatpack.toml'

    if local:
        toml_path = flatpack_dir / 'flatpack.toml'
        if not toml_path.exists():
            message = f"flatpack.toml not found in the specified directory: '{directory_name}'."
            print(f"[ERROR] {message}")
            logger.error("%s", message)
            return
        toml_content = toml_path.read_text()
    else:
        toml_content = fpk_fetch_flatpack_toml_from_dir(directory_name, session)
        if not toml_content:
            message = f"Failed to fetch TOML content for '{directory_name}'."
            print(f"[ERROR] {message}")
            logger.error("%s", message)
            return

    temp_toml_path.write_text(toml_content)
    bash_script_content = parse_toml_to_venv_script(str(temp_toml_path), env_name=flatpack_dir)
    bash_script_path = build_dir / 'flatpack.sh'
    bash_script_path.write_text(bash_script_content)
    temp_toml_path.unlink()

    message = f"Unboxing {directory_name}..."
    print(f"[INFO] {message}")
    logger.info("%s", message)

    safe_script_path = shlex.quote(str(bash_script_path.resolve()))

    try:
        subprocess.run(['/bin/bash', safe_script_path], check=True)
        print("[INFO] All done!")
        logger.info("All done!")
        fpk_cache_unbox(str(flatpack_dir))
    except subprocess.CalledProcessError as e:
        message = f"Failed to execute the bash script: {e}"
        print(f"[ERROR] {message}")
        logger.error(message)
    except Exception as e:
        message = f"An unexpected error occurred: {e}"
        print(f"[ERROR] {message}")
        logger.error(message)


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


def fpk_verify(directory: Union[str, None]):
    """Verify a flatpack.

    Args:
        directory (Union[str, None]): The directory to use for verification. If None, a cached directory will be used if available.

    Returns:
        None
    """
    cache_file_path = HOME_DIR / ".fpk_unbox.cache"
    print(f"[INFO] Looking for cached flatpack in {cache_file_path}.")
    logger.info("Looking for cached flatpack in %s", cache_file_path)

    last_unboxed_flatpack = None

    if directory and fpk_valid_directory_name(directory):
        print(f"[INFO] Using provided directory: {directory}")
        logger.info("Using provided directory: %s", directory)
        last_unboxed_flatpack = directory
    elif cache_file_path.exists():
        print(f"[INFO] Found cached flatpack in {cache_file_path}.")
        logger.info("Found cached flatpack in %s", cache_file_path)
        last_unboxed_flatpack = cache_file_path.read_text().strip()
    else:
        message = "No cached flatpack found, and no valid directory provided."
        print(f"[ERROR] {message}")
        logger.error(message)
        return

    if not last_unboxed_flatpack:
        message = "No valid flatpack directory found."
        print(f"[ERROR] {message}")
        logger.error(message)
        return

    verification_script_path = Path(last_unboxed_flatpack) / 'build' / 'build.sh'

    if not verification_script_path.exists() or not verification_script_path.is_file():
        message = f"Verification script not found in {last_unboxed_flatpack}."
        print(f"[ERROR] {message}")
        logger.error(message)
        return

    safe_script_path = shlex.quote(str(verification_script_path.resolve()))

    try:
        env_vars = {'VERIFY_MODE': 'true'}
        subprocess.run(
            ['/bin/bash', '-u', safe_script_path],
            check=True,
            env={**env_vars, **os.environ}
        )
        print("[INFO] Verification script executed successfully.")
        logger.info("Verification script executed successfully.")
    except subprocess.CalledProcessError as e:
        message = f"An error occurred while executing the verification script: {e}"
        print(f"[ERROR] {message}")
        logger.error(message)
    except Exception as e:
        message = f"An unexpected error occurred: {e}"
        print(f"[ERROR] {message}")
        logger.error(message)


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
    parser_list = subparsers.add_parser(
        'list',
        help='List available flatpack directories.'
    )

    parser_list.set_defaults(
        func=fpk_cli_handle_list
    )

    parser_find = subparsers.add_parser(
        'find',
        help='Find model files in the current directory.'
    )

    parser_find.set_defaults(
        func=fpk_cli_handle_find
    )

    parser_help = subparsers.add_parser(
        'help',
        help='Display help for commands.'
    )

    parser_help.set_defaults(
        func=fpk_cli_handle_help
    )

    parser_version = subparsers.add_parser(
        'version',
        help='Display the version of flatpack.'
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

    # Set API key
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

    # Get API key
    parser_get_api = api_key_subparsers.add_parser(
        'get',
        help='Get the current API key'
    )

    parser_get_api.set_defaults(
        func=fpk_cli_handle_get_api_key
    )

    # Create flatpack
    parser_create = subparsers.add_parser(
        'create',
        help='Create a new flatpack.'
    )

    parser_create.add_argument(
        'input',
        nargs='?',
        default=None,
        help='The name of the flatpack to create.'
    )

    parser_create.set_defaults(
        func=fpk_cli_handle_create
    )

    # Unbox commands
    parser_unbox = subparsers.add_parser(
        'unbox',
        help='Unbox a flatpack from GitHub or a local directory.'
    )

    parser_unbox.add_argument(
        'input',
        nargs='?',
        default=None,
        help='The name of the flatpack to unbox.'
    )

    parser_unbox.add_argument(
        '--local',
        action='store_true',
        help='Unbox from a local directory instead of GitHub.'
    )

    parser_unbox.set_defaults(
        func=fpk_cli_handle_unbox
    )

    # Build commands
    parser_build = subparsers.add_parser(
        'build',
        help='Build a flatpack.'
    )

    parser_build.add_argument(
        'directory',
        nargs='?',
        default=None,
        help='The directory of the flatpack to build.'
    )

    parser_build.set_defaults(
        func=fpk_cli_handle_build
    )

    # Verify commands
    parser_verify = subparsers.add_parser(
        'verify',
        help='Verify a flatpack.'
    )

    parser_verify.add_argument(
        'directory',
        nargs='?',
        default=None,
        help='The directory of the flatpack to verify.'
    )

    parser_verify.set_defaults(
        func=fpk_cli_handle_verify
    )

    # Run server
    parser_run = subparsers.add_parser(
        'run',
        help='Run the FastAPI server.'
    )

    parser_run.add_argument(
        'input',
        nargs='?',
        default=None,
        help='The name of the flatpack to run.'
    )

    parser_run.add_argument(
        '--share',
        action='store_true',
        help='Share using ngrok.'
    )

    parser_run.set_defaults(
        func=fpk_cli_handle_run
    )

    # Vector database management
    parser_vector = subparsers.add_parser(
        'vector',
        help='Vector database management'
    )

    vector_subparsers = parser_vector.add_subparsers(
        dest='vector_command'
    )

    parser_add_text = vector_subparsers.add_parser(
        'add-texts',
        help='Add new texts to generate embeddings and store them.'
    )

    parser_add_text.add_argument(
        'texts',
        nargs='+',
        help='Texts to add.'
    )

    parser_add_text.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory path for storing the vector database and metadata files.'
    )

    parser_add_text.set_defaults(
        func=fpk_cli_handle_vector_commands
    )

    parser_search_text = vector_subparsers.add_parser(
        'search-text',
        help='Search for texts similar to the given query.'
    )

    parser_search_text.add_argument(
        'query',
        help='Text query to search for.'
    )

    parser_search_text.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory path for storing the vector database and metadata files.'
    )

    parser_search_text.set_defaults(
        func=fpk_cli_handle_vector_commands
    )

    parser_add_pdf = vector_subparsers.add_parser(
        'add-pdf',
        help='Add text from a PDF file to the vector database.'
    )

    parser_add_pdf.add_argument(
        'pdf_path',
        help='Path to the PDF file to add.'
    )

    parser_add_pdf.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory path for storing the vector database and metadata files.'
    )

    parser_add_pdf.set_defaults(
        func=fpk_cli_handle_vector_commands
    )

    parser_add_url = vector_subparsers.add_parser(
        'add-url',
        help='Add text from a URL to the vector database.'
    )

    parser_add_url.add_argument(
        'url',
        help='URL to add.'
    )

    parser_add_url.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory path for storing the vector database and metadata files.'
    )

    parser_add_url.set_defaults(
        func=fpk_cli_handle_vector_commands
    )

    parser_add_wikipedia_page = vector_subparsers.add_parser(
        'add-wikipedia',
        help='Add text from a Wikipedia page to the vector database.'
    )

    parser_add_wikipedia_page.add_argument(
        'page_title',
        help='The title of the Wikipedia page to add.'
    )

    parser_add_wikipedia_page.add_argument(
        '--data-dir',
        type=str,
        default='.',
        help='Directory path for storing the vector database and metadata files.'
    )

    parser_add_wikipedia_page.set_defaults(
        func=fpk_cli_handle_vector_commands
    )

    # Add commands for agents
    parser_agents = subparsers.add_parser(
        'agents',
        help='Manage agents'
    )

    agent_subparsers = parser_agents.add_subparsers(
        dest='agent_command',
        help='Available agent commands'
    )

    # Add command to spawn an agent
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

    # Add command to list active agents
    parser_list_agents = agent_subparsers.add_parser(
        'list',
        help='List active agents'
    )

    parser_list_agents.set_defaults(
        func=fpk_cli_handle_list_agents
    )

    # Add command to terminate an agent
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

    # Model compression
    parser_compress = subparsers.add_parser(
        'compress',
        help='Compress a model for deployment.'
    )

    parser_compress.add_argument(
        'model_id',
        type=str,
        help='The name of the Hugging Face repository (format: username/repo_name).'
    )

    parser_compress.add_argument(
        '--token',
        type=str,
        default=None,
        help='Hugging Face token for private repositories.'
    )

    parser_compress.set_defaults(
        func=fpk_cli_handle_compress
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
        print(f"[ERROR] PDF file does not exist: '{pdf_path}'.")
        logger.error("PDF file does not exist: '%s'", pdf_path)
        return

    try:
        vm.add_pdf(pdf_path, pdf_path)
        print(f"[INFO] Added text from PDF: '{pdf_path}' to the vector database.")
        logger.info("Added text from PDF: '%s' to the vector database.", pdf_path)
    except Exception as e:
        print(f"[ERROR] Failed to add PDF to the vector database: {e}")
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
            print(f"[INFO] Added text from URL: '{url}' to the vector database.")
            logger.info("Added text from URL: '%s' to the vector database.", url)
        else:
            print(f"[ERROR] URL is not accessible: '{url}'. HTTP Status Code: {response.status_code}")
            logger.error("URL is not accessible: '%s'. HTTP Status Code: %d", url, response.status_code)
    except requests.RequestException as e:
        print(f"[ERROR] Failed to access URL: '{url}'. Error: {e}")
        logger.error("Failed to access URL: '%s'. Error: %s", url, e)


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
        print("[INFO] No directory name provided. Using cached directory if available.")
        logger.info("No directory name provided. Using cached directory if available.")

    fpk_build(directory_name)


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
        print("[ERROR] Please specify a name for the new flatpack.")
        logger.error("No flatpack name specified.")
        return

    flatpack_name = args.input

    if not fpk_valid_directory_name(flatpack_name):
        print(f"[ERROR] Invalid flatpack name: '{flatpack_name}'.")
        logger.error("Invalid flatpack name: '%s'.", flatpack_name)
        return

    try:
        fpk_create(flatpack_name)
        logger.info("Flatpack '%s' created successfully.", flatpack_name)
    except Exception as e:
        print(f"[ERROR] Failed to create flatpack: {e}")
        logger.error("Failed to create flatpack: %s", e)


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
        print(f"[INFO] Virtual environment created successfully in '{venv_dir}'.")
        logger.info("Virtual environment created successfully in '%s'.", venv_dir)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to create virtual environment: {e}")
        logger.error("Failed to create virtual environment: %s", e)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while creating virtual environment: {e}")
        logger.error("An unexpected error occurred while creating virtual environment: %s", e)


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
        print("[ERROR] Please specify a valid Hugging Face repository in the format 'username/repo_name'.")
        logger.error("Invalid Hugging Face repository format specified.")
        return

    repo_name = model_id.split('/')[-1]
    local_dir = repo_name

    if os.path.exists(local_dir):
        print(f"[INFO] The model '{model_id}' is already downloaded in the directory '{local_dir}'.")
        logger.info("The model '%s' is already downloaded in the directory '%s'.", model_id, local_dir)
    else:
        try:
            if token:
                print(f"[INFO] Downloading model '{model_id}' with provided token...")
                logger.info("Downloading model '%s' with provided token...", model_id)
                snapshot_download(
                    repo_id=model_id,
                    local_dir=local_dir,
                    revision="main",
                    token=token
                )
            else:
                print(f"[INFO] Downloading model '{model_id}'...")
                logger.info("Downloading model '%s'...", model_id)
                snapshot_download(
                    repo_id=model_id,
                    local_dir=local_dir,
                    revision="main"
                )
            print(f"[INFO] Finished downloading {model_id} into the directory '{local_dir}'")
            logger.info("Finished downloading %s into the directory '%s'", model_id, local_dir)
        except Exception as e:
            print(
                f"[ERROR] Failed to download the model. Error: {e}"
            )
            logger.error("Failed to download the model. Error: %s", e)
            return

    llama_cpp_dir = "llama.cpp"
    ready_file = os.path.join(llama_cpp_dir, "ready")
    requirements_file = os.path.join(llama_cpp_dir, "requirements.txt")
    venv_dir = os.path.join(llama_cpp_dir, "venv")
    venv_python = os.path.join(venv_dir, "bin", "python")

    if not os.path.exists(llama_cpp_dir):

        git_executable = shutil.which("git")

        if not git_executable:
            print("[ERROR] The 'git' executable was not found in your PATH.")
            logger.error("The 'git' executable was not found in your PATH.")
            return

        try:
            print("[INFO] Cloning llama.cpp repository...")
            logger.info("Cloning llama.cpp repository...")

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

            print(f"[INFO] Finished cloning llama.cpp repository into '{llama_cpp_dir}'")
            logger.info("Finished cloning llama.cpp repository into '%s'", llama_cpp_dir)

        except subprocess.CalledProcessError as e:
            print(
                f"[ERROR] Failed to clone the llama.cpp repository. Error: {e}"
            )
            logger.error("Failed to clone the llama.cpp repository. Error: %s", e)
            return

    if not os.path.exists(ready_file):
        try:
            print("[INFO] Running 'make' in the llama.cpp directory...")
            logger.info("Running 'make' in the llama.cpp directory...")

            make_executable = shutil.which("make")

            if not make_executable:
                print("[ERROR] 'make' executable not found in PATH.")
                logger.error("'make' executable not found in PATH.")
                return

            subprocess.run(
                [make_executable],
                cwd=directory,
                check=True
            )

            print("[INFO] Finished running 'make' in the llama.cpp directory")
            logger.info("Finished running 'make' in the llama.cpp directory")

            if not os.path.exists(venv_dir):
                print(f"[INFO] Creating virtual environment in '{venv_dir}'...")
                logger.info("Creating virtual environment in '%s'...", venv_dir)
                create_venv(venv_dir)
                print("[INFO] Virtual environment created.")
                logger.info("Virtual environment created.")
            else:
                print(f"[INFO] Virtual environment already exists in '{venv_dir}'")
                logger.info("Virtual environment already exists in '%s'", venv_dir)

            print("[INFO] Installing llama.cpp dependencies in virtual environment...")
            logger.info("Installing llama.cpp dependencies in virtual environment...")

            pip_command = [
                "/bin/bash", "-c",
                (
                    f"source {shlex.quote(os.path.join(venv_dir, 'bin', 'activate'))} && "
                    f"pip install -r {shlex.quote(requirements_file)}"
                )
            ]
            subprocess.run(pip_command, check=True)

            print("[INFO] Finished installing llama.cpp dependencies")
            logger.info("Finished installing llama.cpp dependencies")

            with open(ready_file, 'w') as f:
                f.write("Ready")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to build llama.cpp. Error: {e}")

            logger.error(
                "Failed to build llama.cpp. Error: %s",
                e
            )

            return
        except Exception as e:
            print(
                f"[ERROR] An error occurred during the setup of llama.cpp. Error: {e}"
            )

            logger.error(
                "An error occurred during the setup of llama.cpp. Error: %s",
                e
            )

            return

    output_file = os.path.join(local_dir, f"{repo_name}-fp16.bin")
    quantized_output_file = os.path.join(local_dir, f"{repo_name}-Q4_K_M.gguf")
    outtype = "f16"

    if not os.path.exists(output_file):
        try:
            print("[INFO] Converting the model using llama.cpp...")
            logger.info("Converting the model using llama.cpp...")

            venv_activate = os.path.join(venv_dir, "bin", "activate")
            script_path = os.path.join(llama_cpp_dir, 'convert-hf-to-gguf.py')

            convert_command = [
                "/bin/bash", "-c",
                (
                    f"source {shlex.quote(venv_activate)} && {shlex.quote(venv_python)} "
                    f"{shlex.quote(script_path)} {shlex.quote(local_dir)} --outfile "
                    f"{shlex.quote(output_file)} --outtype {shlex.quote(outtype)}"
                )
            ]
            subprocess.run(convert_command, check=True)

            print(
                f"[INFO] Conversion complete. The model has been compressed and saved as '{output_file}'"
            )
            logger.info("Conversion complete. The model has been compressed and saved as '%s'", output_file)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Conversion failed. Error: {e}")
            logger.error("Conversion failed. Error: %s", e)
            return
        except Exception as e:
            print(f"[ERROR] An error occurred during the model conversion. Error: {e}")
            logger.error("An error occurred during the model conversion. Error: %s", e)
            return
    else:
        print(
            f"[INFO] The model has already been converted and saved as '{output_file}'."
        )
        logger.info("The model has already been converted and saved as '%s'.", output_file)
    if os.path.exists(output_file):
        try:
            print("[INFO] Quantizing the model...")
            logger.info("Quantizing the model...")

            quantize_command = [
                os.path.join(llama_cpp_dir, 'llama-quantize'),
                output_file,
                quantized_output_file,
                "Q4_K_M"
            ]
            subprocess.run(quantize_command, check=True)

            print(
                f"[INFO] Quantization complete. The quantized model has been saved as '{quantized_output_file}'."
            )
            logger.info("Quantization complete. The quantized model has been saved as '%s'.", quantized_output_file)

            print(f"[INFO] Deleting the original .bin file '{output_file}'...")
            logger.info("Deleting the original .bin file '%s'...", output_file)

            os.remove(output_file)

            print(f"[INFO] Deleted the original .bin file '{output_file}'.")
            logger.info("Deleted the original .bin file '%s'.", output_file)

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Quantization failed. Error: {e}")
            logger.error("Quantization failed. Error: %s", e)
            return
        except Exception as e:
            print(f"[ERROR] An error occurred during the quantization process. Error: {e}")
            logger.error("An error occurred during the quantization process. Error: %s", e)
            return
    else:
        print(f"[ERROR] The original model file '{output_file}' does not exist.")
        logger.error("The original model file '%s' does not exist.", output_file)


def fpk_cli_handle_find(args, session):
    """Handle the 'find' command to search for model files."""
    logger.info("Searching for files...")
    model_files = fpk_find_models()
    if model_files:
        print("[INFO] Found the following files:")
        logger.info("Found the following files:")
        for model_file in model_files:
            print(f" - {model_file}")
            logger.info(" - %s", model_file)
    else:
        print("[INFO] No files found.")
        logger.info("No files found.")


def fpk_cli_handle_get_api_key(args, session):
    """Handle the 'get' command to retrieve the API key."""
    logger.info("Retrieving API key...")
    api_key = fpk_get_api_key()
    if api_key:
        print(f"API Key: {api_key}")
        logger.info("API Key: %s", api_key)
    else:
        print("[ERROR] No API key found.")
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
            print(f"[ERROR] Command '{args.command}' not found.")
            logger.error("Command '%s' not found.", args.command)
    else:
        parser.print_help()
        logger.info("Displayed general help.")


def fpk_cli_handle_list(args, session):
    """Handle the 'list' command to fetch and print the list of directories."""
    directories = fpk_list_directories(session)
    if directories:
        print(directories)
        logger.info("Directories found: %s", directories)
    else:
        print("[ERROR] No directories found.")
        logger.error("No directories found.")


def fpk_cli_handle_list_agents(args, session):
    """List active agents."""
    agent_manager = AgentManager()
    agents = agent_manager.list_agents()

    if agents:
        print("Active agents:")
        logger.info("Active agents:")
        for agent in agents:
            agent_info = (f"PID: {agent['pid']}, Script: {agent['script']}, "
                          f"Start Time: {agent['start_time']}, Port: {agent['port']}")
            print(agent_info)
            logger.info(agent_info)
    else:
        print("[INFO] No active agents found.")
        logger.info("No active agents found.")


atexit.register(fpk_safe_cleanup)

app = FastAPI()


class EndpointFilter(logging.Filter):
    def filter(self, record):
        return 'GET /api/heartbeat' not in record.getMessage()


uvicorn_logger = logging.getLogger("uvicorn.access")
uvicorn_logger.addFilter(EndpointFilter())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

flatpack_directory = None


def escape_special_chars(content: str) -> str:
    """Escape special characters in a given string."""
    return content.replace('"', '\\"')


def unescape_special_chars(content: str) -> str:
    """Unescape special characters in a given string."""
    return content.replace('\\"', '"')


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


def set_token(token: str):
    try:
        config = load_config()
        config['token'] = token
        save_config(config)
        print("[INFO] Token set successfully!")
        logger.info("Token set successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to set token: {str(e)}")
        logger.error("Failed to set token: %s", str(e))


def get_token() -> Optional[str]:
    """Retrieve the token from the configuration file."""
    config = load_config()
    return config.get('token')


def validate_api_token(api_token: str) -> bool:
    """Validate the API token."""
    return api_token == get_token()


def authenticate_token(request: Request):
    """Authenticate the token."""
    global VALIDATION_ATTEMPTS
    token = request.headers.get('Authorization')
    stored_token = get_token()

    if not stored_token:
        return

    print(f"[DEBUG] Received token: {token}, Stored token: {stored_token}")
    if token is None or token != f"Bearer {stored_token}":
        VALIDATION_ATTEMPTS += 1
        if VALIDATION_ATTEMPTS >= MAX_ATTEMPTS:
            shutdown_server()
        raise HTTPException(status_code=403, detail="Invalid or missing token")


def shutdown_server():
    """Shutdown the FastAPI server."""
    logging.getLogger("uvicorn.error").info("Shutting down the server after maximum validation attempts.")
    os._exit(0)


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
    return JSONResponse(content={"message": "Invalid API token."}, status_code=403)


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


@app.post("/api/build")
async def build_flatpack(
        request: Request,
        token: str = Depends(authenticate_token)
):
    """Trigger the build process for the flatpack."""
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    try:
        print(f"[INFO] Building flatpack located at {flatpack_directory}")
        logger.info("Building flatpack located at %s", flatpack_directory)

        fpk_build(flatpack_directory)

        return JSONResponse(
            content={"flatpack": flatpack_directory, "message": "Build process completed successfully."},
            status_code=200)
    except Exception as e:
        logger.error("Build process failed: %s", e)
        return JSONResponse(content={"flatpack": flatpack_directory, "message": f"Build process failed: {e}"},
                            status_code=500)


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


@app.get("/api/heartbeat")
async def heartbeat():
    """Endpoint to check the server heartbeat."""
    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    return JSONResponse(content={"server_time": current_time}, status_code=200)


def setup_static_directory(fastapi_app: FastAPI, directory: str):
    """Setup the static directory for serving static files."""
    global flatpack_directory
    flatpack_directory = os.path.abspath(directory)

    if os.path.exists(flatpack_directory) and os.path.isdir(flatpack_directory):
        static_dir = os.path.join(flatpack_directory, 'build')
        fastapi_app.mount(
            "/",
            StaticFiles(directory=static_dir, html=True),
            name="static"
        )
        print(f"[INFO] Static files will be served from: {static_dir}")
        logger.info("Static files will be served from: %s", static_dir)
    else:
        print(f"[ERROR] The directory '{flatpack_directory}' does not exist or is not a directory.")
        logger.error("The directory '%s' does not exist or is not a directory.", flatpack_directory)
        raise ValueError(error_message)


def generate_secure_token(length=32):
    """Generate a secure token of the specified length.

    Args:
        length (int): The length of the token to generate. Default is 32.

    Returns:
        str: A securely generated token.
    """
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def fpk_cli_handle_run(args, session):
    """Handle the 'run' command to start the FastAPI server.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    if not args.input:
        print("[ERROR] Please specify a flatpack for the run command.")
        logger.error("Please specify a flatpack for the run command.")
        return

    directory = args.input

    if os.path.exists(directory) and os.path.isdir(directory):
        print(f"[INFO] Using provided flatpack: {directory}")
        logger.info("Using provided flatpack: %s", directory)
    else:
        print(f"[ERROR] The flatpack '{directory}' does not exist.")
        logger.error("The flatpack '%s' does not exist.", directory)
        return

    if args.share:
        fpk_check_ngrok_auth()

    token = generate_secure_token()

    print(f"[INFO] Generated API token: {token}")
    logger.info("Generated API token: %s", token)

    print("[INFO] Please save this API token securely. You will not be able to retrieve it again.")
    logger.info("Please save this API token securely. You will not be able to retrieve it again.")

    try:
        while True:
            confirmation = input("Have you saved the API token? Type 'YES' to continue: ").strip().upper()
            if confirmation == 'YES':
                break
            print("[INFO] Please save the API token before continuing.")
            logger.info("Please save the API token before continuing.")
    except KeyboardInterrupt:
        print(
            "[ERROR] Process interrupted by user. Please save the API token and try again."
        )
        logger.error("Process interrupted by user. Please save the API token and try again.")
        sys.exit(1)

    set_token(token)
    setup_static_directory(app, directory)

    try:
        port = 8000

        if args.share:
            listener = ngrok.forward(port, authtoken_from_env=True)
            public_url = listener.url()
            print(f"[INFO] Ingress established at {public_url}")
            logger.info("Ingress established at %s", public_url)

        config = uvicorn.Config(app, host="127.0.0.1", port=port)
        global uvicorn_server
        uvicorn_server = uvicorn.Server(config)
        uvicorn_server.run()
    except KeyboardInterrupt:
        print("[INFO] FastAPI server has been stopped.")
        logger.info("FastAPI server has been stopped.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during server run: {e}")
        logger.error("An unexpected error occurred during server run: %s", e)
    finally:
        if args.share:
            ngrok.disconnect(public_url)
            print(f"[INFO] Disconnected ngrok ingress at {public_url}")
            logger.info("Disconnected ngrok ingress at %s", public_url)


def fpk_cli_handle_set_api_key(args, session):
    """Handle the 'set' command to set the API key.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    print(f"[INFO] Setting API key: {args.api_key}")
    logger.info("Setting API key: %s", args.api_key)

    api_key = args.api_key
    config = load_config()
    config['api_key'] = api_key
    save_config(config)

    print("[INFO] API key set successfully!")
    logger.info("API key set successfully!")

    try:
        test_key = fpk_get_api_key()
        if test_key == api_key:
            print("[INFO] Verification successful: API key matches.")
            logger.info("Verification successful: API key matches.")
        else:
            print("[ERROR] Verification failed: API key does not match.")
            logger.error("Verification failed: API key does not match.")
    except Exception as e:
        print(f"[ERROR] Error during API key verification: {e}")
        logger.error("Error during API key verification: %s", e)


def fpk_cli_handle_spawn_agent(args, session):
    """Handle the 'spawn' command to spawn a new agent with a script.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    agent_manager = AgentManager()

    try:
        pid = agent_manager.spawn_agent(args.script_path)
        print(f"[INFO] Agent spawned with PID: {pid}")
        logger.info("Agent spawned with PID: %s", pid)
    except Exception as e:
        print(f"[ERROR] Failed to spawn agent: {e}")
        logger.error("Failed to spawn agent: %s", e)


def fpk_cli_handle_terminate_agent(args, session):
    """Handle the 'terminate' command to terminate an active agent.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    agent_manager = AgentManager()

    try:
        agent_manager.terminate_agent(args.pid)
        print(f"[INFO] Agent with PID {args.pid} terminated successfully.")
        logger.info("Agent with PID %s terminated successfully.", args.pid)
    except Exception as e:
        print(f"[ERROR] Failed to terminate agent with PID {args.pid}: {e}")
        logger.error("Failed to terminate agent with PID %s: %s", args.pid, e)


def fpk_cli_handle_unbox(args, session):
    """Handle the 'unbox' command to unbox a flatpack from GitHub or a local directory.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    if not args.input:
        print("[ERROR] Please specify a flatpack for the unbox command.")
        logger.error("No flatpack specified for the unbox command.")
        return

    directory_name = args.input
    existing_dirs = fpk_fetch_github_dirs(session)

    if directory_name not in existing_dirs and not args.local:
        print(f"[ERROR] The flatpack '{directory_name}' does not exist.")
        logger.error("The flatpack '%s' does not exist.", directory_name)
        return

    fpk_display_disclaimer(directory_name, local=args.local)

    while True:
        user_response = input().strip().upper()
        if user_response == "YES":
            break
        if user_response == "NO":
            print("[INFO] Installation aborted by user.")
            logger.info("Installation aborted by user.")
            return
        print("[ERROR] Invalid input. Please type 'YES' to accept or 'NO' to decline.")
        logger.error("Invalid input from user. Expected 'YES' or 'NO'.")

    if args.local:
        local_directory_path = Path(directory_name)
        if not local_directory_path.exists() or not local_directory_path.is_dir():
            print(f"[ERROR] Local directory does not exist: '{directory_name}'.")
            logger.error("Local directory does not exist: '%s'.", directory_name)
            return
        toml_path = local_directory_path / 'flatpack.toml'
        if not toml_path.exists():
            print(
                f"[ERROR] flatpack.toml not found in '{directory_name}'."
            )
            logger.error(
                "flatpack.toml not found in the specified directory: '%s'.",
                directory_name
            )
            return

    print(f"[INFO] Directory name resolved to: '{directory_name}'")
    logger.info("Directory name resolved to: '%s'", directory_name)

    try:
        fpk_unbox(directory_name, session, local=args.local)
        print(f"[INFO] Unboxed flatpack '{directory_name}' successfully.")
        logger.info("Unboxed flatpack '%s' successfully.", directory_name)
    except Exception as e:
        print(f"[ERROR] Failed to unbox flatpack '{directory_name}': {e}")
        logger.error(f"Failed to unbox flatpack '{directory_name}': %s", e)


def fpk_cli_handle_verify(args, session):
    """Handle the 'verify' command to verify a flatpack.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    directory_name = args.directory
    if not directory_name:
        print("[ERROR] Please specify a directory for the verify command.")
        logger.error("No directory specified for the verify command.")
        return

    print(f"[INFO] Verifying flatpack in directory: {directory_name}")
    logger.info("Verifying flatpack in directory: %s", directory_name)

    try:
        fpk_verify(directory_name)
        print(f"[INFO] Verification successful for directory: {directory_name}")
        logger.info("Verification successful for directory: %s", directory_name)
    except Exception as e:
        print(f"[ERROR] Verification failed for directory '{directory_name}': {e}")
        logger.error("Verification failed for directory '%s': %s", directory_name, e)


def fpk_cli_handle_version(args, session):
    """Handle the 'version' command to display the version of flatpack.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    print(f"[INFO] Flatpack version: {VERSION}")
    logger.info("Flatpack version: %s", VERSION)


def fpk_initialize_vector_manager(args):
    """Initialize the Vector Manager.

    Args:
        args: The command-line arguments.

    Returns:
        VectorManager: An instance of VectorManager.
    """
    data_dir = getattr(args, 'data_dir', '.')
    print(f"[INFO] Initializing Vector Manager and data directory: {data_dir}")
    logger.info(
        "Initializing Vector Manager and data directory: %s",
        data_dir
    )
    return VectorManager(model_id='all-MiniLM-L6-v2', directory=data_dir)


def fpk_cli_handle_vector_commands(args, session, vm):
    """Handle vector database commands.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
        vm: The Vector Manager instance.
    """
    print("[INFO] Handling vector commands...")
    logger.info("Handling vector commands...")

    if args.vector_command == 'add-texts':
        vm.add_texts(args.texts, "manual")
        print(f"[INFO] Added {len(args.texts)} texts to the database.")
        logger.info("Added %d texts to the database.", len(args.texts))
    elif args.vector_command == 'search-text':
        results = vm.search_vectors(args.query)
        if results:
            print("[INFO] Search results:")
            logger.info("Search results:")
            for result in results:
                print(f"{result['id']}: {result['text']}\n")
                logger.info("%s: %s", result['id'], result['text'])
        else:
            print("[INFO] No results found.")
            logger.info("No results found.")
    elif args.vector_command == 'add-pdf':
        fpk_cli_handle_add_pdf(args.pdf_path, vm)
    elif args.vector_command == 'add-url':
        fpk_cli_handle_add_url(args.url, vm)
    elif args.vector_command == 'add-wikipedia':
        vm.add_wikipedia_page(args.page_title)
        print(
            f"[INFO] Added text from Wikipedia page: '{args.page_title}' "
            "to the vector database."
        )
        logger.info(
            "Added text from Wikipedia page: '%s' to the vector database.",
            args.page_title
        )
    else:
        print("[ERROR] Unknown vector command.")
        logger.error("Unknown vector command.")


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
        print(f"[ERROR] An unexpected error occurred: {str(e)}")
        logger.error("An unexpected error occurred: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
