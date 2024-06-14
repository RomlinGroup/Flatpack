import argparse
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
from pathlib import Path
from typing import List, Optional, Union
from zipfile import ZipFile

import httpx
import requests
import toml
import uvicorn

from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from pydantic import BaseModel

from .agent_manager import AgentManager
from .parsers import parse_toml_to_venv_script
from .session_manager import SessionManager
from .vector_manager import VectorManager

# Configuration constants
HOME_DIR = Path.home() / ".fpk"
HOME_DIR.mkdir(exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/romlingroup/flatpack/main/warehouse"
CONFIG_FILE_PATH = HOME_DIR / ".fpk_config.toml"
GITHUB_REPO_URL = "https://api.github.com/repos/romlingroup/flatpack"
KEY_FILE_PATH = HOME_DIR / ".fpk_encryption_key"
VERSION = version("flatpack")

MAX_ATTEMPTS = 5
VALIDATION_ATTEMPTS = 0


def setup_logging(log_file_path: Path):
    """Set up logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.WARNING)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Initialize logging
log_file_path = HOME_DIR / "fpk_local_only.log"
logger = setup_logging(log_file_path)

uvicorn_server = None


def handle_termination_signal(signal_number, frame):
    """Handle termination signals for graceful shutdown."""
    logger.info(f"Received termination signal ({signal_number}), shutting down...")
    uvicorn_server.should_exit = True


signal.signal(signal.SIGINT, handle_termination_signal)
signal.signal(signal.SIGTERM, handle_termination_signal)


def fpk_build(directory: Union[str, None]):
    """Build a flatpack.

    Args:
        directory (Union[str, None]): The directory to use for building the flatpack. If None, a cached directory will be used if available.

    Returns:
        None
    """
    cache_file_path = HOME_DIR / ".fpk_unbox.cache"
    print(f"[INFO] Looking for cached flatpack in {cache_file_path}.")
    logger.info(f"Looking for cached flatpack in {cache_file_path}.")

    last_unboxed_flatpack = None

    if directory and fpk_valid_directory_name(directory):
        print(f"[INFO] Using provided directory: {directory}")
        logger.info(f"Using provided directory: {directory}")
        last_unboxed_flatpack = directory
    elif cache_file_path.exists():
        print(f"[INFO] Found cached flatpack in {cache_file_path}.")
        logger.info(f"Found cached flatpack in {cache_file_path}.")
        last_unboxed_flatpack = cache_file_path.read_text().strip()
    else:
        print("[ERROR] No cached flatpack found, and no valid directory provided.")
        logger.error("No cached flatpack found, and no valid directory provided.")
        return

    if not last_unboxed_flatpack:
        print("[ERROR] No valid flatpack directory found.")
        logger.error("No valid flatpack directory found.")
        return

    building_script_path = Path(last_unboxed_flatpack) / 'build' / 'build.sh'

    log_file_time = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S')
    log_file_path = Path(last_unboxed_flatpack) / 'build' / 'logs' / f"build_{log_file_time}.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    if not building_script_path.exists() or not building_script_path.is_file():
        print(f"[ERROR] Building script not found in {last_unboxed_flatpack}.")
        logger.error(f"Building script not found in {last_unboxed_flatpack}.")
        return

    safe_script_path = shlex.quote(str(building_script_path.resolve()))

    try:
        with open(log_file_path, 'w') as log_file:
            process = subprocess.Popen(
                ['bash', '-u', safe_script_path],
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
        logger.error(f"An error occurred while executing the build script: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        logger.error(f"An unexpected error occurred: {e}")


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
        logger.info(f"Cached the directory name '{directory_name}' to {cache_file_path}.")
    except IOError as e:
        print(f"[ERROR] Failed to cache the directory name '{directory_name}': {e}")
        logger.error(f"Failed to cache the directory name '{directory_name}': {e}")


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
        logger.error(f"Error: {message}")
        sys.exit(1)
    else:
        message = "NGROK_AUTHTOKEN is set."
        print(f"[INFO] {message}")
        logger.info(message)


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
        message = f"Invalid color '{color}' provided. Returning the original text."
        print(f"[ERROR] {message}")
        logger.error(message)
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
    # Validate flatpack name
    if not re.match(r'^[a-z0-9-]+$', flatpack_name):
        error_message = "Invalid name format. Only lowercase letters, numbers, and hyphens are allowed."
        print(f"[ERROR] {error_message}")
        logger.error(error_message)
        raise ValueError(error_message)

    flatpack_name = flatpack_name.lower().replace(' ', '-')
    current_dir = os.getcwd()

    # Download and extract template
    try:
        template_dir = fpk_download_and_extract_template(repo_url, current_dir)
    except Exception as e:
        error_message = f"Failed to download and extract template: {e}"
        print(f"[ERROR] {error_message}")
        logger.error(error_message)
        return

    flatpack_dir = os.path.join(current_dir, flatpack_name)

    # Create flatpack directory
    try:
        os.makedirs(flatpack_dir, exist_ok=True)
        print(f"[INFO] Created flatpack directory: {flatpack_dir}")
        logger.info(f"Created flatpack directory: {flatpack_dir}")
    except OSError as e:
        error_message = f"Failed to create flatpack directory: {e}"
        print(f"[ERROR] {error_message}")
        logger.error(error_message)
        return

    # Copy files from template to flatpack directory
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
        logger.info(f"Copied template files to flatpack directory: {flatpack_dir}")
    except OSError as e:
        error_message = f"Failed to copy template files: {e}"
        print(f"[ERROR] {error_message}")
        logger.error(error_message)
        return

    # Edit specific files
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
        logger.info(f"Edited template files for flatpack: {flatpack_name}")
    except OSError as e:
        error_message = f"Failed to edit template files: {e}"
        print(f"[ERROR] {error_message}")
        logger.error(error_message)
        return

    # Remove the temporary template directory
    try:
        shutil.rmtree(template_dir)
        print(f"[INFO] Removed temporary template directory: {template_dir}")
        logger.info(f"Removed temporary template directory: {template_dir}")
    except OSError as e:
        error_message = f"Failed to remove temporary template directory: {e}"
        print(f"[ERROR] {error_message}")
        logger.error(error_message)
        return

    print(f"[INFO] Successfully created {flatpack_name}.")
    logger.info(f"Successfully created {flatpack_name}.")


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
    logger.info("Displayed disclaimer for flatpack '{}' with local set to {}.".format(directory_name, local))


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
        response.raise_for_status()  # Raise an HTTPError for bad responses
        with ZipFile(BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"[INFO] Downloaded and extracted template from {repo_url} to {dest_dir}")
        logger.info(f"Downloaded and extracted template from {repo_url} to {dest_dir}")
        return template_dir
    except requests.RequestException as e:
        error_message = f"Failed to download template from {repo_url}: {e}"
        print(f"[ERROR] {error_message}")
        logger.error(error_message)
        raise RuntimeError(error_message)
    except (OSError, IOError) as e:
        error_message = f"Failed to extract template to {dest_dir}: {e}"
        print(f"[ERROR] {error_message}")
        logger.error(error_message)
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
        logger.error(message)
        return None

    toml_url = f"{BASE_URL}/{directory_name}/flatpack.toml"
    try:
        response = session.get(toml_url)
        response.raise_for_status()
        print(f"[INFO] Successfully fetched TOML from {toml_url}")
        logger.info(f"Successfully fetched TOML from {toml_url}")
        return response.text
    except httpx.HTTPStatusError as e:
        message = f"HTTP error occurred: {e.response.status_code} - {e.response.text}"
        print(f"[ERROR] {message}")
        logger.error(message)
        return None
    except httpx.RequestError as e:
        message = f"Network error occurred: {e}"
        print(f"[ERROR] {message}")
        logger.error(message)
        return None
    except Exception as e:
        message = f"An unexpected error occurred: {e}"
        print(f"[ERROR] {message}")
        logger.error(message)
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
                if isinstance(item, dict) and item.get('type') == 'dir' and
                   item.get('name', '').lower()
            ]
            logger.info(f"Fetched directory names from GitHub: {directories}")
            return sorted(directories)
        else:
            message = f"Unexpected response format from GitHub: {json_data}"
            print(f"[ERROR] {message}")
            logger.error(message)
            return []
    except httpx.HTTPError as e:
        message = f"Unable to connect to GitHub: {e}"
        print(f"[ERROR] {message}")
        logger.error(message)
        sys.exit(1)
    except (ValueError, KeyError) as e:
        message = f"Error processing the response from GitHub: {e}"
        print(f"[ERROR] {message}")
        logger.error(message)
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
    logger.info(f"Searching for model files in directory: {directory_path}")

    model_file_formats = ['.caffemodel', '.ckpt', '.gguf', '.h5', '.json', '.mar', '.mlmodel', '.model', '.onnx',
                          '.params', '.pb', '.pkl', '.pickle', '.pt', '.pth', '.sav', '.tflite', '.weights']
    model_files = []

    try:
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.endswith(fmt) for fmt in model_file_formats):
                    model_file_path = os.path.join(root, file)
                    model_files.append(model_file_path)
                    logger.info(f"Found model file: {model_file_path}")
        print(f"[INFO] Found {len(model_files)} model file(s).")
        logger.info(f"Total number of model files found: {len(model_files)}")
    except Exception as e:
        error_message = f"An error occurred while searching for model files: {e}"
        print(f"[ERROR] {error_message}")
        logger.error(error_message)

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
            print(f"[INFO] API key retrieved successfully.")
            logger.info("API key retrieved successfully.")
        else:
            print(f"[WARNING] API key not found in the configuration.")
            logger.warning("API key not found in the configuration.")
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
                print(f"[INFO] Last unboxed flatpack directory retrieved: {last_flatpack}")
                logger.info(f"Last unboxed flatpack directory retrieved: {last_flatpack}")
                return last_flatpack
        else:
            print(f"[WARNING] Cache file does not exist: {cache_file_path}")
            logger.warning(f"Cache file does not exist: {cache_file_path}")
    except (OSError, IOError) as e:
        error_message = f"An error occurred while accessing the cache file: {e}"
        print(f"[ERROR] {error_message}")
        logger.error(error_message)
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
                    print(f"[INFO] Running on a Raspberry Pi.")
                    logger.info("Running on a Raspberry Pi.")
                    return True
    except IOError as e:
        print(f"[WARNING] Could not access /proc/cpuinfo: {e}")
        logger.warning(f"Could not access /proc/cpuinfo: {e}")
    print(f"[INFO] Not running on a Raspberry Pi.")
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
        logger.info(f"Fetched directories: {directories_str}")
        return directories_str
    except Exception as e:
        error_message = f"An error occurred while listing directories: {e}"
        print(f"[ERROR] {error_message}")
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
                print(f"[INFO] {message}")
                logger.info(message)
    except Exception as e:
        error_message = f"Exception during safe_cleanup: {e}"
        print(f"[ERROR] {error_message}")
        logger.error(error_message)


def load_config():
    """Load the configuration from the file.

    Returns:
        dict: The loaded configuration.
    """
    if not os.path.exists(CONFIG_FILE_PATH):
        print(f"[WARNING] Configuration file does not exist: {CONFIG_FILE_PATH}")
        logger.warning(f"Configuration file does not exist: {CONFIG_FILE_PATH}")
        return {}

    try:
        with open(CONFIG_FILE_PATH, 'r') as config_file:
            config = toml.load(config_file)
            print(f"[INFO] Configuration loaded successfully from {CONFIG_FILE_PATH}")
            logger.info(f"Configuration loaded successfully from {CONFIG_FILE_PATH}")
            return config
    except Exception as e:
        error_message = f"Error loading config: {e}"
        print(f"[ERROR] {error_message}")
        logger.error(error_message)
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
        logger.info(f"Configuration saved successfully to {CONFIG_FILE_PATH}")
    except Exception as e:
        error_message = f"Error saving config: {e}"
        print(f"[ERROR] {error_message}")
        logger.error(error_message)


def fpk_set_secure_file_permissions(file_path):
    """Set secure file permissions for the specified file.

    Args:
        file_path (str): Path to the file for which to set secure permissions.
    """
    try:
        os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)
        print(f"[INFO] Set secure file permissions for {file_path}")
        logger.info(f"Set secure file permissions for {file_path}")
    except OSError as e:
        error_message = f"Failed to set secure file permissions for {file_path}: {e}"
        print(f"[ERROR] {error_message}")
        logger.error(error_message)


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
        logger.error(message)
        return

    flatpack_dir = Path.cwd() / directory_name

    if flatpack_dir.exists() and not local:
        message = "Flatpack directory already exists."
        print(f"[ERROR] {message}")
        logger.error(message)
        return
    else:
        build_dir = flatpack_dir / "build"

        if build_dir.exists():
            message = "Build directory already exists."
            print(f"[ERROR] {message}")
            logger.error(message)
            return

    flatpack_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    temp_toml_path = build_dir / 'temp_flatpack.toml'
    if local:
        toml_path = flatpack_dir / 'flatpack.toml'
        if not toml_path.exists():
            message = f"flatpack.toml not found in the specified directory: '{directory_name}'."
            print(f"[ERROR] {message}")
            logger.error(message)
            return
        toml_content = toml_path.read_text()
    else:
        toml_content = fpk_fetch_flatpack_toml_from_dir(directory_name, session)
        if not toml_content:
            message = f"Failed to fetch TOML content for '{directory_name}'."
            print(f"[ERROR] {message}")
            logger.error(message)
            return

    temp_toml_path.write_text(toml_content)
    bash_script_content = parse_toml_to_venv_script(str(temp_toml_path), '3.11.8', directory_name)
    bash_script_path = build_dir / 'flatpack.sh'
    bash_script_path.write_text(bash_script_content)
    temp_toml_path.unlink()

    message = f"Unboxing {directory_name}..."
    print(f"[INFO] {message}")
    logger.info(message)
    safe_script_path = shlex.quote(str(bash_script_path.resolve()))
    try:
        result = subprocess.run(['bash', safe_script_path], check=True)
        print(f"[INFO] All done!")
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
    """Validate that the directory name contains only alphanumeric characters, dashes, and underscores.

    Args:
        name (str): The directory name to validate.

    Returns:
        bool: True if the name is valid, False otherwise.
    """
    pattern = r'^[\w-]+$'
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
    logger.info(f"Looking for cached flatpack in {cache_file_path}.")

    last_unboxed_flatpack = None

    if directory and fpk_valid_directory_name(directory):
        print(f"[INFO] Using provided directory: {directory}")
        logger.info(f"Using provided directory: {directory}")
        last_unboxed_flatpack = directory
    elif cache_file_path.exists():
        print(f"[INFO] Found cached flatpack in {cache_file_path}.")
        logger.info(f"Found cached flatpack in {cache_file_path}.")
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
        result = subprocess.run(['bash', '-u', safe_script_path], check=True, env={**env_vars, **os.environ})
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
        func=lambda args, session: fpk_cli_handle_list(args, session)
    )

    parser_find = subparsers.add_parser(
        'find',
        help='Find model files in the current directory.'
    )

    parser_find.set_defaults(
        func=lambda args, session: fpk_cli_handle_find(args, session)
    )

    parser_help = subparsers.add_parser(
        'help',
        help='Display help for commands.'
    )

    parser_help.set_defaults(
        func=lambda args, session: fpk_cli_handle_help(args, session)
    )

    parser_version = subparsers.add_parser(
        'version',
        help='Display the version of flatpack.'
    )

    parser_version.set_defaults(
        func=lambda args, session: fpk_cli_handle_version(args, session)
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
        func=lambda args, session: fpk_cli_handle_set_api_key(args, session)
    )

    # Get API key
    parser_get_api = api_key_subparsers.add_parser(
        'get',
        help='Get the current API key'
    )

    parser_get_api.set_defaults(
        func=lambda args, session: fpk_cli_handle_get_api_key(args, session)
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
        func=lambda args, session: fpk_cli_handle_create(args, session)
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
        func=lambda args, session: fpk_cli_handle_unbox(args, session)
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
        func=lambda args, session: fpk_cli_handle_build(args, session)
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
        func=lambda args, session: fpk_cli_handle_verify(args, session)
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
        func=lambda args, session: fpk_cli_handle_run(args, session)
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
        func=lambda args, session, vm: fpk_cli_handle_vector_commands(args, session, vm)
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
        func=lambda args, session, vm: fpk_cli_handle_vector_commands(args, session, vm)
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
        func=lambda args, session, vm: fpk_cli_handle_vector_commands(args, session, vm)
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
        func=lambda args, session, vm: fpk_cli_handle_vector_commands(args, session, vm)
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
        func=lambda args, session, vm: fpk_cli_handle_vector_commands(args, session, vm)
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
        func=lambda args, session: fpk_cli_handle_spawn_agent(args, session)
    )

    # Add command to list active agents
    parser_list_agents = agent_subparsers.add_parser(
        'list',
        help='List active agents'
    )

    parser_list_agents.set_defaults(
        func=lambda args, session: fpk_cli_handle_list_agents(args, session)
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
        func=lambda args, session: fpk_cli_handle_terminate_agent(args, session)
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
        func=lambda args, session: fpk_cli_handle_compress(args, session)
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
        logger.error(f"PDF file does not exist: '{pdf_path}'.")
        return

    try:
        vm.add_pdf(pdf_path, pdf_path)
        print(f"[INFO] Added text from PDF: '{pdf_path}' to the vector database.")
        logger.info(f"Added text from PDF: '{pdf_path}' to the vector database.")
    except Exception as e:
        print(f"[ERROR] Failed to add PDF to the vector database: {e}")
        logger.error(f"Failed to add PDF to the vector database: {e}")


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
            logger.info(f"Added text from URL: '{url}' to the vector database.")
        else:
            print(f"[ERROR] URL is not accessible: '{url}'. HTTP Status Code: {response.status_code}")
            logger.error(f"URL is not accessible: '{url}'. HTTP Status Code: {response.status_code}")
    except requests.RequestException as e:
        print(f"[ERROR] Failed to access URL: '{url}'. Error: {e}")
        logger.error(f"Failed to access URL: '{url}'. Error: {e}")


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
        logger.error(f"Invalid flatpack name: '{flatpack_name}'.")
        return

    try:
        fpk_create(flatpack_name)
        logger.info(f"Flatpack '{flatpack_name}' created successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to create flatpack: {e}")
        logger.error(f"Failed to create flatpack: {e}")


def create_venv(venv_dir: str):
    """
    Create a virtual environment in the specified directory.

    Args:
        venv_dir (str): The directory where the virtual environment will be created.

    Returns:
        None
    """
    try:
        subprocess.run(
            ["python3", "-m", "venv", venv_dir],
            check=True
        )
        print(f"[INFO] Virtual environment created successfully in '{venv_dir}'.")
        logger.info(f"Virtual environment created successfully in '{venv_dir}'.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to create virtual environment: {e}")
        logger.error(f"Failed to create virtual environment: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while creating virtual environment: {e}")
        logger.error(f"An unexpected error occurred while creating virtual environment: {e}")


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
        logger.info(f"The model '{model_id}' is already downloaded in the directory '{local_dir}'.")
    else:
        try:
            if token:
                print(f"[INFO] Downloading model '{model_id}' with provided token...")
                logger.info(f"Downloading model '{model_id}' with provided token...")
                snapshot_download(
                    repo_id=model_id,
                    local_dir=local_dir,
                    revision="main",
                    token=token
                )
            else:
                print(f"[INFO] Downloading model '{model_id}'...")
                logger.info(f"Downloading model '{model_id}'...")
                snapshot_download(
                    repo_id=model_id,
                    local_dir=local_dir,
                    revision="main"
                )
            print(f"[INFO] Finished downloading {model_id} into the directory '{local_dir}'")
            logger.info(f"Finished downloading {model_id} into the directory '{local_dir}'")
        except Exception as e:
            print(
                f"[ERROR] Failed to download the model. Please check your internet connection and try again. Error: {e}")
            logger.error(f"Failed to download the model. Error: {e}")
            return

    llama_cpp_dir = "llama.cpp"
    ready_file = os.path.join(llama_cpp_dir, "ready")
    requirements_file = os.path.join(llama_cpp_dir, "requirements.txt")
    venv_dir = os.path.join(llama_cpp_dir, "venv")
    venv_python = os.path.join(venv_dir, "bin", "python")

    if not os.path.exists(llama_cpp_dir):
        try:
            print("[INFO] Cloning llama.cpp repository...")
            logger.info("Cloning llama.cpp repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp", llama_cpp_dir],
                check=True
            )
            print(f"[INFO] Finished cloning llama.cpp repository into '{llama_cpp_dir}'")
            logger.info(f"Finished cloning llama.cpp repository into '{llama_cpp_dir}'")
        except subprocess.CalledProcessError as e:
            print(
                f"[ERROR] Failed to clone the llama.cpp repository. Please check your internet connection and try again. Error: {e}")
            logger.error(f"Failed to clone the llama.cpp repository. Error: {e}")
            return

    if not os.path.exists(ready_file):
        try:
            print("[INFO] Running 'make' in the llama.cpp directory...")
            logger.info("Running 'make' in the llama.cpp directory...")
            subprocess.run(["make"], cwd=llama_cpp_dir, check=True)
            print("[INFO] Finished running 'make' in the llama.cpp directory")
            logger.info("Finished running 'make' in the llama.cpp directory")

            if not os.path.exists(venv_dir):
                print(f"[INFO] Creating virtual environment in '{venv_dir}'...")
                logger.info(f"Creating virtual environment in '{venv_dir}'...")
                create_venv(venv_dir)
                print("[INFO] Virtual environment created.")
                logger.info("Virtual environment created.")
            else:
                print(f"[INFO] Virtual environment already exists in '{venv_dir}'")
                logger.info(f"Virtual environment already exists in '{venv_dir}'")

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
            print(f"[ERROR] Failed to run 'make' or install dependencies in the llama.cpp directory. Error: {e}")
            logger.error(f"Failed to run 'make' or install dependencies in the llama.cpp directory. Error: {e}")
            return
        except Exception as e:
            print(f"[ERROR] An error occurred during the setup of llama.cpp. Error: {e}")
            logger.error(f"An error occurred during the setup of llama.cpp. Error: {e}")
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

            print(f"[INFO] Conversion complete. The model has been compressed and saved as '{output_file}'.")
            logger.info(f"Conversion complete. The model has been compressed and saved as '{output_file}'.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Conversion failed. Error: {e}")
            logger.error(f"Conversion failed. Error: {e}")
            return
        except Exception as e:
            print(f"[ERROR] An error occurred during the model conversion. Error: {e}")
            logger.error(f"An error occurred during the model conversion. Error: {e}")
            return
    else:
        print(f"[INFO] The model has already been converted and saved as '{output_file}'.")
        logger.info(f"The model has already been converted and saved as '{output_file}'.")

    if os.path.exists(output_file):
        try:
            print("[INFO] Quantizing the model...")
            logger.info("Quantizing the model...")

            quantize_command = [
                os.path.join(llama_cpp_dir, 'quantize'),
                output_file,
                quantized_output_file,
                "Q4_K_M"
            ]
            subprocess.run(quantize_command, check=True)

            print(f"[INFO] Quantization complete. The quantized model has been saved as '{quantized_output_file}'.")
            logger.info(f"Quantization complete. The quantized model has been saved as '{quantized_output_file}'.")
            print(f"[INFO] Deleting the original .bin file '{output_file}'...")
            logger.info(f"Deleting the original .bin file '{output_file}'...")
            os.remove(output_file)
            print(f"[INFO] Deleted the original .bin file '{output_file}'.")
            logger.info(f"Deleted the original .bin file '{output_file}'.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Quantization failed. Error: {e}")
            logger.error(f"Quantization failed. Error: {e}")
            return
        except Exception as e:
            print(f"[ERROR] An error occurred during the quantization process. Error: {e}")
            logger.error(f"An error occurred during the quantization process. Error: {e}")
            return
    else:
        print(f"[ERROR] The original model file '{output_file}' does not exist.")
        logger.error(f"The original model file '{output_file}' does not exist.")


def fpk_cli_handle_find(args, session):
    """Handle the 'find' command to search for model files."""
    logger.info("Searching for model files...")
    model_files = fpk_find_models()
    if model_files:
        print("[INFO] Found the following model files:")
        logger.info("Found the following model files:")
        for model_file in model_files:
            print(f" - {model_file}")
            logger.info(f" - {model_file}")
    else:
        print("[INFO] No model files found.")
        logger.info("No model files found.")


def fpk_cli_handle_get_api_key(args, session):
    """Handle the 'get' command to retrieve the API key."""
    logger.info("Retrieving API key...")
    api_key = fpk_get_api_key()
    if api_key:
        print(f"[INFO] API Key: {api_key}")
        logger.info(f"API Key: {api_key}")
    else:
        print("[ERROR] No API key found.")
        logger.error("No API key found.")


def fpk_cli_handle_help(args, session):
    """Handle the 'help' command to display the help message."""
    parser = setup_arg_parser()
    if args.command:
        # Print help for the specific command if provided
        subparser = parser._subparsers._actions[1].choices.get(args.command)
        if subparser:
            subparser.print_help()
            logger.info(f"Displayed help for command '{args.command}'.")
        else:
            print(f"[ERROR] Command '{args.command}' not found.")
            logger.error(f"Command '{args.command}' not found.")
    else:
        # Print general help if no specific command is provided
        parser.print_help()
        logger.info("Displayed general help.")


def fpk_cli_handle_list(args, session):
    """Handle the 'list' command to fetch and print the list of directories."""
    directories = fpk_list_directories(session)
    if directories:
        print(directories)
        logger.info(f"Directories found: {directories}")
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


# Apply the filter to the uvicorn access logger
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
    """Set the token in the configuration file."""
    try:
        config = load_config()
        config['token'] = token
        save_config(config)
        print("[INFO] Token set successfully!")
        logger.info("Token set successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to set token: {e}")
        logger.error(f"Failed to set token: {e}")


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
    if token is None or token != f"Bearer {get_token()}":
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
    if validate_api_token(api_token):
        return JSONResponse(content={"message": "API token is valid."}, status_code=200)
    else:
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

    # Ensure the requested file is within the flatpack directory to prevent directory traversal attacks
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

    # Ensure the requested file is within the flatpack directory to prevent directory traversal attacks
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
        fpk_build(flatpack_directory)
        return JSONResponse(content={"message": "Build process completed successfully."}, status_code=200)
    except Exception as e:
        logger.error(f"Build process failed: {e}")
        return JSONResponse(content={"message": f"Build process failed: {e}"}, status_code=500)


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
        logger.error(f"Verification process failed: {e}")
        return JSONResponse(content={"message": f"Verification process failed: {e}"}, status_code=500)


@app.get("/api/logs")
async def get_all_logs(request: Request, token: str = Depends(authenticate_token)):
    """Get a list of all available logs ordered by date."""
    cache_file_path = HOME_DIR / ".fpk_unbox.cache"
    if cache_file_path.exists():
        last_unboxed_flatpack = cache_file_path.read_text().strip()
        logs_directory = Path(last_unboxed_flatpack) / 'build' / 'logs'
    else:
        raise HTTPException(status_code=500, detail="No cached flatpack directory found")

    try:
        if not logs_directory.exists():
            raise HTTPException(status_code=404, detail="Logs directory not found")

        log_files = sorted(
            [f for f in os.listdir(logs_directory) if f.startswith("build_") and f.endswith(".log")],
            key=lambda x: datetime.strptime(x, "build_%Y_%m_%d_%H_%M_%S.log"),
            reverse=True
        )
        return JSONResponse(content={"logs": log_files}, status_code=200)
    except Exception as e:
        logger.error(f"Failed to list log files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list log files: {e}")


@app.get("/api/logs/{log_filename}")
async def get_log_file(request: Request, log_filename: str, token: str = Depends(authenticate_token)):
    """Get the content of a specific log file."""
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
            logger.error(f"Error reading log file '{log_filename}': {e}")
            raise HTTPException(status_code=500, detail=f"Error reading log file: {e}")
    else:
        raise HTTPException(status_code=404, detail="Log file not found")


@app.get("/api/heartbeat")
async def heartbeat():
    """Endpoint to check the server heartbeat."""
    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    return JSONResponse(content={"server_time": current_time}, status_code=200)


def setup_static_directory(app: FastAPI, directory: str):
    """Setup the static directory for serving static files."""
    global flatpack_directory
    flatpack_directory = os.path.abspath(directory)

    if os.path.exists(flatpack_directory) and os.path.isdir(flatpack_directory):
        static_dir = os.path.join(flatpack_directory, 'build')
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
        print(f"[INFO] Static files will be served from: {static_dir}")
        logger.info(f"Static files will be served from: {static_dir}")
    else:
        error_message = f"The directory '{flatpack_directory}' does not exist or is not a directory."
        print(f"[ERROR] {error_message}")
        logger.error(error_message)
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
        logger.info(f"Using provided flatpack: {directory}")
    else:
        print(f"[ERROR] The flatpack '{directory}' does not exist.")
        logger.error(f"The flatpack '{directory}' does not exist.")
        return

    if args.share:
        fpk_check_ngrok_auth()

    token = generate_secure_token()
    print(f"[INFO] Generated API token: {token}")
    logger.info(f"Generated API token: {token}")
    print("[INFO] Please save this API token securely. You will not be able to retrieve it again.")
    logger.info("Please save this API token securely. You will not be able to retrieve it again.")

    try:
        while True:
            confirmation = input("Have you saved the API token? Type 'YES' to continue: ").strip().upper()
            if confirmation == 'YES':
                break
            else:
                print("[INFO] Please save the API token before continuing.")
                logger.info("Please save the API token before continuing.")
    except KeyboardInterrupt:
        print("\n[ERROR] Process interrupted by user. Please save the API token and try again.")
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
            logger.info(f"Ingress established at {public_url}")

        # Set up the Uvicorn server instance for signal handling
        config = uvicorn.Config(app, host="127.0.0.1", port=port)
        global uvicorn_server
        uvicorn_server = uvicorn.Server(config)
        uvicorn_server.run()
    except KeyboardInterrupt:
        print("[INFO] FastAPI server has been stopped.")
        logger.info("FastAPI server has been stopped.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during server run: {e}")
        logger.error(f"An unexpected error occurred during server run: {e}")
    finally:
        if args.share:
            ngrok.disconnect(public_url)
            print(f"[INFO] Disconnected ngrok ingress at {public_url}")
            logger.info(f"Disconnected ngrok ingress at {public_url}")


def fpk_cli_handle_set_api_key(args, session):
    """Handle the 'set' command to set the API key.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    print(f"[INFO] Setting API key: {args.api_key}")
    logger.info(f"Setting API key: {args.api_key}")

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
        logger.error(f"Error during API key verification: {e}")


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
        logger.info(f"Agent spawned with PID: {pid}")
    except Exception as e:
        print(f"[ERROR] Failed to spawn agent: {e}")
        logger.error(f"Failed to spawn agent: {e}")


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
        logger.info(f"Agent with PID {args.pid} terminated successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to terminate agent with PID {args.pid}: {e}")
        logger.error(f"Failed to terminate agent with PID {args.pid}: {e}")


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
        logger.error(f"The flatpack '{directory_name}' does not exist.")
        return

    fpk_display_disclaimer(directory_name, local=args.local)

    while True:
        user_response = input().strip().upper()
        if user_response == "YES":
            break
        elif user_response == "NO":
            print("[INFO] Installation aborted by user.")
            logger.info("Installation aborted by user.")
            return
        else:
            print("[ERROR] Invalid input. Please type 'YES' to accept or 'NO' to decline.")
            logger.error("Invalid input from user. Expected 'YES' or 'NO'.")

    if args.local:
        local_directory_path = Path(directory_name)
        if not local_directory_path.exists() or not local_directory_path.is_dir():
            print(f"[ERROR] Local directory does not exist: '{directory_name}'.")
            logger.error(f"Local directory does not exist: '{directory_name}'.")
            return
        toml_path = local_directory_path / 'flatpack.toml'
        if not toml_path.exists():
            print(f"[ERROR] flatpack.toml not found in the specified directory: '{directory_name}'.")
            logger.error(f"flatpack.toml not found in the specified directory: '{directory_name}'.")
            return

    print(f"[INFO] Directory name resolved to: '{directory_name}'")
    logger.info(f"Directory name resolved to: '{directory_name}'")
    try:
        fpk_unbox(directory_name, session, local=args.local)
        print(f"[INFO] Unboxed flatpack '{directory_name}' successfully.")
        logger.info(f"Unboxed flatpack '{directory_name}' successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to unbox flatpack '{directory_name}': {e}")
        logger.error(f"Failed to unbox flatpack '{directory_name}': {e}")


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
    logger.info(f"Verifying flatpack in directory: {directory_name}")

    try:
        fpk_verify(directory_name)
        print(f"[INFO] Verification successful for directory: {directory_name}")
        logger.info(f"Verification successful for directory: {directory_name}")
    except Exception as e:
        print(f"[ERROR] Verification failed for directory '{directory_name}': {e}")
        logger.error(f"Verification failed for directory '{directory_name}': {e}")


def fpk_cli_handle_version(args, session):
    """Handle the 'version' command to display the version of flatpack.

    Args:
        args: The command-line arguments.
        session: The HTTP session.
    """
    print(f"[INFO] Flatpack version: {VERSION}")
    logger.info(f"Flatpack version: {VERSION}")


def fpk_initialize_vector_manager(args):
    """Initialize the Vector Manager.

    Args:
        args: The command-line arguments.

    Returns:
        VectorManager: An instance of VectorManager.
    """
    data_dir = getattr(args, 'data_dir', '.')
    print(f"[INFO] Initializing Vector Manager with model 'all-MiniLM-L6-v2' and data directory: {data_dir}")
    logger.info(f"Initializing Vector Manager with model 'all-MiniLM-L6-v2' and data directory: {data_dir}")
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
        logger.info(f"Added {len(args.texts)} texts to the database.")
    elif args.vector_command == 'search-text':
        results = vm.search_vectors(args.query)
        if results:
            print("[INFO] Search results:")
            logger.info("Search results:")
            for result in results:
                print(f"{result['id']}: {result['text']}\n")
                logger.info(f"{result['id']}: {result['text']}")
        else:
            print("[INFO] No results found.")
            logger.info("No results found.")
    elif args.vector_command == 'add-pdf':
        fpk_cli_handle_add_pdf(args.pdf_path, vm)
    elif args.vector_command == 'add-url':
        fpk_cli_handle_add_url(args.url, vm)
    elif args.vector_command == 'add-wikipedia':
        vm.add_wikipedia_page(args.page_title)
        print(f"[INFO] Added text from Wikipedia page: '{args.page_title}' to the vector database.")
        logger.info(f"Added text from Wikipedia page: '{args.page_title}' to the vector database.")
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

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
