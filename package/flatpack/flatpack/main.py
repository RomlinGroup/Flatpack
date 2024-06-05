import argparse
import atexit
import httpx
import ngrok
import os
import re
import requests
import shlex
import shutil
import stat
import subprocess
import sys
import toml
import uvicorn

from .agent_manager import AgentManager
from datetime import datetime
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import snapshot_download
from importlib.metadata import version
from io import BytesIO
from .parsers import parse_toml_to_venv_script
from pathlib import Path
from pydantic import BaseModel
from .session_manager import SessionManager
from typing import List, Optional, Union
from .vector_manager import VectorManager
from zipfile import ZipFile

HOME_DIR = Path.home()

BASE_URL = "https://raw.githubusercontent.com/romlingroup/flatpack/main/warehouse"
CONFIG_FILE_PATH = HOME_DIR / ".fpk_config.toml"
GITHUB_REPO_URL = "https://api.github.com/repos/romlingroup/flatpack"
KEY_FILE_PATH = HOME_DIR / ".fpk_encryption_key"
VERSION = version("flatpack")

config = {
    "api_key": None
}


def fpk_build(directory: Union[str, None]):
    """Build a flatpack."""
    cache_file_path = HOME_DIR / ".fpk_unbox.cache"
    print(f"Looking for cached flatpack in {cache_file_path}.")

    last_unboxed_flatpack = None

    if directory and fpk_valid_directory_name(directory):
        print(f"Using provided directory: {directory}")
        last_unboxed_flatpack = directory
    elif cache_file_path.exists():
        print(f"Found cached flatpack in {cache_file_path}.")
        last_unboxed_flatpack = cache_file_path.read_text().strip()
    else:
        print("❌ No cached flatpack found, and no valid directory provided.")
        return

    if not last_unboxed_flatpack:
        print("❌ No valid flatpack directory found.")
        return

    building_script_path = Path(last_unboxed_flatpack) / 'build' / 'build.sh'

    if not building_script_path.exists() or not building_script_path.is_file():
        print(f"❌ Building script not found in {last_unboxed_flatpack}.")
        return

    safe_script_path = shlex.quote(str(building_script_path.resolve()))

    try:
        result = subprocess.run(['bash', '-u', safe_script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ An error occurred while executing the build script: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


def fpk_cache_unbox(directory_name: str):
    """Cache the last unboxed flatpack's directory name to a file within the corresponding build directory."""
    cache_file_path = HOME_DIR / ".fpk_unbox.cache"
    with open(cache_file_path, 'w') as f:
        f.write(directory_name)


def fpk_check_ngrok_auth():
    """Check if the NGROK_AUTHTOKEN environment variable is set."""
    ngrok_auth_token = os.environ.get('NGROK_AUTHTOKEN')
    if not ngrok_auth_token:
        print("❌ Error: NGROK_AUTHTOKEN is not set. Please set it using:")
        print("export NGROK_AUTHTOKEN='your_ngrok_auth_token'")
        sys.exit(1)
    else:
        print("NGROK_AUTHTOKEN is set.")


def fpk_colorize(text, color):
    """Colorize a given text with the specified color.

    Args:
        text (str): The text to be colorized.
        color (str): The color to apply to the text.

    Returns:
        str: Colorized text.
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
    return colors[color] + text + colors["default"]


def fpk_create(flatpack_name, repo_url="https://github.com/RomlinGroup/template"):
    if not re.match(r'^[a-z0-9-]+$', flatpack_name):
        raise ValueError("Error: Invalid name format. Only lowercase letters, numbers, and hyphens are allowed.")

    flatpack_name = flatpack_name.lower().replace(' ', '-')

    current_dir = os.getcwd()
    template_dir = fpk_download_and_extract_template(repo_url, current_dir)

    flatpack_dir = os.path.join(current_dir, flatpack_name)
    os.makedirs(flatpack_dir, exist_ok=True)

    for item in os.listdir(template_dir):
        s = os.path.join(template_dir, item)
        d = os.path.join(flatpack_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

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

    shutil.rmtree(template_dir)

    print(f"✨ Successfully created {flatpack_name}.")


def fpk_display_disclaimer(directory_name: str, local: bool):
    """Display a disclaimer message with details about a specific flatpack.

    Args:
        directory_name (str): Name of the flatpack directory.
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
        please_note_content = """
PLEASE NOTE: The flatpack you are about to unbox is
governed by its own licenses and terms, separate from
this software. You may find further details at:

https://fpk.ai/w/{}
            """.format(directory_name)
        please_note_colored = fpk_colorize(please_note_content, "yellow")
    else:
        please_note_colored = ""

    print(disclaimer_template.format(please_note=please_note_colored))


def fpk_download_and_extract_template(repo_url, dest_dir):
    response = requests.get(f"{repo_url}/archive/refs/heads/main.zip")
    with ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(dest_dir)
    return os.path.join(dest_dir, "template-main")


def fpk_fetch_flatpack_toml_from_dir(directory_name: str, session: httpx.Client) -> Optional[str]:
    """Fetch the flatpack TOML configuration from a specific directory.
    Args:
        directory_name (str): Name of the flatpack directory.
        session (httpx.Client): HTTP client session for making requests.
    Returns:
        Optional[str]: The TOML content if found, otherwise None.
    """
    if not fpk_valid_directory_name(directory_name):
        print(f"❌ Invalid directory name: '{directory_name}'.")
        return None

    toml_url = f"{BASE_URL}/{directory_name}/flatpack.toml"
    try:
        response = session.get(toml_url)
        response.raise_for_status()
        return response.text
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP error occurred: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.RequestError as e:
        print(f"❌ Network error occurred: {e}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
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
            return sorted(directories)
        else:
            print(f"❌ Unexpected response format from GitHub: {json_data}")
            return []
    except httpx.HTTPError as e:
        print(f"❌ Unable to connect to GitHub: {e}")
        sys.exit(1)
    except (ValueError, KeyError) as e:
        print(f"❌ Error processing the response from GitHub: {e}")
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
    model_file_formats = ['.caffemodel', '.ckpt', '.gguf', '.h5', '.json', '.mar', '.mlmodel', '.model', '.onnx',
                          '.params', '.pb', '.pkl', '.pickle', '.pt', '.pth', '.sav', '.tflite', '.weights']
    model_files = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.endswith(fmt) for fmt in model_file_formats):
                model_files.append(os.path.join(root, file))
    return model_files


def fpk_get_api_key() -> Optional[str]:
    """Retrieve the API key from the configuration file."""
    if not os.path.exists(CONFIG_FILE_PATH):
        return None

    try:
        with open(CONFIG_FILE_PATH, 'r') as config_file:
            loaded_config = toml.load(config_file)
        api_key = loaded_config.get('api_key')
        return api_key
    except Exception as e:
        print(f"Error retrieving API key: {e}")

    return None


def fpk_get_last_flatpack(directory_name: str) -> Optional[str]:
    """Retrieve the last unboxed flatpack's directory name from the cache file within the correct build directory."""
    cache_file_path = HOME_DIR / ".fpk_unbox.cache"
    try:
        if cache_file_path.exists():
            with cache_file_path.open('r') as cache_file:
                return cache_file.read().strip()
    except (OSError, IOError) as e:
        print(f"❌ An error occurred while accessing the cache file: {e}")
    return None


def fpk_is_raspberry_pi():
    """Check if we're running on a Raspberry Pi."""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('Hardware') and 'BCM' in line:
                    return True
    except IOError:
        return False
    return False


def fpk_list_directories(session: httpx.Client) -> str:
    """Fetch a list of directories from GitHub and return as a newline-separated string.

    Parameters:
    - session (httpx.Client): HTTP client session for making requests.

    Returns:
    - str: A newline-separated string of directory names.
    """
    dirs = fpk_fetch_github_dirs(session)
    return "\n".join(dirs)


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


def fpk_set_api_key(api_key: str):
    """Set the API key in the configuration file."""
    global config

    config = {'api_key': api_key}

    with open(CONFIG_FILE_PATH, "w") as config_file:
        toml.dump(config, config_file)
    os.chmod(CONFIG_FILE_PATH, 0o600)
    print("API key set successfully!")

    try:
        test_key = fpk_get_api_key()
        if test_key == api_key:
            print("Verification successful: API key matches.")
        else:
            print("Verification failed: API key does not match.")
    except Exception as e:
        print(f"Error during API key verification: {e}")


def fpk_set_secure_file_permissions(file_path):
    """Set secure file permissions for the specified file."""
    os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)


def fpk_unbox(directory_name: str, session, local: bool = False):
    """Unbox a flatpack from GitHub or a local directory."""
    if not fpk_valid_directory_name(directory_name):
        print(f"❌ Invalid directory name: '{directory_name}'.")
        return

    flatpack_dir = Path.cwd() / directory_name

    if flatpack_dir.exists() and not local:
        print("❌ Error: Flatpack directory already exists.")
        return
    else:
        build_dir = flatpack_dir / "build"

        if build_dir.exists():
            print("❌ Error: Build directory already exists.")
            return

    flatpack_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    temp_toml_path = build_dir / 'temp_flatpack.toml'
    if local:
        toml_path = flatpack_dir / 'flatpack.toml'
        if not toml_path.exists():
            print(f"❌ flatpack.toml not found in the specified directory: '{directory_name}'.")
            return
        toml_content = toml_path.read_text()
    else:
        toml_content = fpk_fetch_flatpack_toml_from_dir(directory_name, session)
        if not toml_content:
            print(f"❌ Error: Failed to fetch TOML content for '{directory_name}'.")
            return

    temp_toml_path.write_text(toml_content)
    bash_script_content = parse_toml_to_venv_script(str(temp_toml_path), '3.11.8', directory_name)
    bash_script_path = build_dir / 'flatpack.sh'
    bash_script_path.write_text(bash_script_content)
    temp_toml_path.unlink()

    print(f"📦 Unboxing {directory_name}...")
    safe_script_path = shlex.quote(str(bash_script_path.resolve()))
    try:
        result = subprocess.run(['bash', safe_script_path], check=True)
        print("🎉 All done!")
        fpk_cache_unbox(str(flatpack_dir))
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: Failed to execute the bash script: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


def fpk_valid_directory_name(name: str) -> bool:
    """
    Validate that the directory name contains only alphanumeric characters, dashes, and underscores.

    Args:
        name (str): The directory name to validate.

    Returns:
        bool: True if the name is valid, False otherwise.
    """
    return re.match(r'^[\w-]+$', name) is not None


atexit.register(fpk_safe_cleanup)


def fpk_verify(directory: Union[str, None]):
    """Verify a flatpack."""
    cache_file_path = HOME_DIR / ".fpk_unbox.cache"
    print(f"Looking for cached flatpack in {cache_file_path}.")

    last_unboxed_flatpack = None

    if directory and fpk_valid_directory_name(directory):
        print(f"Using provided directory: {directory}")
        last_unboxed_flatpack = directory
    elif cache_file_path.exists():
        print(f"Found cached flatpack in {cache_file_path}.")
        last_unboxed_flatpack = cache_file_path.read_text().strip()
    else:
        print("❌ No cached flatpack found, and no valid directory provided.")
        return

    if not last_unboxed_flatpack:
        print("❌ No valid flatpack directory found.")
        return

    verification_script_path = Path(last_unboxed_flatpack) / 'build' / 'build.sh'

    if not verification_script_path.exists() or not verification_script_path.is_file():
        print(f"❌ Verification script not found in {last_unboxed_flatpack}.")
        return

    safe_script_path = shlex.quote(str(verification_script_path.resolve()))

    try:
        env_vars = {'VERIFY_MODE': 'true'}
        result = subprocess.run(['bash', '-u', safe_script_path], check=True, env={**env_vars, **os.environ})
        print("✅ Verification script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ An error occurred while executing the verification script: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")


def setup_arg_parser():
    # Create the top-level parser
    parser = argparse.ArgumentParser(description='Flatpack command line interface')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # General commands
    parser_list = subparsers.add_parser('list', help='List available flatpack directories.')
    parser_list.set_defaults(func=lambda args, session: fpk_cli_handle_list(args, session))

    parser_find = subparsers.add_parser('find', help='Find model files in the current directory.')
    parser_find.set_defaults(func=lambda args, session: fpk_cli_handle_find(args, session))

    parser_help = subparsers.add_parser('help', help='Display help for commands.')
    parser_help.set_defaults(func=lambda args, session: fpk_cli_handle_help(args, session))

    parser_version = subparsers.add_parser('version', help='Display the version of flatpack.')
    parser_version.set_defaults(func=lambda args, session: fpk_cli_handle_version(args, session))

    # API Key management
    parser_api_key = subparsers.add_parser('api-key', help='API key management commands')
    api_key_subparsers = parser_api_key.add_subparsers(dest='api_key_command')

    # Set API key
    parser_set_api = api_key_subparsers.add_parser('set', help='Set the API key')
    parser_set_api.add_argument('api_key', type=str, help='API key to set')
    parser_set_api.set_defaults(func=lambda args, session: fpk_cli_handle_set_api_key(args, session))

    # Get API key
    parser_get_api = api_key_subparsers.add_parser('get', help='Get the current API key')
    parser_get_api.set_defaults(func=lambda args, session: fpk_cli_handle_get_api_key(args, session))

    # Create flatpack
    parser_create = subparsers.add_parser('create', help='Create a new flatpack.')
    parser_create.add_argument('input', nargs='?', default=None, help='The name of the flatpack to create.')
    parser_create.set_defaults(func=lambda args, session: fpk_cli_handle_create(args, session))

    # Unbox commands
    parser_unbox = subparsers.add_parser('unbox', help='Unbox a flatpack from GitHub or a local directory.')
    parser_unbox.add_argument('input', nargs='?', default=None, help='The name of the flatpack to unbox.')
    parser_unbox.add_argument('--local', action='store_true', help='Unbox from a local directory instead of GitHub.')
    parser_unbox.set_defaults(func=lambda args, session: fpk_cli_handle_unbox(args, session))

    # Build commands
    parser_build = subparsers.add_parser('build', help='Build a flatpack.')
    parser_build.add_argument('directory', nargs='?', default=None, help='The directory of the flatpack to build.')
    parser_build.set_defaults(func=lambda args, session: fpk_cli_handle_build(args, session))

    # Verify commands
    parser_build = subparsers.add_parser('verify', help='Verify a flatpack.')
    parser_build.add_argument('directory', nargs='?', default=None, help='The directory of the flatpack to verify.')
    parser_build.set_defaults(func=lambda args, session: fpk_cli_handle_verify(args, session))

    # Run server
    parser_run = subparsers.add_parser('run', help='Run the FastAPI server.')
    parser_run.add_argument('input', nargs='?', default=None, help='The name of the flatpack to run.')
    parser_run.add_argument('--share', action='store_true', help='Share using ngrok.')
    parser_run.set_defaults(func=lambda args, session: fpk_cli_handle_run(args, session))

    # Vector database management
    parser_vector = subparsers.add_parser('vector', help='Vector database management')
    vector_subparsers = parser_vector.add_subparsers(dest='vector_command')

    parser_add_text = vector_subparsers.add_parser('add-texts',
                                                   help='Add new texts to generate embeddings and store them.')
    parser_add_text.add_argument('texts', nargs='+', help='Texts to add.')
    parser_add_text.add_argument('--data-dir', type=str, default='.',
                                 help='Directory path for storing the vector database and metadata files.')
    parser_add_text.set_defaults(func=lambda args, session, vm: fpk_cli_handle_vector_commands(args, session, vm))

    parser_search_text = vector_subparsers.add_parser('search-text',
                                                      help='Search for texts similar to the given query.')
    parser_search_text.add_argument('query', help='Text query to search for.')
    parser_search_text.add_argument('--data-dir', type=str, default='.',
                                    help='Directory path for storing the vector database and metadata files.')
    parser_search_text.set_defaults(func=lambda args, session, vm: fpk_cli_handle_vector_commands(args, session, vm))

    parser_add_pdf = vector_subparsers.add_parser('add-pdf', help='Add text from a PDF file to the vector database.')
    parser_add_pdf.add_argument('pdf_path', help='Path to the PDF file to add.')
    parser_add_pdf.add_argument('--data-dir', type=str, default='.',
                                help='Directory path for storing the vector database and metadata files.')
    parser_add_pdf.set_defaults(func=lambda args, session, vm: fpk_cli_handle_vector_commands(args, session, vm))

    parser_add_url = vector_subparsers.add_parser('add-url', help='Add text from a URL to the vector database.')
    parser_add_url.add_argument('url', help='URL to add.')
    parser_add_url.add_argument('--data-dir', type=str, default='.',
                                help='Directory path for storing the vector database and metadata files.')
    parser_add_url.set_defaults(func=lambda args, session, vm: fpk_cli_handle_vector_commands(args, session, vm))

    parser_add_wikipedia_page = vector_subparsers.add_parser('add-wikipedia',
                                                             help='Add text from a Wikipedia page to the vector database.')
    parser_add_wikipedia_page.add_argument('page_title', help='The title of the Wikipedia page to add.')
    parser_add_wikipedia_page.add_argument('--data-dir', type=str, default='.',
                                           help='Directory path for storing the vector database and metadata files.')
    parser_add_wikipedia_page.set_defaults(
        func=lambda args, session, vm: fpk_cli_handle_vector_commands(args, session, vm))

    # Add commands for agents
    parser_agents = subparsers.add_parser('agents', help='Manage agents')
    agent_subparsers = parser_agents.add_subparsers(dest='agent_command', help='Available agent commands')

    # Add command to spawn an agent
    parser_spawn = agent_subparsers.add_parser('spawn', help='Spawn a new agent with a script')
    parser_spawn.add_argument('script_path', type=str, help='Path to the script to execute')
    parser_spawn.set_defaults(func=lambda args, session: fpk_cli_handle_spawn_agent(args, session))

    # Add command to list active agents
    parser_list = agent_subparsers.add_parser('list', help='List active agents')
    parser_list.set_defaults(func=lambda args, session: fpk_cli_handle_list_agents(args, session))

    # Add command to terminate an agent
    parser_terminate = agent_subparsers.add_parser('terminate', help='Terminate an active agent')
    parser_terminate.add_argument('pid', type=int, help='Process ID of the agent to terminate')
    parser_terminate.set_defaults(func=lambda args, session: fpk_cli_handle_terminate_agent(args, session))

    # Model compression
    parser_compress = subparsers.add_parser('compress', help='Compress a model for deployment.')
    parser_compress.add_argument('model_id', type=str,
                                 help='The name of the Hugging Face repository (format: username/repo_name).')
    parser_compress.add_argument('--token', type=str, help='Hugging Face token for private repositories.', default=None)
    parser_compress.set_defaults(func=lambda args, session: fpk_cli_handle_compress(args, session))

    return parser


def fpk_cli_handle_add_pdf(pdf_path, vm):
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file does not exist: '{pdf_path}'.")
        return
    vm.add_pdf(pdf_path, pdf_path)
    print(f"✅ Added text from PDF: '{pdf_path}' to the vector database.")


def fpk_cli_handle_add_url(url, vm):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if response.status_code >= 200 and response.status_code < 400:
            vm.add_url(url)
            print(f"✅ Added text from URL: '{url}' to the vector database.")
        else:
            print(f"❌ URL is not accessible: '{url}'. HTTP Status Code: {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Failed to access URL: '{url}'. Error: {e}")


def fpk_cli_handle_build(args, session):
    directory_name = args.directory
    fpk_build(directory_name)


def fpk_cli_handle_create(args, session):
    if not args.input:
        print("❌ Please specify a name for the new flatpack.")
        return

    flatpack_name = args.input
    fpk_create(flatpack_name)


def create_venv(venv_dir):
    subprocess.run(["python3", "-m", "venv", venv_dir], check=True)


def fpk_cli_handle_compress(args, session: httpx.Client):
    model_id = args.model_id
    token = args.token

    if not re.match(r'^[\w-]+/[\w.-]+$', model_id):
        print("❌ Please specify a valid Hugging Face repository in the format 'username/repo_name'.")
        return

    repo_name = model_id.split('/')[-1]
    local_dir = repo_name

    if os.path.exists(local_dir):
        print(f"📂 The model '{model_id}' is already downloaded in the directory '{local_dir}'.")
    else:
        try:
            if token:
                print(f"📥 Downloading model '{model_id}' with provided token...")
                snapshot_download(
                    repo_id=model_id, local_dir=local_dir, revision="main", token=token
                )
            else:
                print(f"📥 Downloading model '{model_id}'...")
                snapshot_download(repo_id=model_id, local_dir=local_dir, revision="main")
            print(f"🤗 Finished downloading {model_id} into the directory '{local_dir}'")
        except Exception as e:
            print(f"❌ Failed to download the model. Please check your internet connection and try again. Error: {e}")
            return

    llama_cpp_dir = "llama.cpp"
    ready_file = os.path.join(llama_cpp_dir, "ready")
    requirements_file = os.path.join(llama_cpp_dir, "requirements.txt")
    venv_dir = os.path.join(llama_cpp_dir, "venv")
    venv_python = os.path.join(venv_dir, "bin", "python")

    if not os.path.exists(llama_cpp_dir):
        try:
            print("📥 Cloning llama.cpp repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/ggerganov/llama.cpp", llama_cpp_dir],
                check=True
            )
            print(f"📥 Finished cloning llama.cpp repository into '{llama_cpp_dir}'")
        except subprocess.CalledProcessError as e:
            print(
                f"❌ Failed to clone the llama.cpp repository. Please check your internet connection and try again. Error: {e}"
            )
            return

    if not os.path.exists(ready_file):
        try:
            print("🔨 Running 'make' in the llama.cpp directory...")
            subprocess.run(["make"], cwd=llama_cpp_dir, check=True)
            print("🔨 Finished running 'make' in the llama.cpp directory")

            if not os.path.exists(venv_dir):
                print(f"🐍 Creating virtual environment in '{venv_dir}'...")
                create_venv(venv_dir)
                print("🐍 Virtual environment created.")
            else:
                print(f"📂 Virtual environment already exists in '{venv_dir}'")

            print("📦 Installing llama.cpp dependencies in virtual environment...")

            pip_command = [
                "/bin/bash", "-c",
                (
                    f"source {shlex.quote(os.path.join(venv_dir, 'bin', 'activate'))} && "
                    f"pip install -r {shlex.quote(requirements_file)}"
                )
            ]
            subprocess.run(pip_command, check=True)

            print("📦 Finished installing llama.cpp dependencies")

            with open(ready_file, 'w') as f:
                f.write("Ready")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to run 'make' or install dependencies in the llama.cpp directory. Error: {e}")
            return
        except Exception as e:
            print(f"❌ An error occurred during the setup of llama.cpp. Error: {e}")
            return

    output_file = os.path.join(local_dir, f"{repo_name}-fp16.bin")
    quantized_output_file = os.path.join(local_dir, f"{repo_name}-Q4_K_M.gguf")
    outtype = "f16"

    if not os.path.exists(output_file):
        try:
            print("🛠 Converting the model using llama.cpp...")

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

            print(f"✅ Conversion complete. The model has been compressed and saved as '{output_file}'.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Conversion failed. Error: {e}")
            return
        except Exception as e:
            print(f"❌ An error occurred during the model conversion. Error: {e}")
            return
    else:
        print(f"📂 The model has already been converted and saved as '{output_file}'.")

    if os.path.exists(output_file):
        try:
            print("🛠 Quantizing the model...")

            quantize_command = [
                os.path.join(llama_cpp_dir, 'quantize'),
                output_file,
                quantized_output_file,
                "Q4_K_M"
            ]
            subprocess.run(quantize_command, check=True)

            print(f"✅ Quantization complete. The quantized model has been saved as '{quantized_output_file}'.")
            print(f"🗑 Deleting the original .bin file '{output_file}'...")
            os.remove(output_file)
            print(f"🗑 Deleted the original .bin file '{output_file}'.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Quantization failed. Error: {e}")
            return
        except Exception as e:
            print(f"❌ An error occurred during the quantization process. Error: {e}")
            return
    else:
        print(f"❌ The original model file '{output_file}' does not exist.")


def fpk_cli_handle_find(args, session):
    print(fpk_find_models())


def fpk_cli_handle_get_api_key(args, session):
    print("Retrieving API key...")
    print(fpk_get_api_key())


def fpk_cli_handle_help(args, session):
    parser = setup_arg_parser()
    parser.print_help()


def fpk_cli_handle_list(args, session):
    print(fpk_list_directories(session))


def fpk_cli_handle_list_agents(args, session):
    """List active agents."""
    agent_manager = AgentManager()
    agent_manager.list_agents()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

flatpack_directory = None


def escape_special_chars(content: str) -> str:
    return content.replace('"', '\\"')


def unescape_special_chars(content: str) -> str:
    return content.replace('\\"', '"')


def escape_content_parts(content: str) -> str:
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


@app.get("/load_file")
async def load_file(filename: str):
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    file_path = os.path.join(flatpack_directory, 'build', filename)
    if os.path.commonpath([flatpack_directory, file_path]) != flatpack_directory:
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
async def save_file(filename: str = Form(...), content: str = Form(...)):
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")

    file_path = os.path.join(flatpack_directory, 'build', filename)
    if os.path.commonpath([flatpack_directory, file_path]) != flatpack_directory:
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
async def build_flatpack():
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")
    fpk_build(flatpack_directory)


@app.post("/api/verify")
async def verify_flatpack():
    if not flatpack_directory:
        raise HTTPException(status_code=500, detail="Flatpack directory is not set")
    fpk_verify(flatpack_directory)


@app.get("/api/heartbeat")
async def heartbeat():
    current_time = datetime.utcnow().isoformat()
    return JSONResponse(content={"server_time": current_time})


def setup_static_directory(app, directory: str):
    global flatpack_directory
    flatpack_directory = os.path.abspath(directory)
    if os.path.exists(flatpack_directory) and os.path.isdir(flatpack_directory):
        app.mount("/", StaticFiles(directory=f"{flatpack_directory}/build", html=True), name="static")
        print(f"Static files will be served from: {flatpack_directory}/build")
    else:
        print(f"The directory '{flatpack_directory}' does not exist or is not a directory.")


def fpk_cli_handle_run(args, session):
    if not args.input:
        print("❌ Please specify a flatpack for the run command.")
        return

    directory = args.input

    if os.path.exists(directory) and os.path.isdir(directory):
        print(f"Using provided flatpack: {directory}")
    else:
        print(f"❌ The flatpack '{directory}' does not exist.")
        return

    if args.share:
        fpk_check_ngrok_auth()

    setup_static_directory(app, directory)

    try:
        port = 8000

        if args.share:
            listener = ngrok.forward(port, authtoken_from_env=True)
            public_url = listener.url()
            print(f"Ingress established at {public_url}")

        uvicorn.run(app, host="127.0.0.1", port=port)
    except KeyboardInterrupt:
        print("❌ FastAPI server has been stopped.")
    except Exception as e:
        print(f"❌ An unexpected error occurred during server run: {e}")
    finally:
        if args.share:
            ngrok.disconnect(public_url)


def fpk_cli_handle_set_api_key(args, session):
    print(f"Setting API key: {args.api_key}")
    global config
    api_key = args.api_key

    config = {'api_key': api_key}

    with open(CONFIG_FILE_PATH, "w") as config_file:
        toml.dump(config, config_file)
    os.chmod(CONFIG_FILE_PATH, 0o600)
    print("API key set successfully!")


def fpk_cli_handle_spawn_agent(args, session):
    """Spawn a new agent with a script."""
    agent_manager = AgentManager()
    pid = agent_manager.spawn_agent(args.script_path)
    print(f"Agent spawned with PID: {pid}")


def fpk_cli_handle_terminate_agent(args, session):
    """Terminate an active agent."""
    agent_manager = AgentManager()
    agent_manager.terminate_agent(args.pid)


def fpk_cli_handle_unbox(args, session):
    if not args.input:
        print("❌ Please specify a flatpack for the unbox command.")
        return

    directory_name = args.input
    existing_dirs = fpk_fetch_github_dirs(session)

    if directory_name not in existing_dirs and not args.local:
        print(f"❌ The flatpack '{directory_name}' does not exist.")
        return

    if args.local:
        fpk_display_disclaimer(directory_name, local=True)
    else:
        fpk_display_disclaimer(directory_name, local=False)

    while True:
        user_response = input().strip().upper()
        if user_response == "YES":
            break
        elif user_response == "NO":
            print("❌ Installation aborted by user.")
            return
        else:
            print("❌ Invalid input. Please type 'YES' to accept or 'NO' to decline.")

    if args.local:
        local_directory_path = Path(directory_name)
        if not local_directory_path.exists() or not local_directory_path.is_dir():
            print(f"❌ Local directory does not exist: '{directory_name}'.")
            return
        toml_path = local_directory_path / 'flatpack.toml'
        if not toml_path.exists():
            print(f"❌ flatpack.toml not found in the specified directory: '{directory_name}'.")
            return

    print(f"✅ Directory name resolved to: '{directory_name}'")
    fpk_unbox(directory_name, session, local=args.local)


def fpk_cli_handle_verify(args, session):
    directory_name = args.directory
    fpk_verify(directory_name)


def fpk_cli_handle_version(args, session):
    print(VERSION)


def fpk_initialize_vector_manager(args):
    return VectorManager(model_id='all-MiniLM-L6-v2', directory=getattr(args, 'data_dir', '.'))


def fpk_cli_handle_vector_commands(args, session, vm):
    print("Handling vector commands...")

    if args.vector_command == 'add-texts':
        vm.add_texts(args.texts, "manual")
        print(f"Added {len(args.texts)} texts to the database.")
    elif args.vector_command == 'search-text':
        results = vm.search_vectors(args.query)
        if results:
            print("Search results:")
            for result in results:
                print(f"{result['id']}: {result['text']}\n")
        else:
            print("No results found.")
    elif args.vector_command == 'add-pdf':
        fpk_cli_handle_add_pdf(args.pdf_path, vm)
    elif args.vector_command == 'add-url':
        fpk_cli_handle_add_url(args.url, vm)
    elif args.vector_command == 'add-wikipedia':
        vm.add_wikipedia_page(args.page_title)
        print(f"✅ Added text from Wikipedia page: '{args.page_title}' to the vector database.")


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
        print(f"❌ An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
