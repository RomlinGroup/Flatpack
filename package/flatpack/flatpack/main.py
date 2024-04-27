import argparse
import atexit
import faiss
import httpx
import ngrok
import os
import re
import requests
import select
import shlex
import signal
import sys
import toml
import uvicorn

from .agent_manager import AgentManager
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from importlib.metadata import version
from .parsers import parse_toml_to_venv_script
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .session_manager import SessionManager
from typing import List, Optional
from .vector_manager import VectorManager

CONFIG_FILE_PATH = os.path.join(os.path.expanduser("~"), ".fpk_config.toml")
KEY_FILE_PATH = os.path.join(os.path.expanduser("~"), ".fpk_encryption_key")
GITHUB_REPO_URL = "https://api.github.com/repos/romlingroup/flatpack"
BASE_URL = "https://raw.githubusercontent.com/romlingroup/flatpack/main/warehouse"
VERSION = version("flatpack")

config = {
    "api_key": None
}


class FPKEncryptionKeyError(Exception):
    """Custom exception for missing encryption key."""
    pass


def fpk_build(directory: str):
    """Build a model using a building script from the last unboxed flatpack."""
    cache_file_path = Path('last_flatpack.cache')
    print(f"Looking for cached flatpack in {cache_file_path}.")

    if directory and fpk_valid_directory_name(directory):
        print(f"Using provided directory: {directory}")
        last_unboxed_flatpack = directory
    elif cache_file_path.exists():
        print(f"Found cached flatpack in {cache_file_path}.")
        last_unboxed_flatpack = cache_file_path.read_text().strip()
        if not fpk_valid_directory_name(last_unboxed_flatpack):
            print(f"❌ Invalid directory name from cache: '{last_unboxed_flatpack}'.")
            return
    else:
        print("❌ No cached flatpack found, and no valid directory provided.")
        return

    building_script_path = Path(last_unboxed_flatpack) / 'build' / 'build.sh'
    if not building_script_path.exists():
        print(f"❌ Building script not found in {last_unboxed_flatpack}.")
        return

    # Ensure the script path is safe to use in the shell
    safe_script_path = shlex.quote(str(building_script_path))

    # Running the build script directly
    result = os.system(f"bash -u {safe_script_path}")
    if result != 0:
        print("❌ An error occurred while executing the build script.")
    else:
        print("✅ Build script executed successfully.")


def fpk_cache_last_flatpack(directory_name: str):
    """Cache the last unboxed flatpack's directory name to a file within the corresponding build directory."""
    flatpack_dir = Path.cwd()
    flatpack_dir.mkdir(parents=True, exist_ok=True)

    cache_file_path = flatpack_dir / 'last_flatpack.cache'
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


def fpk_decrypt_data(encrypted_data: str, key: bytes) -> str:
    """Decrypt data using the provided key."""
    fernet = Fernet(key)
    return fernet.decrypt(encrypted_data.encode()).decode()


def fpk_display_disclaimer(directory_name: str):
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

    please_note_content = """
PLEASE NOTE: The flatpack you are about to unbox is
governed by its own licenses and terms, separate from
this software. You may find further details at:

https://fpk.ai/w/{}
""".format(directory_name)

    please_note_colored = fpk_colorize(please_note_content, "yellow")
    print(disclaimer_template.format(please_note=please_note_colored))


def fpk_encrypt_data(data: str, key: bytes) -> str:
    """Encrypt data using the provided key."""
    fernet = Fernet(key)
    return fernet.encrypt(data.encode()).decode()


def fpk_fetch_flatpack_toml_from_dir(directory_name: str, session: httpx.Client) -> Optional[str]:
    """Fetch the flatpack TOML configuration from a specific directory.

    Args:
        directory_name (str): Name of the flatpack directory.
        session (httpx.Client): HTTP client session for making requests.

    Returns:
        Optional[str]: The TOML content if found, otherwise None.
    """
    toml_url = f"{BASE_URL}/{directory_name}/flatpack.toml"
    response = session.get(toml_url)
    if response.status_code != 200:
        return None
    return response.text


def fpk_fetch_github_dirs(session: httpx.Client) -> List[str]:
    """Fetch a list of directory names from the GitHub repository.

    Args:
        session (httpx.Client): HTTP client session for making requests.

    Returns:
        List[str]: List of directory names.
    """
    response = session.get(GITHUB_REPO_URL + "/contents/warehouse")
    if response.status_code != 200:
        return ["❌ Error fetching data from GitHub"]
    directories = [item['name'] for item in response.json() if
                   item['type'] == 'dir' and item['name'].lower() != 'legacy' and item['name'].lower() != 'template']
    return sorted(directories)


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
    """Retrieve and decrypt the API key from the configuration file."""
    if not os.path.exists(CONFIG_FILE_PATH):
        return None  # Avoid unnecessary prints when the config file doesn't exist

    try:
        with open(CONFIG_FILE_PATH, 'r') as config_file:
            loaded_config = toml.load(config_file)
        encrypted_api_key = loaded_config.get('api_key')

        if encrypted_api_key:
            encryption_key = fpk_get_or_create_encryption_key()
            return fpk_decrypt_data(encrypted_api_key, encryption_key)
    except Exception as e:
        print(f"Error decrypting API key: {e}")

    return None


def fpk_get_last_flatpack(directory_name: str) -> Optional[str]:
    """Retrieve the last unboxed flatpack's directory name from the cache file within the correct build directory."""
    flatpack_dir = Path.cwd()
    cache_file_path = flatpack_dir / 'last_flatpack.cache'
    if cache_file_path.exists():
        return cache_file_path.read_text().strip()
    return None


def fpk_get_or_create_encryption_key() -> bytes:
    """Retrieve or generate and save an encryption key."""
    try:
        with open(KEY_FILE_PATH, "rb") as key_file:
            key = key_file.read()
    except FileNotFoundError:
        print("No encryption key found. Generating a new key for persistent use.")
        key = Fernet.generate_key()
        with open(KEY_FILE_PATH, "wb") as key_file:
            key_file.write(key)
        os.chmod(KEY_FILE_PATH, 0o600)
    return key


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

    """Set and encrypt the API key."""
    encryption_key = fpk_get_or_create_encryption_key()
    encrypted_api_key = fpk_encrypt_data(api_key, encryption_key)
    config = {'api_key': encrypted_api_key}

    with open(CONFIG_FILE_PATH, "w") as config_file:
        toml.dump(config, config_file)
    os.chmod(CONFIG_FILE_PATH, 0o600)
    print("API key set successfully!")

    try:
        test_key = fpk_get_api_key()
        if test_key == api_key:
            print("Verification successful: API key can be decrypted correctly.")
        else:
            print("Verification failed: Decrypted key does not match the original.")
    except Exception as e:
        print(f"Error during API key verification: {e}")


def fpk_set_secure_file_permissions(file_path):
    """Set secure file permissions for the specified file."""
    os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)


def fpk_unbox(directory_name: str, session, verbose: bool = False, local: bool = False):
    """Unbox a flatpack from GitHub or a local directory."""
    flatpack_dir = Path.cwd() / directory_name
    flatpack_dir.mkdir(parents=True, exist_ok=True)
    build_dir = flatpack_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    temp_toml_path = build_dir / 'temp_flatpack.toml'

    if local:
        local_directory_path = flatpack_dir
        toml_path = local_directory_path / 'flatpack.toml'
        if not toml_path.exists():
            print(f"❌ flatpack.toml not found in the specified directory: '{directory_name}'.")
            return
        toml_content = toml_path.read_text()
    else:
        if not fpk_valid_directory_name(directory_name):
            print(f"❌ Invalid directory name: '{directory_name}'.")
            return

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
    command = f"bash {bash_script_path}"
    result = os.system(command)

    if result != 0:
        print("❌ Error: Failed to execute the bash script.")
    else:
        fpk_cache_last_flatpack(directory_name)
        print(f"🎉 All done!")


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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def landing_page():
    return JSONResponse(content={"message": "flatpack"})


@app.get("/test")
async def test_endpoint():
    return JSONResponse(content={"message": "Hello, World!"})


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
    parser_help.set_defaults(func=fpk_cli_handle_help)

    parser_version = subparsers.add_parser('version', help='Display the version of flatpack.')
    parser_version.set_defaults(func=fpk_cli_handle_version)

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

    # Unbox commands
    parser_unbox = subparsers.add_parser('unbox', help='Unbox a flatpack from GitHub or a local directory.')
    parser_unbox.add_argument('input', nargs='?', default=None, help='The name of the flatpack to unbox.')
    parser_unbox.add_argument('--local', action='store_true', help='Unbox from a local directory instead of GitHub.')
    parser_unbox.add_argument('--verbose', action='store_true', help='Display detailed outputs for debugging.')
    parser_unbox.set_defaults(func=lambda args, session: fpk_cli_handle_unbox(args, session))

    # Build commands
    parser_build = subparsers.add_parser('build',
                                         help='Build a model using the building script from the last unboxed flatpack.')
    parser_build.add_argument('directory', nargs='?', default=None, help='The directory of the flatpack to build.')
    parser_build.add_argument('--verbose', action='store_true', help='Display detailed outputs for debugging.')
    parser_build.set_defaults(func=lambda args, session: fpk_cli_handle_build(args, session))

    # Run server
    parser_run = subparsers.add_parser('run', help='Run the FastAPI server.')
    parser_run.set_defaults(func=lambda args, session: fpk_cli_handle_run(args, session))

    # Vector database management
    parser_vector = subparsers.add_parser('vector', help='Vector database management')
    vector_subparsers = parser_vector.add_subparsers(dest='vector_command')

    parser_add_text = vector_subparsers.add_parser('add-texts',
                                                   help='Add new texts to generate embeddings and store them.')
    parser_add_text.add_argument('texts', nargs='+', help='Texts to add.')
    parser_add_text.add_argument('--data-dir', type=str, default='.',
                                 help='Directory path for storing the vector database and metadata files.')
    parser_add_text.set_defaults(func=lambda args, session: fpk_cli_handle_vector_commands(args, session))

    parser_search_text = vector_subparsers.add_parser('search-text',
                                                      help='Search for texts similar to the given query.')
    parser_search_text.add_argument('query', help='Text query to search for.')
    parser_search_text.add_argument('--data-dir', type=str, default='.',
                                    help='Directory path for storing the vector database and metadata files.')
    parser_search_text.set_defaults(func=lambda args, session: fpk_cli_handle_vector_commands(args, session))

    parser_add_pdf = vector_subparsers.add_parser('add-pdf', help='Add text from a PDF file to the vector database.')
    parser_add_pdf.add_argument('pdf_path', help='Path to the PDF file to add.')
    parser_add_pdf.add_argument('--data-dir', type=str, default='.',
                                help='Directory path for storing the vector database and metadata files.')
    parser_add_pdf.set_defaults(func=lambda args, session: fpk_cli_handle_vector_commands(args, session))

    parser_add_url = vector_subparsers.add_parser('add-url', help='Add text from a URL to the vector database.')
    parser_add_url.add_argument('url', help='URL to add.')
    parser_add_url.add_argument('--data-dir', type=str, default='.',
                                help='Directory path for storing the vector database and metadata files.')
    parser_add_url.set_defaults(func=lambda args, session: fpk_cli_handle_vector_commands(args, session))

    parser_add_wikipedia_page = vector_subparsers.add_parser('add-wikipedia',
                                                             help='Add text from a Wikipedia page to the vector database.')
    parser_add_wikipedia_page.add_argument('page_title', help='The title of the Wikipedia page to add.')
    parser_add_wikipedia_page.add_argument('--data-dir', type=str, default='.',
                                           help='Directory path for storing the vector database and metadata files.')
    parser_add_wikipedia_page.set_defaults(func=lambda args, session: fpk_cli_handle_vector_commands(args, session))

    # Add commands for agents
    parser_agents = subparsers.add_parser('agents', help='Manage agents')
    agent_subparsers = parser_agents.add_subparsers(dest='agent_command', help='Available agent commands')

    # Add command to spawn an agent
    parser_spawn = agent_subparsers.add_parser('spawn', help='Spawn a new agent with a script')
    parser_spawn.add_argument('script_path', type=str, help='Path to the script to execute')
    parser_spawn.set_defaults(func=fpk_cli_handle_spawn_agent)

    # Add command to list active agents
    parser_list = agent_subparsers.add_parser('list', help='List active agents')
    parser_list.set_defaults(func=fpk_cli_handle_list_agents)

    # Add command to terminate an agent
    parser_terminate = agent_subparsers.add_parser('terminate', help='Terminate an active agent')
    parser_terminate.add_argument('pid', type=int, help='Process ID of the agent to terminate')
    parser_terminate.set_defaults(func=fpk_cli_handle_terminate_agent)

    return parser


def fpk_cli_handle_add_pdf(pdf_path, vm):
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file does not exist: '{pdf_path}'.")
        return
    vm.add_pdf(pdf_path)
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


def fpk_cli_handle_run(args, session):
    fpk_check_ngrok_auth()
    try:
        port = 8000
        listener = ngrok.forward(port, authtoken_from_env=True)
        print(f"Ingress established at {listener.url()}")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        print("❌ FastAPI server has been stopped.")
    except Exception as e:
        print(f"❌ An unexpected error occurred during server run: {e}")
    finally:
        ngrok.disconnect(listener.url())


def fpk_cli_handle_set_api_key(args, session):
    print(f"Setting API key: {args.api_key}")
    global config
    api_key = args.api_key

    """Set and encrypt the API key."""
    encryption_key = fpk_get_or_create_encryption_key()
    encrypted_api_key = fpk_encrypt_data(api_key, encryption_key)
    config = {'api_key': encrypted_api_key}

    with open(CONFIG_FILE_PATH, "w") as config_file:
        toml.dump(config, config_file)
    os.chmod(CONFIG_FILE_PATH, 0o600)
    print("API key set successfully!")

    # Optionally, verify the key immediately after setting
    try:
        test_key = fpk_get_api_key()  # Attempt to retrieve the key to ensure it can be decrypted properly
        if test_key == api_key:
            print("Verification successful: API key can be decrypted correctly.")
        else:
            print("Verification failed: Decrypted key does not match the original.")
    except Exception as e:
        print(f"Error during API key verification: {e}")


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
    if not args.local and directory_name not in fpk_fetch_github_dirs(session):
        print(f"❌ The flatpack '{directory_name}' does not exist.")
        return

    if args.local:
        local_directory_path = Path(directory_name)
        if not local_directory_path.exists() or not local_directory_path.is_dir():
            print(f"❌ Local directory does not exist: '{directory_name}'.")
            return
        toml_path = local_directory_path / 'flatpack.toml'
        if not toml_path.exists():
            print(f"❌ flatpack.toml not found in the specified directory: '{directory_name}'.")
            return

    print("Verbose mode:", args.verbose)
    print(f"✅ Directory name resolved to: '{directory_name}'")
    fpk_unbox(directory_name, session, verbose=args.verbose, local=args.local)


def fpk_cli_handle_version(args, session):
    print(VERSION)


def fpk_cli_handle_vector_commands(args, session):
    print("Handling vector commands...")

    # Instantiate the vector manager with possible custom data directory
    vm = VectorManager(model_name='all-MiniLM-L6-v2', directory=getattr(args, 'data_dir', '.'))

    if args.vector_command == 'add-texts':
        vm.add_texts(args.texts)
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

            # Directly call the function assigned to the current command
            if hasattr(args, 'func'):
                args.func(args, session)  # Passing session if needed by the function
            else:
                parser.print_help()

    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
