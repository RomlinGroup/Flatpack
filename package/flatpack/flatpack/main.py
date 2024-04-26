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
import stat
import subprocess
import sys
import toml
import uvicorn

from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .parsers import parse_toml_to_venv_script
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from .vector_manager import VectorManager

CONFIG_FILE_PATH = os.path.join(os.path.expanduser("~"), ".fpk_config.toml")
KEY_FILE_PATH = os.path.join(os.path.expanduser("~"), ".fpk_encryption_key")
LOGGING_BATCH_SIZE = 10
GITHUB_REPO_URL = "https://api.github.com/repos/romlingroup/flatpack"
BASE_URL = "https://raw.githubusercontent.com/romlingroup/flatpack/main/warehouse"
LOGGING_ENDPOINT = "https://fpk.ai/api/index.php"

config = {
    "api_key": None
}


def safe_cleanup():
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


atexit.register(safe_cleanup)


class AgentManager:
    def __init__(self, filepath='processes.json'):
        self.processes = {}
        self.filepath = filepath
        self.load_processes()

    def save_processes(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.processes, f, default=str)

    def load_processes(self):
        try:
            with open(self.filepath, 'r') as f:
                self.processes = json.load(f)

                for pid in list(self.processes.keys()):
                    process = subprocess.Popen.poll(subprocess.Popen(pid=pid))
                    if process is None or process.returncode is not None:
                        del self.processes[pid]
        except FileNotFoundError:
            self.processes = {}

    def spawn_agent(self, script_path):
        process = subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pid = process.pid
        self.processes[str(pid)] = {
            'process': process.pid,
            'start_time': datetime.now(),
            'script_path': script_path
        }
        self.save_processes()
        print(f"Started process {pid} running '{script_path}'")
        return pid

    def list_agents(self):
        print("Active processes:")
        for pid, details in self.processes.items():
            print(f"PID: {pid}, Script: {details['script_path']}, Start Time: {details['start_time']}")

    def terminate_agent(self, pid):
        pid = str(pid)
        if pid in self.processes:
            process = subprocess.Popen(pid=int(pid))
            process.terminate()
            process.wait()
            print(f"Terminated process {pid}")
            del self.processes[pid]
            self.save_processes()
        else:
            print(f"No process with PID {pid} found.")


class SessionManager:
    def __enter__(self):
        """Create an HTTP session."""
        self.session = httpx.Client()
        return self.session

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the HTTP session."""
        self.session.close()


def fpk_encrypt_data(data: str, key: bytes) -> str:
    """Encrypt data using the provided key."""
    fernet = Fernet(key)
    return fernet.encrypt(data.encode()).decode()


def fpk_decrypt_data(encrypted_data: str, key: bytes) -> str:
    """Decrypt data using the provided key."""
    fernet = Fernet(key)
    return fernet.decrypt(encrypted_data.encode()).decode()


def fpk_set_secure_file_permissions(file_path):
    """Set secure file permissions for the specified file."""
    os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)


class FPKEncryptionKeyError(Exception):
    """Custom exception for missing encryption key."""
    pass


def get_or_create_encryption_key() -> bytes:
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


def fpk_set_api_key(api_key: str):
    """Set the API key in the configuration file."""
    global config

    """Set and encrypt the API key."""
    encryption_key = get_or_create_encryption_key()
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


def fpk_get_api_key() -> Optional[str]:
    """Retrieve and decrypt the API key from the configuration file."""
    if not os.path.exists(CONFIG_FILE_PATH):
        return None

    try:
        with open(CONFIG_FILE_PATH, 'r') as config_file:
            loaded_config = toml.load(config_file)
        encrypted_api_key = loaded_config.get('api_key')

        if encrypted_api_key:
            encryption_key = get_or_create_encryption_key()
            return fpk_decrypt_data(encrypted_api_key, encryption_key)
    except Exception as e:
        print(f"Error decrypting API key: {e}")

    return None


def fpk_cache_last_flatpack(directory_name: str):
    """Cache the last unboxed flatpack's directory name to a file within the corresponding build directory."""
    flatpack_dir = Path.cwd()
    flatpack_dir.mkdir(parents=True, exist_ok=True)

    cache_file_path = flatpack_dir / 'last_flatpack.cache'
    with open(cache_file_path, 'w') as f:
        f.write(directory_name)


def fpk_get_last_flatpack(directory_name: str) -> Optional[str]:
    """Retrieve the last unboxed flatpack's directory name from the cache file within the correct build directory."""
    flatpack_dir = Path.cwd()
    cache_file_path = flatpack_dir / 'last_flatpack.cache'
    if cache_file_path.exists():
        return cache_file_path.read_text().strip()
    return None


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

    command = ["bash", str(bash_script_path)]

    try:
        if verbose:
            process = subprocess.Popen(command)
            process.wait()
        else:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print("❌ Error: Failed to execute the bash script.")
                if verbose:
                    print("Standard Output:", stdout.decode())
                    print("Standard Error:", stderr.decode())

        if process.returncode == 0:
            fpk_cache_last_flatpack(directory_name)
            print(f"🎉 All done!")
    except subprocess.SubprocessError as e:
        print(f"❌ An error occurred: {e}")


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


def fpk_log_to_api(message: str, session: httpx.Client, api_key: Optional[str] = None,
                   model_name: str = "YOUR_MODEL_NAME"):
    """Log a message to the API.

    Args:
        message (str): The log message.
        session (httpx.Client): HTTP client session for making requests.
        api_key (Optional[str]): The API key for authentication. Defaults to the global API_KEY.
        model_name (str): Name of the model associated with the log. Defaults to "YOUR_MODEL_NAME".
    """
    if not api_key:
        api_key = config["api_key"]
        if not api_key:
            print("❌ API key not set.")
            return

    headers = {
        "Content-Type": "application/json"
    }
    params = {
        "endpoint": "log-message",
        "api_key": api_key
    }
    data = {
        "model_name": model_name,
        "log_message": message
    }

    try:
        response = session.post(LOGGING_ENDPOINT, params=params, json=data, headers=headers, timeout=10)
    except httpx.RequestError as e:
        print(f"Failed to send request: {e}")


def fpk_process_output(output, session, last_unboxed_flatpack):
    """Process output and log it."""
    api_key = fpk_get_api_key()

    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    for line in output.splitlines():
        line = line.strip()
        line = ansi_escape.sub('', line)

        if line:
            print(f"(*) {line}", flush=True)

            if api_key:
                fpk_log_to_api(line, session, api_key=api_key, model_name=last_unboxed_flatpack)


def fpk_build(directory: str, session: httpx.Client = None):
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

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    safe_script_path = shlex.quote(str(building_script_path))

    try:
        proc = subprocess.Popen(["bash", "-u", safe_script_path], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True,
                                env=env)

        outputs = [proc.stdout, proc.stderr]

        while True:
            retcode = proc.poll()
            if retcode is not None:
                break

            rlist, _, _ = select.select(outputs, [], [], 0.1)
            for r in rlist:
                line = r.readline()
                if line:
                    fpk_process_output(line, session, last_unboxed_flatpack)

                    if not line.endswith('\n'):
                        try:
                            user_input = input()
                            print(user_input, file=proc.stdin)
                        except EOFError:
                            break

    except subprocess.SubprocessError as e:
        print(f"❌ An error occurred while executing the subprocess: {e}")
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        if proc and proc.poll() is None:
            proc.wait()


def fpk_valid_directory_name(name: str) -> bool:
    """
    Validate that the directory name contains only alphanumeric characters, dashes, and underscores.

    Args:
        name (str): The directory name to validate.

    Returns:
        bool: True if the name is valid, False otherwise.
    """
    return re.match(r'^[\w-]+$', name) is not None


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


def check_ngrok_auth():
    """Check if the NGROK_AUTHTOKEN environment variable is set."""
    ngrok_auth_token = os.environ.get('NGROK_AUTHTOKEN')
    if not ngrok_auth_token:
        print("❌ Error: NGROK_AUTHTOKEN is not set. Please set it using:")
        print("export NGROK_AUTHTOKEN='your_ngrok_auth_token'")
        sys.exit(1)
    else:
        print("NGROK_AUTHTOKEN is set.")


def setup_arg_parser():
    # Create the top-level parser
    parser = argparse.ArgumentParser(description='Flatpack command line interface')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # General commands
    subparsers.add_parser('list', help='List available flatpack directories.')
    subparsers.add_parser('find', help='Find model files in the current directory.')
    subparsers.add_parser('help', help='Display help for commands.')
    subparsers.add_parser('version', help='Display the version of flatpack.')

    # API Key Management
    parser_api_key = subparsers.add_parser('api-key', help='API key management commands')
    api_key_subparsers = parser_api_key.add_subparsers(dest='api_key_command')
    get_api_key_parser = api_key_subparsers.add_parser('get', help='Get the current API key.')
    set_api_key_parser = api_key_subparsers.add_parser('set', help='Set the API key.')
    set_api_key_parser.add_argument('api_key', help='API key to set.')

    # Unbox commands
    parser_unbox = subparsers.add_parser('unbox', help='Unbox a flatpack from GitHub or a local directory.')
    parser_unbox.add_argument('input', nargs='?', default=None, help='The name of the flatpack to unbox.')
    parser_unbox.add_argument('--local', action='store_true', help='Unbox from a local directory instead of GitHub.')
    parser_unbox.add_argument('--verbose', action='store_true', help='Display detailed outputs for debugging.')

    # Build commands
    parser_build = subparsers.add_parser('build',
                                         help='Build a model using the building script from the last unboxed flatpack.')
    parser_build.add_argument('directory', nargs='?', default=None, help='The directory of the flatpack to build.')

    # Run server
    subparsers.add_parser('run', help='Run the FastAPI server.')

    # Vector database management
    parser_vector = subparsers.add_parser('vector', help='Vector database management')
    vector_subparsers = parser_vector.add_subparsers(dest='vector_command')

    parser_add_text = vector_subparsers.add_parser('add-texts',
                                                   help='Add new texts to generate embeddings and store them.')
    parser_add_text.add_argument('texts', nargs='+', help='Texts to add.')
    parser_add_text.add_argument('--data-dir', type=str, default='./data',
                                 help='Directory path for storing the vector database and metadata files.')

    parser_search_text = vector_subparsers.add_parser('search-text',
                                                      help='Search for texts similar to the given query.')
    parser_search_text.add_argument('query', help='Text query to search for.')
    parser_search_text.add_argument('--data-dir', type=str, default='./data',
                                    help='Directory path for storing the vector database and metadata files.')

    parser_add_pdf = vector_subparsers.add_parser('add-pdf', help='Add text from a PDF file to the vector database.')
    parser_add_pdf.add_argument('pdf_path', help='Path to the PDF file to add.')
    parser_add_pdf.add_argument('--data-dir', type=str, default='./data',
                                help='Directory path for storing the vector database and metadata files.')

    parser_add_url = vector_subparsers.add_parser('add-url', help='Add text from a URL to the vector database.')
    parser_add_url.add_argument('url', help='URL to add.')
    parser_add_url.add_argument('--data-dir', type=str, default='./data',
                                help='Directory path for storing the vector database and metadata files.')

    parser_add_wikipedia_page = vector_subparsers.add_parser('add-wikipedia',
                                                             help='Add text from a Wikipedia page to the vector database.')
    parser_add_wikipedia_page.add_argument('page_title', help='The title of the Wikipedia page to add.')
    parser_add_wikipedia_page.add_argument('--data-dir', type=str, default='./data',
                                           help='Directory path for storing the vector database and metadata files.')

    return parser


def main():
    """Main entry point for the flatpack command line interface."""
    try:
        with SessionManager() as session:
            parser = setup_arg_parser()
            args = parser.parse_args()

            # Dispatch to the appropriate function based on the command
            command_handlers = {
                'list': lambda: handle_list(session),
                'find': handle_find,
                'help': handle_help,
                'get-api-key': handle_get_api_key,
                'unbox': lambda: handle_unbox(args, session),
                'run': handle_run,
                'set-api-key': lambda: handle_set_api_key(args.api_key),
                'build': lambda: handle_build(args, session),
                'version': handle_version,
            }

            # Handling vector-related commands as a group
            if args.command in command_handlers:
                command_handlers[args.command]()
            elif 'vector-' in args.command:
                handle_vector_commands(args, session)

    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        sys.exit(1)


def handle_list(session):
    print(fpk_list_directories(session))


def handle_find():
    print(fpk_find_models())


def handle_help():
    print("[HELP]")


def handle_get_api_key():
    print(fpk_get_api_key())


def handle_unbox(args, session):
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


def handle_run():
    check_ngrok_auth()
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


def handle_set_api_key(api_key):
    fpk_set_api_key(api_key)


def handle_build(args, session):
    if args.directory:
        print("Building flatpack...")
        fpk_build(args.directory, session)
    else:
        fpk_build(None, session)


def handle_version():
    print("[VERSION]")


def handle_vector_commands(args, session):
    # Instantiate the vector manager with possible custom data directory
    vm = VectorManager(model_name='all-MiniLM-L6-v2', directory=getattr(args, 'data_dir', './data'))

    if args.command == 'vector-add-texts':
        vm.add_texts(args.texts)
        print(f"Added {len(args.texts)} texts to the database.")
    elif args.command == 'vector-search':
        results = vm.search_vectors(args.query)
        if results:
            print("Search results:")
            for result in results:
                print(f"({result['distance']}) {result['id']}: {result['text']}\n")
        else:
            print("No results found.")
    elif args.command == 'vector-add-pdf':
        handle_add_pdf(args.pdf_path, vm)
    elif args.command == 'vector-add-url':
        handle_add_url(args.url, vm)
    elif args.command == 'vector-add-wikipedia-page':
        vm.add_wikipedia_page(args.page_title)
        print(f"✅ Added text from Wikipedia page: '{args.page_title}' to the vector database.")


def handle_add_pdf(pdf_path, vm):
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file does not exist: '{pdf_path}'.")
        return
    vm.add_pdf(pdf_path)
    print(f"✅ Added text from PDF: '{pdf_path}' to the vector database.")


def handle_add_url(url, vm):
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if response.status_code >= 200 and response.status_code < 400:
            vm.add_url(url)
            print(f"✅ Added text from URL: '{url}' to the vector database.")
        else:
            print(f"❌ URL is not accessible: '{url}'. HTTP Status Code: {response.status_code}")
    except requests.RequestException as e:
        print(f"❌ Failed to access URL: '{url}'. Error: {e}")


if __name__ == "__main__":
    main()
