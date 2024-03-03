from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .parsers import parse_toml_to_venv_script
from pathlib import Path
from typing import List, Optional

import argparse
import chromadb
import httpx
import ngrok
import os
import re
import select
import shlex
import stat
import subprocess
import sys
import toml
import uvicorn

CONFIG_FILE_PATH = os.path.join(os.path.expanduser("~"), ".fpk_config.toml")
LOGGING_BATCH_SIZE = 10
GITHUB_REPO_URL = "https://api.github.com/repos/romlingroup/flatpack"
BASE_URL = "https://raw.githubusercontent.com/romlingroup/flatpack/main/warehouse"
LOGGING_ENDPOINT = "https://fpk.ai/api/index.php"

log_queue = []

config = {
    "api_key": None
}


class SessionManager:
    def __enter__(self):
        self.session = httpx.Client()
        return self.session

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()


def fpk_get_encryption_key():
    key = os.environ.get("FPK_ENCRYPTION_KEY")
    if key is not None:
        return key.encode()
    return None


def fpk_encrypt_data(data, key):
    fernet = Fernet(key)
    if isinstance(data, str):
        data = data.encode()
    return fernet.encrypt(data)


def fpk_decrypt_data(data, key):
    fernet = Fernet(key)
    if isinstance(data, str):
        data = data.encode()
    decrypted = fernet.decrypt(data)
    return decrypted.decode()


def fpk_set_secure_file_permissions(file_path):
    os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR)


class FPKEncryptionKeyError(Exception):
    """Custom exception for missing encryption key."""
    pass


def fpk_set_api_key(api_key: str):
    encryption_key = fpk_get_encryption_key()
    if not encryption_key:
        raise FPKEncryptionKeyError("❌ Encryption key not set.")
    encrypted_api_key = fpk_encrypt_data(api_key, encryption_key)
    config["api_key"] = encrypted_api_key.decode()

    with open(CONFIG_FILE_PATH, "w") as config_file:
        toml.dump(config, config_file)
    fpk_set_secure_file_permissions(CONFIG_FILE_PATH)


def fpk_get_api_key() -> Optional[str]:
    encryption_key = fpk_get_encryption_key()
    if not encryption_key or not os.path.exists(CONFIG_FILE_PATH):
        return None

    with open(CONFIG_FILE_PATH, "r") as config_file:
        loaded_config = toml.load(config_file)
        encrypted_api_key_str = loaded_config.get("api_key")
        if encrypted_api_key_str:
            encrypted_api_key_bytes = encrypted_api_key_str.encode()
            return fpk_decrypt_data(encrypted_api_key_bytes, encryption_key)
        else:
            return None


def fpk_cache_last_flatpack(directory_name: str):
    """Cache the last installed flatpack's directory name to a file.

    Args:
        directory_name (str): Name of the flatpack directory.
    """
    cache_file_path = os.path.join(os.getcwd(), 'last_flatpack.cache')
    with open(cache_file_path, 'w') as f:
        f.write(directory_name)


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
PLEASE NOTE: The flatpack you are about to install is
governed by its own licenses and terms, separate from
this installer. You may find further details at:

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


def fpk_install(directory_name: str, session, verbose: bool = False):
    """Install a specified flatpack."""
    if not fpk_valid_directory_name(directory_name):
        print(f"❌ Invalid directory name: '{directory_name}'.")
        return

    toml_content = fpk_fetch_flatpack_toml_from_dir(directory_name, session)
    if not toml_content:
        print(f"❌ Error: Failed to fetch TOML content for '{directory_name}'.")
        return

    with open('temp_flatpack.toml', 'w') as f:
        f.write(toml_content)

    bash_script_content = parse_toml_to_venv_script('temp_flatpack.toml', '3.10.12', directory_name)

    with open('flatpack.sh', 'w') as f:
        f.write(bash_script_content)

    os.remove('temp_flatpack.toml')

    print(f"Installing {directory_name}...")

    command = ["bash", shlex.quote("flatpack.sh")]

    try:
        if verbose:
            process = subprocess.Popen(command)
            process.wait()
        else:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print("❌ Error: Failed to execute the bash script.")
                # Uncomment for debugging
                # print("Standard Output:", stdout.decode())
                # print("Standard Error:", stderr.decode())

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


def fpk_process_output(output, session, last_installed_flatpack):
    """Process output and log it."""
    # Get the API key for logging
    api_key = fpk_get_api_key()

    # Regular expression pattern to match ANSI escape codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    # Iterate over each line in the output
    for line in output.splitlines():
        # Remove any leading or trailing whitespace
        line = line.strip()

        # Remove ANSI escape codes from the line
        line = ansi_escape.sub('', line)

        # If the line isn't empty, process it
        if line:
            # Display the line with a prefix
            print(f"(*) {line}", flush=True)

            # If we have an API key, log the line to the API
            if api_key:
                fpk_log_to_api(line, session, api_key=api_key, model_name=last_installed_flatpack)


def fpk_train(directory_name: str = None, session: httpx.Client = None):
    """Train a model using a training script from a specific or last installed flatpack."""
    cache_file_path = Path.cwd() / 'last_flatpack.cache'

    if directory_name:
        if not fpk_valid_directory_name(directory_name):
            print(f"❌ Invalid directory name: '{directory_name}'.")
            return
        last_installed_flatpack = directory_name
        fpk_cache_last_flatpack(directory_name)
    else:
        if not cache_file_path.exists():
            print("❌ No cached flatpack found.")
            return
        with cache_file_path.open('r') as f:
            last_installed_flatpack = f.read().strip()
            if not fpk_valid_directory_name(last_installed_flatpack):
                print(f"❌ Invalid directory name from cache: '{last_installed_flatpack}'.")
                return

    training_script_path = Path(last_installed_flatpack) / 'train.sh'
    if not training_script_path.exists():
        print(f"❌ Training script not found in {last_installed_flatpack}.")
        return

    # Start the subprocess for the training script
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    safe_script_path = shlex.quote(str(training_script_path))

    try:
        proc = subprocess.Popen(["bash", "-u", safe_script_path], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True,
                                env=env)

        outputs = [proc.stdout, proc.stderr]

        try:
            # Continuously read from the subprocess's output
            while True:
                # Check if process is still running
                retcode = proc.poll()
                if retcode is not None:
                    break  # Process finished

                # Read lines from stdout and stderr
                rlist, _, _ = select.select(outputs, [], [], 0.1)
                for r in rlist:
                    line = r.readline()
                    if line:
                        fpk_process_output(line, session, last_installed_flatpack)

                        # If the line does not end with a newline, it's likely a prompt waiting for input
                        if not line.endswith('\n'):
                            user_input = input()  # Get input from user
                            print(user_input, file=proc.stdin)  # Send input to subprocess

        finally:
            # Ensure subprocess finishes
            proc.wait()
    except subprocess.SubprocessError as e:
        print(f"❌ An error occurred while executing the subprocess: {e}")


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
client = chromadb.Client()

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


def main():
    try:
        with SessionManager() as session:
            parser = argparse.ArgumentParser(description='flatpack command line interface')
            parser.add_argument('command', help='Command to run')
            parser.add_argument('input', nargs='?', default=None, help='Input for the callback')
            parser.add_argument('--verbose', action='store_true', help='Display detailed outputs for debugging.')

            args = parser.parse_args()
            command = args.command

            fpk_get_api_key()

            if command == "find":
                print(fpk_find_models())
            elif command == "help":
                print("[HELP]")
            elif command == "get-api-key":
                print(fpk_get_api_key())
            elif command == "install":
                if not args.input:
                    print("❌ Please specify a flatpack for the install command.")
                    return

                directory_name = args.input

                existing_dirs = fpk_fetch_github_dirs(session)
                if directory_name not in existing_dirs:
                    print(f"❌ The flatpack '{directory_name}' does not exist.")
                    return

                fpk_display_disclaimer(directory_name)
                while True:
                    user_response = input().strip().upper()
                    if user_response == "YES":
                        break
                    elif user_response == "NO":
                        print("❌ Installation aborted by user.")
                        return
                    else:
                        print("❌ Invalid input. Please type 'YES' to accept or 'NO' to decline.")

                print("Verbose mode:", args.verbose)
                fpk_install(directory_name, session, verbose=args.verbose)

            elif command == "list":
                print(fpk_list_directories(session))
            elif command == "run":

                try:
                    port = 8000
                    listener = ngrok.forward(port, authtoken_from_env=True)
                    print(f"Ingress established at {listener.url()}")
                    uvicorn.run(app, host="0.0.0.0", port=port)
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                except KeyboardInterrupt:
                    print("FastAPI server has been stopped.")
                except Exception as e:
                    print(f"An unexpected error occurred: {e}")
                finally:
                    print("Finalizing...")
                    ngrok.disconnect(listener.url())

            elif command == "set-api-key":
                if not args.input:
                    print("❌ Please provide an API key to set.")
                    return
                fpk_set_api_key(args.input)
                print("API key set successfully!")
            elif command == "train":
                directory_name = args.input
                fpk_train(directory_name, session)
            elif command == "version":
                print("[VERSION]")
            else:
                print(f"Unknown command: {command}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    sys.exit(1)


if __name__ == "__main__":
    main()
