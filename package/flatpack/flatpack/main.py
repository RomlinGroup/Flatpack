from collections import deque
from .instructions import build
from .parsers import parse_toml_to_venv_script
from pathlib import Path
from typing import List, Optional

import argparse
import httpx
import logging
import os
import pty
import re
import select
import subprocess
import sys
import time
import toml

CONFIG_FILE_PATH = os.path.join(os.path.expanduser("~"), ".fpk_config.toml")
LOGGING_BATCH_SIZE = 10
GITHUB_REPO_URL = "https://api.github.com/repos/romlingroup/flatpack-ai"
BASE_URL = "https://raw.githubusercontent.com/romlingroup/flatpack-ai/main/warehouse"
LOGGING_ENDPOINT = "https://fpk.ai/api/index.php"

logger = logging.getLogger(__name__)
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


def fpk_cache_last_flatpack(directory_name: str):
    """Cache the last installed flatpack's directory name to a file.

    Args:
        directory_name (str): Name of the flatpack directory.
    """
    cache_file_path = os.path.join(os.getcwd(), 'last_flatpack.cache')
    with open(cache_file_path, 'w') as f:
        f.write(directory_name)


def fpk_callback(input_variable=None):
    """Print a callback message with or without a provided input.

    Args:
        input_variable (Optional[str]): User-provided input. Defaults to None.
    """
    if input_variable:
        print(f"You provided the input: {input_variable}")
    else:
        print("No input provided!")
    print("It works!")


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
Copyright 2023 Romlin Group AB

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
    model_file_formats = ['.caffemodel', '.ckpt', '.gguf', '.h5', '.json', '.mar', '.mlmodel',
                          '.model', '.onnx', '.params', '.pb', '.pkl', '.pickle', '.pt', '.sav', '.tflite', '.weights']
    model_files = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.endswith(fmt) for fmt in model_file_formats):
                model_files.append(os.path.join(root, file))
    return model_files


def fpk_get_api_key() -> Optional[str]:
    """Fetch the API key from the configuration file.

    Returns:
        Optional[str]: The API key if found, otherwise None.
    """
    if not config["api_key"]:
        if not os.path.exists(CONFIG_FILE_PATH):
            return None
        with open(CONFIG_FILE_PATH, "r") as config_file:
            loaded_config = toml.load(config_file)
            config["api_key"] = loaded_config.get("api_key")
    return config["api_key"]


def fpk_install(directory_name: str, session, verbose: bool = False):
    """Install a specified flatpack."""

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

    if verbose:
        process = subprocess.Popen(["bash", "flatpack.sh"])
        process.wait()
    else:
        process = subprocess.Popen(["bash", "flatpack.sh"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print("❌ Error: Failed to execute the bash script.")
            # Optional: Uncomment the following lines to display the stdout and stderr for debugging
            # print("Standard Output:", stdout.decode())
            # print("Standard Error:", stderr.decode())

    if process.returncode == 0:
        fpk_cache_last_flatpack(directory_name)
        # os.remove('flatpack.sh')
        print(f"🎉 All done!")


def fpk_list_directories(session: httpx.Client) -> str:
    """Fetch a list of directories from GitHub and return as a newline-separated string.

    Parameters:
    - session (httpx.Client): HTTP client session for making requests.

    Returns:
    - str: A newline-separated string of directory names.
    """
    dirs = fpk_fetch_github_dirs(session)
    return "\n".join(dirs)


def fpk_list_processes():
    """Placeholder for a function that lists processes."""
    print("Placeholder for fpk_list_processes")


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
            # logger.warning("API key not set.")
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
        logger.error(f"Failed to send request: {e}")


def fpk_set_api_key(api_key: str):
    """Set and save the API key to the configuration file."""
    global config
    config["api_key"] = api_key
    with open(CONFIG_FILE_PATH, "w") as config_file:
        toml.dump(config, config_file)
    logger.info("API key set successfully!")


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
    # Define the path for the cache file
    cache_file_path = Path.cwd() / 'last_flatpack.cache'

    # If a directory name is provided, set it as the last installed flatpack
    if directory_name:
        last_installed_flatpack = directory_name
        fpk_cache_last_flatpack(directory_name)
    else:
        # Otherwise, use the directory name from the cache file
        if not cache_file_path.exists():
            print("❌ No cached flatpack found.")
            return
        with cache_file_path.open('r') as f:
            last_installed_flatpack = f.read().strip()

        # Determine the path for the training script
        training_script_path = Path(last_installed_flatpack) / 'train.sh'

        # Check if the training script is present
        if not training_script_path.exists():
            print(f"❌ Training script not found in {last_installed_flatpack}.")
            return

        # Start the subprocess for the training script
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(["bash", "-u", str(training_script_path)], stdin=subprocess.PIPE,
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


def main():
    with SessionManager() as session:
        parser = argparse.ArgumentParser(description='flatpack.ai command line interface')
        parser.add_argument('command', help='Command to run')
        parser.add_argument('input', nargs='?', default=None, help='Input for the callback')
        parser.add_argument('--model-name', default="YOUR_MODEL_NAME",
                            help='Name of the model to associate with the log')
        parser.add_argument('--verbose', action='store_true', help='Display detailed outputs for debugging.')

        args = parser.parse_args()
        command = args.command
        fpk_get_api_key()

        if command == "callback":
            fpk_callback(args.input)
        elif command == "find":
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
        elif command == "ps":
            print(fpk_list_processes())
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


if __name__ == "__main__":
    main()
