import argparse
import asyncio
import httpx
import logging
import os
import pty
import select
import subprocess
import sys
from .instructions import build
from .parsers import parse_toml_to_pyenv_script

CONFIG_FILE_PATH = os.path.join(os.path.expanduser("~"), ".fpk_config.toml")
API_KEY: Optional[str] = None
logger = logging.getLogger(__name__)
LOGGING_BATCH_SIZE = 10
log_queue = asyncio.Queue()

# Fetch and store the API key once if it doesn't change frequently
API_KEY = fpk_get_api_key()


async def initialize_session():
    global session
    session = httpx.AsyncClient()


def fpk_cache_last_flatpack(directory_name: str):
    cache_file_path = os.path.join(os.getcwd(), 'last_flatpack.cache')
    with open(cache_file_path, 'w') as f:
        f.write(directory_name)


def fpk_callback(input_variable=None):
    if input_variable:
        print(f"You provided the input: {input_variable}")
    else:
        print("No input provided!")
    print("It works!")


def fpk_colorize(text, color):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "grey": "\033[90m",
        "default": "\033[0m"
    }
    return colors[color] + text + colors["default"]


def fpk_display_disclaimer(directory_name: str):
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


async def fpk_fetch_flatpack_toml_from_dir(directory_name: str) -> Optional[str]:
    base_url = "https://raw.githubusercontent.com/romlingroup/flatpack-ai/main/warehouse"
    toml_url = f"{base_url}/{directory_name}/flatpack.toml"

    async with httpx.AsyncClient() as client:
        response = await client.get(toml_url)
        if response.status_code != 200:
            return None
        return response.text


async def fpk_fetch_github_dirs() -> List[str]:
    url = "https://api.github.com/repos/romlingroup/flatpack-ai/contents/warehouse"

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        if response.status_code != 200:
            return ["❌ Error fetching data from GitHub"]
        directories = [item['name'] for item in response.json() if
                       item['type'] == 'dir' and item['name'].lower() != 'archive']
        return sorted(directories)


def fpk_find_models(directory_path: str = None) -> List[str]:
    if directory_path is None:
        directory_path = os.getcwd()
    model_file_formats = ['.h5', '.json', '.onnx', '.pb', '.pt']
    model_files = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.endswith(fmt) for fmt in model_file_formats):
                model_files.append(os.path.join(root, file))
    return model_files


def fpk_get_api_key() -> str:
    if not os.path.exists(CONFIG_FILE_PATH):
        return None
    with open(CONFIG_FILE_PATH, "r") as config_file:
        config = toml.load(config_file)
        return config.get("api_key")


def fpk_install(directory_name: str, session: httpx.AsyncClient):
    existing_dirs = asyncio.run(fpk_fetch_github_dirs(session))
    if directory_name not in existing_dirs:
        print(f"❌ Error: The directory '{directory_name}' does not exist.")
        return

    toml_content = asyncio.run(fpk_fetch_flatpack_toml_from_dir(directory_name, session))

    if toml_content:
        with open('temp_flatpack.toml', 'w') as f:
            f.write(toml_content)

        bash_script_content = parse_toml_to_pyenv_script('temp_flatpack.toml')

        with open('flatpack.sh', 'w') as f:
            f.write(bash_script_content)

        os.remove('temp_flatpack.toml')

        try:
            print(f"Installing {directory_name}...")
            process = subprocess.Popen(["bash", "flatpack.sh"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            process.communicate()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args)

            fpk_cache_last_flatpack(directory_name)

            print(f"🎉 All done!")

        except subprocess.CalledProcessError:
            print("❌ Error: Failed to execute the bash script.")
        finally:
            if os.path.exists("flatpack.sh"):
                os.remove("flatpack.sh")

    else:
        print(f"❌ No flatpack.toml found in {directory_name}.\n")


def fpk_list_directories(session: httpx.AsyncClient) -> str:
    dirs = asyncio.run(fpk_fetch_github_dirs(session))
    return "\n".join(dirs)


def fpk_list_processes():
    print("Placeholder for fpk_list_processes")


def fpk_log_to_api(message: str, model_name: str = "YOUR_MODEL_NAME"):
    if not API_KEY:
        logger.warning("API key not set.")
        return

    url = "https://fpk.ai/api/index.php"
    headers = {
        "Content-Type": "application/json"
    }
    params = {
        "endpoint": "log-message",
        "api_key": API_KEY
    }
    data = {
        "model_name": model_name,
        "log_message": message
    }

    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, params=params, json=data, headers=headers, timeout=10)
        except httpx.RequestError as e:
            logger.error(f"Failed to send request: {e}")


def fpk_set_api_key(api_key: str):
    config = {"api_key": api_key}
    with open(CONFIG_FILE_PATH, "w") as config_file:
        toml.dump(config, config_file)


def fpk_train(directory_name: str = None, session: httpx.AsyncClient = None):
    cache_file_path = os.path.join(os.getcwd(), 'last_flatpack.cache')

    if directory_name:
        last_installed_flatpack = directory_name
        fpk_cache_last_flatpack(directory_name)
    else:
        if not os.path.exists(cache_file_path):
            print("❌ No cached flatpack found.")
            return
        with open(cache_file_path, 'r') as f:
            last_installed_flatpack = f.read().strip()

    training_script_path = os.path.join(last_installed_flatpack, 'train.sh')

    if not os.path.exists(training_script_path):
        print(f"❌ Training script not found in {last_installed_flatpack}.")
        return

    master, slave = pty.openpty()

    pid = os.fork()
    if pid == 0:
        os.close(master)
        os.dup2(slave, 0)
        os.dup2(slave, 1)
        os.dup2(slave, 2)
        os.execvp("bash", ["bash", training_script_path])
    else:
        os.close(slave)
        last_printed = None
        last_user_input = None

        buffered_output = ""

        try:
            while True:
                rlist, _, _ = select.select([master, 0], [], [])
                if master in rlist:
                    output = os.read(master, 4096).decode()
                    buffered_output += output
                    lines = buffered_output.splitlines()

                    if buffered_output.endswith(('\n', '\r')):
                        buffered_output = ""
                    else:
                        buffered_output = lines.pop()

                    for line in lines:
                        line = line.strip()

                        if not line or line == last_user_input:
                            continue

                        print(f"(*) {line}")
                        log_queue.put_nowait((line, last_installed_flatpack))

                if 0 in rlist:
                    user_input = sys.stdin.readline().strip()
                    last_user_input = user_input

                    print(fpk_colorize(f"(*) {last_user_input}", "yellow"))
                    log_queue.put_nowait((user_input, last_installed_flatpack))
                    os.write(master, (user_input + '\n').encode())

        except OSError:
            pass

        # After the loop, process any remaining data in buffered_output
        if buffered_output:
            print(f"(*) {buffered_output}")
            log_queue.put_nowait((buffered_output, last_installed_flatpack))

        _, exit_status = os.waitpid(pid, 0)
        if exit_status != 0:
            print("❌ Failed to execute the training script.")
        else:
            print("🎉 All done!")
        buffered_output = ""  # Clear the buffered_output to release memory


def main():
    parser = argparse.ArgumentParser(description='flatpack.ai command line interface')
    parser.add_argument('command', help='Command to run')
    parser.add_argument('input', nargs='?', default=None, help='Input for the callback')
    parser.add_argument('--model-name', default="YOUR_MODEL_NAME", help='Name of the model to associate with the log')

    args = parser.parse_args()
    command = args.command

    session = httpx.AsyncClient()

    if command == "callback":
        fpk_callback(args.input)
    elif command == "find":
        print(fpk_find_models())
    elif command == "help":
        print("[HELP]")
    elif command == "get-api-key":
        print(fpk_get_api_key())
    elif command == "install":
        if len(sys.argv) < 3:
            print("❌ Please specify a flatpack for the install command.")
            return
        directory_name = sys.argv[2]
        fpk_display_disclaimer(directory_name)
        while True:
            user_response = input().strip().upper()
            if user_response == "YES":
                break
            elif user_response == "NO":
                print("❌ Installation aborted by user.")
                exit(0)
            else:
                print("❌ Invalid input. Please type 'YES' to accept or 'NO' to decline.")
        fpk_install(directory_name)
    elif command == "list":
        print(fpk_list_directories())
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

    session.aclose()


async def logging_task(session, api_key):
    batch = []
    while True:
        message, model_name = await log_queue.get()
        batch.append((message, model_name))

        if len(batch) >= LOGGING_BATCH_SIZE or log_queue.empty():
            # Send this batch to the server
            for msg, model in batch:
                await fpk_log_to_api(msg, api_key, session, model)
            batch.clear()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(initialize_session())
    main()
