import argparse
import datetime
import os
import pty
import requests
import select
import subprocess
import sys
import tempfile
import termios
import threading
import time
import toml
from .parsers import parse_toml_to_pyenv_script
from .instructions import build


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
        "default": "\033[0m"  # Resets the color
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


def fpk_fetch_flatpack_toml_from_dir(directory_name: str) -> str:
    base_url = "https://raw.githubusercontent.com/romlingroup/flatpack-ai/main/warehouse"
    toml_url = f"{base_url}/{directory_name}/flatpack.toml"
    response = requests.get(toml_url)
    if response.status_code != 200:
        return None
    return response.text


def fpk_fetch_github_dirs() -> list:
    url = "https://api.github.com/repos/romlingroup/flatpack-ai/contents/warehouse"
    response = requests.get(url)
    if response.status_code != 200:
        return ["❌ Error fetching data from GitHub"]
    directories = [item['name'] for item in response.json() if
                   item['type'] == 'dir' and item['name'].lower() != 'archive']
    return sorted(directories)


def fpk_find_models(directory_path: str = None) -> list:
    if directory_path is None:
        directory_path = os.getcwd()
    model_file_formats = ['.h5', '.json', '.onnx', '.pb', '.pt']
    model_files = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.endswith(fmt) for fmt in model_file_formats):
                model_files.append(os.path.join(root, file))
    return model_files


def fpk_install(directory_name: str):
    existing_dirs = fpk_fetch_github_dirs()
    if directory_name not in existing_dirs:
        print(f"❌ Error: The directory '{directory_name}' does not exist.")
        return

    toml_content = fpk_fetch_flatpack_toml_from_dir(directory_name)

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

            fpk_log_session(f"Installed {directory_name}")
            fpk_cache_last_flatpack(directory_name)

            print(f"🎉 All done!")

        except subprocess.CalledProcessError:
            print("❌ Error: Failed to execute the bash script.")
        finally:
            if os.path.exists("flatpack.sh"):
                os.remove("flatpack.sh")

    else:
        print(f"❌ No flatpack.toml found in {directory_name}.\n")


def fpk_list_directories() -> str:
    dirs = fpk_fetch_github_dirs()
    return "\n".join(dirs)


def fpk_list_processes():
    print("Placeholder for fpk_list_processes")


def fpk_log_session(message: str):
    session_file_path = os.path.join(os.getcwd(), 'fpk_session.log')
    with open(session_file_path, 'a') as f:
        formatted_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{formatted_date}: {message.strip()}\n")


def fpk_record_data(message: str):
    record_data_path = os.path.join(os.getcwd(), 'fpk_record_data.log')
    with open(record_data_path, 'a') as f:
        formatted_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{formatted_date}: {message.strip()}\n")


def fpk_train(directory_name: str = None):
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
        try:
            while True:
                rlist, _, _ = select.select([master, 0], [], [])
                if master in rlist:
                    output = os.read(master, 1024).decode()
                    lines = output.splitlines()
                    for line in lines:

                        if line == last_user_input:
                            continue

                        if line.strip() and line != last_printed:
                            # BEGIN Record line
                            print(f"(*) {line}")
                            record_line = line
                            fpk_record_data(record_line)
                            # END Record line

                            last_printed = line

                if 0 in rlist:
                    user_input = sys.stdin.readline().strip()
                    last_user_input = user_input

                    # BEGIN Record user input
                    print(fpk_colorize(f"(*) {last_user_input}", "yellow"))
                    record_user_input = last_user_input
                    fpk_record_data(record_user_input)
                    # END Record user input

                    os.write(master, (user_input + '\n').encode())

        except OSError:
            pass

        _, exit_status = os.waitpid(pid, 0)
        if exit_status != 0:
            print("❌ Failed to execute the training script.")
        else:
            print("🎉 All done!")


def main():
    parser = argparse.ArgumentParser(description='flatpack.ai command line interface')
    parser.add_argument('command', help='Command to run')
    parser.add_argument('input', nargs='?', default=None, help='Input for the callback')

    args = parser.parse_args()
    command = args.command
    if command == "callback":
        fpk_callback(args.input)
    elif command == "find":
        print(fpk_find_models())
    elif command == "help":
        print("[HELP]")
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
    elif command == "train":
        directory_name = args.input
        fpk_train(directory_name)
    elif command == "version":
        print("[VERSION]")
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
