import argparse
import os
import requests
import sys
import toml
from .parsers import parse_toml_to_pyenv_script
from .instructions import build


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
        print("🎉 Bash script generated and saved as 'flatpack.sh'.")
        print(f"🔎 Location: {os.path.join(os.getcwd(), 'flatpack.sh')}")
        os.remove('temp_flatpack.toml')
    else:
        print(f"❌ No flatpack.toml found in {directory_name}.\n")


def fpk_list_directories() -> str:
    dirs = fpk_fetch_github_dirs()
    return "\n".join(dirs)


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


def fpk_callback():
    print("It works!")


def main():
    parser = argparse.ArgumentParser(description='flatpack.ai command line interface')
    parser.add_argument('command', help='Command to run')

    args = parser.parse_args(sys.argv[1:2])
    command = args.command
    if command == "callback":
        fpk_callback()
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
    elif command == "version":
        print("[VERSION]")
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
