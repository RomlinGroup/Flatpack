import os
import requests
import sys
import toml

from .parsers import parse_toml_to_pyenv_script


def colorize(text, color):
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


def display_disclaimer(directory_name: str):
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

    https://fpk.ai/{}
    """.format(directory_name)

    please_note_colored = colorize(please_note_content, "yellow")

    print(disclaimer_template.format(please_note=please_note_colored))


def fetch_flatpack_toml_from_dir(directory_name: str) -> str:
    """Fetch the flatpack.toml content from a specified GitHub directory."""
    base_url = "https://raw.githubusercontent.com/romlingroup/flatpack-ai/main/warehouse"
    toml_url = f"{base_url}/{directory_name}/flatpack.toml"
    response = requests.get(toml_url)

    if response.status_code != 200:
        return None

    return response.text


def fetch_github_dirs() -> list:
    """Retrieve a list of directories from the specified GitHub repo."""
    url = "https://api.github.com/repos/romlingroup/flatpack-ai/contents/warehouse"
    response = requests.get(url)

    if response.status_code != 200:
        return ["❌ Error fetching data from GitHub"]

    directories = [item['name'] for item in response.json() if item['type'] == 'dir']
    return sorted(directories)


def install(directory_name: str):
    """Fetch the flatpack.toml file, convert it to a bash script, and save it."""

    # Verify directory existence
    existing_dirs = fetch_github_dirs()
    if directory_name not in existing_dirs:
        print(f"❌ Error: The directory '{directory_name}' does not exist.")
        return

    toml_content = fetch_flatpack_toml_from_dir(directory_name)
    if toml_content:
        # print(f"Contents of flatpack.toml in {directory_name}:\n{toml_content}\n")

        # Save the TOML content to a temporary file
        with open('temp_flatpack.toml', 'w') as f:
            f.write(toml_content)

        # Convert TOML to bash script
        bash_script_content = parse_toml_to_pyenv_script('temp_flatpack.toml')

        # Store the bash script in a file
        with open('flatpack.sh', 'w') as f:
            f.write(bash_script_content)

        print("🎉 Bash script generated and saved as 'flatpack.sh'.")
        print(f"🔎 Location: {os.path.join(os.getcwd(), 'flatpack.sh')}")

        # Cleanup temporary TOML file
        os.remove('temp_flatpack.toml')
    else:
        print(f"❌ No flatpack.toml found in {directory_name}.\n")


def list_directories() -> str:
    """Retrieve and format a list of directories from the GitHub repository."""
    dirs = fetch_github_dirs()
    return "\n".join(dirs)


def main():
    """Main function that interprets user commands."""

    directory_name = sys.argv[2]
    display_disclaimer(directory_name)

    while True:
        user_response = input().strip().upper()
        if user_response == "YES":
            break
        elif user_response == "NO":
            print("❌ Installation aborted by user.")
            exit(0)
        else:
            print("❌ Invalid input. Please type 'YES' to accept or 'NO' to decline.")

    if len(sys.argv) < 2:
        print("Usage: flatpack.ai <command>")
        print("Available commands: help, install, list, test, version")
        return

    command = sys.argv[1]
    if command == "help":
        print("[HELP]")
    elif command == "install":
        if len(sys.argv) < 3:
            print("❌ Please specify a flatpack for the install command.")
            return
        install(directory_name)
    elif command == "list":
        print(list_directories())
    elif command == "test":
        print("[TEST]")
    elif command == "version":
        print("[VERSION]")
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
