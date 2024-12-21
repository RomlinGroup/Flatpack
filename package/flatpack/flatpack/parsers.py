import os
import toml

from textwrap import dedent


def is_valid_path(base_path, user_path):
    """
    Validates whether a given user_path is within the base_path.

    Parameters:
    - base_path: The base directory path (str).
    - user_path: The user-specified path to validate (str).

    Returns:
    - True if user_path is within base_path, False otherwise.
    Raises:
    - ValueError if either path is None or empty
    """
    if not base_path or not user_path:
        raise ValueError("Base path and user path must not be None or empty")

    base_path = os.path.abspath(base_path)
    user_path = os.path.abspath(os.path.join(base_path, user_path))

    return os.path.commonpath([base_path, user_path]) == base_path


def is_url(s):
    """Check if a string is a URL."""
    return s.startswith('http://') or s.startswith('https://')


def check_command_availability(commands):
    """Generate bash snippets to check if each command in the provided list is available."""
    checks = [
        dedent(f"""\
            if ! command -v {cmd} >/dev/null; then
                echo "{cmd} not found. Please install {cmd}."
                exit 1
            fi
        """).strip() for cmd in commands
    ]

    return checks


def load_toml_config(file_path):
    """
    Load and validate the TOML configuration file.

    Parameters:
    - file_path: Path to the TOML file (str)

    Returns:
    - tuple: (config dict, model_name str, python_version str)

    Raises:
    - ValueError: If file_path is invalid or required fields are missing
    - FileNotFoundError: If the TOML file doesn't exist
    """
    if not file_path:
        raise ValueError("File path cannot be None or empty")

    base_dir = os.path.dirname(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")

    if not is_valid_path(base_dir, file_path):
        raise ValueError(f"Invalid file path: {file_path}")

    with open(file_path, 'r') as f:
        config = toml.load(f)

    if "environment" not in config:
        raise ValueError("Missing 'environment' section in flatpack.toml")

    model_name = config["environment"].get("model_name")
    if not model_name:
        raise ValueError("Missing model_name in flatpack.toml")

    python_version = config["environment"].get("python_version")
    return config, model_name, python_version


def generate_venv_setup_script(env_name, build_prefix, python_version=None):
    python_cmd = f"python{python_version}" if python_version else "python3"
    return dedent(f"""\
        echo "Checking for Python"

        if [[ "{python_version}" != "None" ]] && [[ -x "$(command -v python{python_version})" ]]; then
            PYTHON_CMD=python{python_version}
        elif [[ -x "$(command -v python3.12)" ]]; then
            PYTHON_CMD=python3.12
        elif [[ -x "$(command -v python3.11)" ]]; then
            PYTHON_CMD=python3.11
        elif [[ -x "$(command -v python3.10)" ]]; then
            PYTHON_CMD=python3.10
        elif [[ -x "$(command -v python3)" ]]; then
            PYTHON_CMD=python3
        else
            PYTHON_CMD=python
        fi

        echo "Python command to be used: $PYTHON_CMD"
        umask 022

        VENV_PATH="{env_name}/{build_prefix}"

        echo "Creating the virtual environment at $VENV_PATH"

        if ! $PYTHON_CMD -m venv --copies --without-pip "$VENV_PATH"; then
            echo "Failed to create the virtual environment using $PYTHON_CMD"
            exit 1
        else
            echo "Successfully created the virtual environment"
            chmod -R go-w "$VENV_PATH"
        fi

        export VENV_PYTHON="$VENV_PATH/bin/python"

        if [[ -f "$VENV_PYTHON" ]]; then
            echo "VENV_PYTHON is set correctly to $VENV_PYTHON"
            echo "Checking Python version in the virtual environment..."
            $VENV_PYTHON --version
        else
            echo "VENV_PYTHON is set to $VENV_PYTHON, but this file does not exist"
            exit 1
        fi

        echo "Installing pip within the virtual environment..."
        curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py

        $VENV_PYTHON get-pip.py

        rm get-pip.py

        export VENV_PIP="$VENV_PYTHON -m pip"
        """).strip()


def create_directories_script(model_name, build_prefix, directories_map):
    """Generate the script to create directories."""
    script = []
    if directories_map:
        for directory_path in directories_map.values():
            formatted_path = directory_path.lstrip('/').replace("home/content/", "")
            script.append(f"mkdir -p {model_name}/{build_prefix}/{formatted_path}")
    return script


def install_python_packages_script(package_list):
    """Generate the script to install Python packages."""
    if package_list:
        return [f"$VENV_PIP install {' '.join(package_list)}"]
    return []


def clone_git_repositories_script(git_repos, model_name, build_prefix):
    script = []

    for git in git_repos:
        from_source = git.get("from_source")
        to_destination = git.get("to_destination")
        branch = git.get("branch")
        setup_commands = git.get("setup_commands", [])
        requirements_file = git.get("requirements_file", "requirements.txt")

        if from_source and to_destination and branch:
            repo_path = f"{model_name}/{build_prefix}/{to_destination}"
            git_clone = dedent(f"""\
                echo "Cloning repository from: {from_source}"
                git clone --depth=1 {from_source} -b {branch} {repo_path}

                if [ $? -eq 0 ]; then
                    echo "Git clone was successful."

                    cd {repo_path} > /dev/null

                    current_branch=$(git rev-parse --abbrev-ref HEAD)

                    if [ "$current_branch" = "{branch}" ]; then
                        echo "Confirmed: Successfully cloned the correct branch: {branch}"
                    else
                        echo "Warning: Cloned branch ($current_branch) does not match the intended branch ({branch})"
                        exit 1
                    fi

                    cd - > /dev/null
                else
                    echo "Git clone failed."
                    exit 1
                fi

                if [ -f {repo_path}/{requirements_file} ]; then
                    echo "Found {requirements_file}, installing dependencies..."
                    ${{VENV_PIP}} install -r {repo_path}/{requirements_file}
                else
                    echo "No {requirements_file} found."
                fi
                """).strip()

            script.append(git_clone)

            for command in setup_commands:
                setup_command_script = dedent(f"""\
                    echo "Running setup command: {command}"

                    pushd {repo_path}

                    {command}

                    if [ $? -eq 0 ]; then
                        echo "Setup command '{command}' executed successfully."
                    else
                        echo "Setup command '{command}' failed."
                        exit 1
                    fi

                    popd > /dev/null
                    """).strip()

                script.append(setup_command_script)

    return script


def download_files_script(items, model_name, build_prefix):
    """
    Generate the script to download datasets or files.

    Parameters:
    - items: List of download items (list of dict)
    - model_name: Name of the model (str)
    - build_prefix: Build directory prefix (str)

    Returns:
    - list: Shell commands to execute
    """
    if not model_name or not build_prefix:
        raise ValueError("model_name and build_prefix must not be None or empty")

    script = []
    for item in items or []:
        from_source = item.get("from_source")
        to_destination = item.get("to_destination")

        if not from_source or not to_destination:
            continue

        destination_dir = os.path.dirname(f"./{model_name}/{build_prefix}/{to_destination}")
        if destination_dir:
            script.append(f"mkdir -p {destination_dir}")

        if is_url(from_source):
            download_command = f"curl -s -L {from_source} -o ./{model_name}/{build_prefix}/{to_destination}"
            script.append(download_command)
        else:
            download_command = f"cp -r ./{model_name}/{from_source} ./{model_name}/{build_prefix}/{to_destination}"
            script.append(download_command)
    return script


def execute_run_commands_script(run_vec, model_name, build_prefix):
    """Generate the script to execute specified run commands."""
    script = []
    for run in run_vec:
        command, file = run.get("command"), run.get("file")
        if command and file:
            prepended_file = f"./{model_name}/{build_prefix}/" + file
            script.append(f"{command} {prepended_file}")
    return script


def parse_toml_to_venv_script(file_path: str, env_name="myenv") -> str:
    """
    Convert a TOML configuration to a bash script that sets up a python environment.

    Parameters:
    - file_path: Path to the TOML file (str)
    - env_name: Name of the virtual environment (str, default="myenv")

    Returns:
    - str: Generated bash script

    Raises:
    - ValueError: If file_path is invalid or required fields are missing
    """
    if not file_path:
        raise ValueError("file_path cannot be None or empty")
    if not env_name:
        raise ValueError("env_name cannot be None or empty")

    config, model_name, python_version = load_toml_config(file_path)
    build_prefix = "build"

    script = ["#!/bin/bash", f"mkdir -p {model_name}/{build_prefix}"]
    script.extend(check_command_availability(["curl", "wget", "git"]))

    unix_packages = config.get("packages", {}).get("unix", {}) or {}
    package_list_unix = list(unix_packages.keys())

    if package_list_unix:
        apt_install = dedent(f"""\
           OS=$(uname)
           if [[ "$OS" = "Linux" && -f /etc/debian_version ]]; then
               echo "Installing required Unix packages..."
               sudo apt update
               sudo apt install -y {' '.join(package_list_unix)}
           fi
           """).strip()

        script.append(apt_install)

    script.append(generate_venv_setup_script(env_name, build_prefix, python_version))
    script.extend(create_directories_script(model_name, build_prefix, config.get("directories", {})))
    script.append(f"export model_name={model_name}")

    python_packages = config.get("packages", {}).get("python", {}) or {}
    package_list = [f"{package}=={version}" if version != "*" and version else package
                    for package, version in python_packages.items()]
    script.extend(install_python_packages_script(package_list))

    script.extend(clone_git_repositories_script(
        config.get("git", []) or [], model_name, build_prefix
    ))

    script.extend(download_files_script(
        config.get("dataset", []) or [], model_name, build_prefix
    ))

    script.extend(download_files_script(
        config.get("file", []) or [], model_name, build_prefix
    ))

    script.extend(execute_run_commands_script(
        config.get("run", []) or [], model_name, build_prefix
    ))

    return "\n".join(script)
