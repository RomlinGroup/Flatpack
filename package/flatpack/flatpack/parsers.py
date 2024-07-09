import os
import toml


def is_valid_path(base_path, user_path):
    """
    Validates whether a given user_path is within the base_path.

    Parameters:
    - base_path: The base directory path.
    - user_path: The user-specified path to validate.

    Returns:
    - True if user_path is within base_path, False otherwise.
    """
    base_path = os.path.abspath(base_path)
    user_path = os.path.abspath(os.path.join(base_path, user_path))

    return os.path.commonpath([base_path, user_path]) == base_path


def is_url(s):
    """Check if a string is a URL."""
    return s.startswith('http://') or s.startswith('https://')


def check_command_availability(commands):
    """Generate bash snippets to check if each command in the provided list is available."""
    checks = [
        f"""
if [[ $IS_COLAB -eq 0 ]] && ! command -v {cmd} >/dev/null; then
  echo "{cmd} not found. Please install {cmd}."
  exit 1
fi
        """.strip() for cmd in commands
    ]
    return checks


def load_toml_config(file_path):
    """Load and validate the TOML configuration file."""
    base_dir = os.path.dirname(file_path)
    if not is_valid_path(base_dir, file_path):
        raise ValueError("Invalid file path")

    with open(file_path, 'r') as f:
        config = toml.load(f)

    model_name = config["environment"].get("model_name")
    if not model_name:
        raise ValueError("Missing model_name in flatpack.toml")

    return config, model_name


def generate_colab_check_script():
    """Generate the script to check for Google Colab environment."""
    return """
if [[ -d "/content" ]]; then
  # Detected Google Colab environment
  if command -v nvidia-smi &> /dev/null; then
    echo "Running in Google Colab with GPU"
    IS_COLAB=1
    DEVICE="cuda"
  else
    echo "Running in Google Colab with CPU only"
    IS_COLAB=1
    DEVICE="cpu"
  fi
else
  echo "Not running in Google Colab environment"
  IS_COLAB=0
fi
    """.strip()


def generate_venv_setup_script(env_name, build_prefix):
    """Generate the script to set up the virtual environment."""
    return f"""
handle_error() {{
    echo "Oops! Something went wrong."
    exit 1
}}

if [[ $IS_COLAB -ne 0 ]]; then
    apt install python3.10-venv
fi

echo "Checking for Python"
if [[ -x "$(command -v python3.11)" ]]; then
    PYTHON_CMD=python3.11
elif [[ -x "$(command -v python3.10)" ]]; then
    PYTHON_CMD=python3.10
elif [[ -x "$(command -v python3)" ]]; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

echo "Python command to be used: $PYTHON_CMD"

VENV_PATH="{env_name}/{build_prefix}"
echo "Creating the virtual environment at $VENV_PATH"

if ! $PYTHON_CMD -m venv --copies "$VENV_PATH"; then
    echo "Failed to create the virtual environment using $PYTHON_CMD"
    handle_error
else
    echo "Successfully created the virtual environment"
fi

# Ensuring the VENV_PYTHON path does not begin with a dot and is correctly formed
export VENV_PYTHON="$VENV_PATH/bin/python"
if [[ -f "$VENV_PYTHON" ]]; then
    echo "VENV_PYTHON is set correctly to $VENV_PYTHON"
    echo "Checking Python version in the virtual environment..."
    $VENV_PYTHON --version
else
    echo "VENV_PYTHON is set to $VENV_PYTHON, but this file does not exist"
    handle_error
fi

# Ensure pip is installed within the virtual environment
if [ ! -x "$VENV_PYTHON -m pip" ]; then
    echo "Installing pip within the virtual environment..."
    $VENV_PYTHON -m ensurepip
fi

# Set VENV_PIP variable to the path of pip within the virtual environment
export VENV_PIP="$VENV_PYTHON -m pip"
    """.strip()


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
    """Generate the script to clone Git repositories."""
    script = []
    for git in git_repos:
        from_source = git.get("from_source")
        to_destination = git.get("to_destination")
        branch = git.get("branch")
        setup_commands = git.get("setup_commands", [])

        if from_source and to_destination and branch:
            repo_path = f"{model_name}/{build_prefix}/{to_destination}"
            git_clone = f"""
echo "Cloning repository from: {from_source}"
git clone --depth=1 -b {branch} {from_source} {repo_path}
if [ $? -eq 0 ]; then
    echo "Git clone was successful."
else
    echo "Git clone failed."
    exit 1
fi

if [ -f {repo_path}/requirements.txt ]; then
    echo "Found requirements.txt, installing dependencies..."
    ${{VENV_PIP}} install -r {repo_path}/requirements.txt
else
    echo "No requirements.txt found."
fi
            """.strip()

            script.append(git_clone)

            for command in setup_commands:
                setup_command_script = f"""
echo "Running setup command: {command}"
pushd {repo_path}
{command}
if [ $? -eq 0 ]; then
    echo "Setup command '{command}' executed successfully."
else
    echo "Setup command '{command}' failed."
    exit 1
fi
popd
                """.strip()
                script.append(setup_command_script)

    return script


def download_files_script(items, model_name, build_prefix):
    """Generate the script to download datasets or files."""
    script = []
    for item in items:
        from_source, to_destination = item.get("from_source"), item.get("to_destination")

        if from_source and to_destination:
            destination_dir = os.path.dirname(f"./{model_name}/{build_prefix}/{to_destination}")
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
    Convert a TOML configuration to a bash script that sets up a python environment using venv and performs actions based on the TOML.
    Now ensures all directories, Git repositories, and other related files are created within a `/build` directory.

    Parameters:
    - file_path: The path to the TOML file.
    - env_name: Name of the virtual environment using venv.

    Returns:
    - Bash script as a string.
    """
    config, model_name = load_toml_config(file_path)
    build_prefix = "build"

    script = ["#!/bin/bash"]
    script.append(generate_colab_check_script())
    script.append(f"mkdir -p {model_name}/{build_prefix}")
    script.extend(check_command_availability(["curl", "wget", "git"]))

    unix_packages = config.get("packages", {}).get("unix", {})
    package_list_unix = list(unix_packages.keys())
    if package_list_unix:
        apt_install = f"""
OS=$(uname)
if [[ $IS_COLAB -eq 0 && "$OS" = "Linux" && -f /etc/debian_version ]]; then
    echo "Installing required Unix packages..."
    sudo apt update
    sudo apt install -y {' '.join(package_list_unix)}
fi
        """.strip()
        script.append(apt_install)

    script.append(generate_venv_setup_script(env_name, build_prefix))
    script.extend(create_directories_script(model_name, build_prefix, config.get("directories", {})))
    script.append(f"export model_name={model_name}")

    python_packages = config.get("packages", {}).get("python", {})
    package_list = [f"{package}=={version}" if version != "*" and version else package for package, version in
                    python_packages.items()]
    script.extend(install_python_packages_script(package_list))

    script.extend(clone_git_repositories_script(
        config.get("git", []), model_name, build_prefix
    ))

    script.extend(download_files_script(
        config.get("dataset", []), model_name, build_prefix
    ))

    script.extend(download_files_script(
        config.get("file", []), model_name, build_prefix
    ))

    script.extend(execute_run_commands_script(
        config.get("run", []), model_name, build_prefix
    ))

    return "\n".join(script)
