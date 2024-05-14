import toml


def parse_toml_to_venv_script(file_path: str, python_version="3.11.8", env_name="myenv") -> str:
    """
    Convert a TOML configuration to a bash script that sets up a python environment using venv and performs actions based on the TOML.
    Now ensures all directories, Git repositories, and other related files are created within a `/build` directory.

    Parameters:
    - file_path: The path to the TOML file.
    - python_version: The desired Python version (unused, but kept for function signature compatibility).
    - env_name: Name of the virtual environment using venv.

    Returns:
    - Bash script as a string.
    """

    def is_url(s):
        """Check if a string is a URL."""
        return s.startswith('http://') or s.startswith('https://')

    def check_command_availability(commands: list) -> list:
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

    # Load TOML configuration
    with open(file_path, 'r') as f:
        config = toml.load(f)

    model_name = config["environment"].get("model_name")
    if not model_name:
        raise ValueError("Missing model_name in flatpack.toml")

    build_prefix = "build"

    script = ["#!/bin/bash"]

    # Check if running in Google Colab and whether it's a GPU or CPU environment
    colab_check = """
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
    script.append(colab_check)

    # Ensure the build/model_name directory exists
    script.append(f"mkdir -p {model_name}/{build_prefix}")

    # Ensure required commands are available
    script.extend(check_command_availability(["curl", "wget", "git"]))

    # Install required Unix packages using apt, but only if not in Google Colab and on a Debian-based system
    unix_packages = config.get("packages", {}).get("unix", {})
    package_list_unix = [package for package in unix_packages.keys()]
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

    venv_setup = f"""
handle_error() {{
    echo "😟 Oops! Something went wrong."
    exit 1
}}

if [[ $IS_COLAB -ne 0 ]]; then
    apt install python3.10-venv
fi

echo "🐍 Checking for Python"
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

echo "🦄 Creating the virtual environment at {env_name}/{build_prefix}"

if ! $PYTHON_CMD -m venv "{env_name}/{build_prefix}"; then
    echo "❌ Failed to create the virtual environment using $PYTHON_CMD"
    handle_error
else
    echo "✅ Successfully created the virtual environment"
fi

# Ensuring the VENV_PYTHON path does not begin with a dot and is correctly formed
export VENV_PYTHON="{env_name}/{build_prefix}/bin/python"
if [[ -f "$VENV_PYTHON" ]]; then
    echo "✅ VENV_PYTHON is set correctly to $VENV_PYTHON"
    echo "🐍 Checking Python version in the virtual environment..."
    $VENV_PYTHON --version
else
    echo "❌ VENV_PYTHON is set to $VENV_PYTHON, but this file does not exist"
    handle_error
fi

# Ensure pip is installed within the virtual environment
if [ ! -x "$VENV_PYTHON -m pip" ]; then
    echo "Installing pip within the virtual environment..."
    $VENV_PYTHON -m ensurepip
fi

# Set VENV_PIP variable to the path of pip within the virtual environment
export VENV_PIP="$VENV_PYTHON -m pip"
    """
    script.append(venv_setup)

    # Create other directories within the build directory as per the TOML configuration
    directories_map = config.get("directories")
    if directories_map:
        for directory_path in directories_map.values():
            formatted_path = directory_path.lstrip('/').replace("home/content/", "")
            script.append(f"mkdir -p {model_name}/{build_prefix}/{formatted_path}")

    # Set model name as an environment variable
    script.append(f"export model_name={model_name}")

    # Install python packages
    packages = config.get("packages", {}).get("python", {})
    package_list = [f"{package}=={version}" if version != "*" and version else package for package, version in
                    packages.items()]
    if package_list:
        script.append(f"$VENV_PIP install {' '.join(package_list)}")

    # Clone required git repositories
    for git in config.get("git", []):
        from_source, to_destination, branch = git.get("from_source"), git.get("to_destination"), git.get("branch")
        if from_source and to_destination and branch:
            repo_path = f"{model_name}/{build_prefix}/{to_destination}"
            git_clone = f"""
echo "Cloning repository from: {from_source}"
git clone -b {branch} {from_source} {repo_path}
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

    # Download datasets or files
    for item_type in ["dataset", "file"]:
        for item in config.get(item_type, []):
            from_source, to_destination = item.get("from_source"), item.get("to_destination")

            if from_source and to_destination:

                if is_url(from_source):
                    download_command = f"curl -s -L {from_source} -o ./{model_name}/{build_prefix}/{to_destination}"
                    script.append(download_command)
                else:
                    download_command = f"cp -r ./{model_name}/{from_source} ./{model_name}/{build_prefix}/{to_destination}"
                    script.append(download_command)

    # Execute specified run commands
    run_vec = config.get("run", [])
    for run in run_vec:
        command, file = run.get("command"), run.get("file")
        if command and file:
            prepended_file = f"./{model_name}/{build_prefix}/" + file
            script.append(f"{command} {prepended_file}")

    return "\n".join(script)
