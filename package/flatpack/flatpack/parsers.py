import toml


def parse_toml_to_venv_script(file_path: str, python_version="3.10.12", env_name="myenv") -> str:
    """
    Convert a TOML configuration to a bash script that sets up a python environment using venv and performs actions based on the TOML.

    Parameters:
    - file_path: The path to the TOML file.
    - python_version: The desired Python version (unused, but kept for function signature compatibility).
    - env_name: Name of the virtual environment using venv.

    Returns:
    - Bash script as a string.
    """

    def check_command_availability(commands: list) -> list:
        """Generate bash snippets to check if each command in the provided list is available."""
        checks = [f"""
if [[ $IS_COLAB -eq 0 ]] && ! command -v {cmd} >/dev/null; then
  echo "{cmd} not found. Please install {cmd}."
  exit 1
fi
        """.strip() for cmd in commands]
        return checks

    # Load TOML configuration
    with open(file_path, 'r') as f:
        config = toml.load(f)

    model_name = config["environment"].get("model_name")
    if not model_name:
        raise ValueError("Missing model_name in flatpack.toml")

    script = ["#!/bin/bash"]

    # Check if running in Google Colab
    colab_check = """
if [[ "${COLAB_GPU}" == "1" ]]; then
  echo "Running in Google Colab environment"
  IS_COLAB=1
else
  echo "Not running in Google Colab environment"
  IS_COLAB=0
fi
    """.strip()
    script.append(colab_check)

    # Bash check for directory existence
    script.append(f"""
if [[ $IS_COLAB -eq 1 ]]; then
    if [ ! -d "{model_name}" ]; then
        mkdir -p {model_name}
    else
        echo 'Directory {model_name} already exists in Google Colab. Moving on.'
    fi
fi
    """)

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

    # Setup venv environment
    venv_setup = f"""
handle_error() {{
    echo "😟 Oops! Something went wrong."
    exit 1
}}

if [[ $IS_COLAB -eq 0 ]]; then
    # Check if python3 is available
    if command -v python3 &>/dev/null; then
        PYTHON_CMD=python3
    else
        PYTHON_CMD=python
    fi

    if [ ! -d "{env_name}" ]; then
        $PYTHON_CMD -m venv {env_name} || handle_error
    fi

    export VENV_PYTHON={env_name}/bin/python
fi
    """.strip()
    script.append(venv_setup)

    # Create other directories as per the TOML configuration
    directories_map = config.get("directories")
    if directories_map:
        for directory_path in directories_map.values():
            formatted_path = directory_path.lstrip('/').replace("home/content/", "")
            script.append(f"mkdir -p ./{model_name}/{formatted_path}")

    # Set model name as an environment variable
    script.append(f"export model_name={model_name}")

    # Install python packages
    packages = config.get("packages", {}).get("python", {})
    package_list = [f"{package}=={version}" if version != "*" and version else package for package, version in
                    packages.items()]
    if package_list:
        script.append(f"$VENV_PYTHON -m pip install {' '.join(package_list)}")

    # Clone required git repositories
    for git in config.get("git", []):
        from_source, to_destination, branch = git.get("from_source"), git.get("to_destination"), git.get("branch")
        if from_source and to_destination and branch:
            repo_path = f"./{model_name}/{to_destination.replace('/home/content/', '')}"
            git_clone = f"""
echo "Cloning repository from: {from_source}"
git clone -b {branch} {from_source} {repo_path}
if [ $? -eq 0 ]; then
    echo "Git clone was successful."
    flatpack callback git-success
else
    echo "Git clone failed."
    exit 1
fi
if [ -f {repo_path}/requirements.txt ]; then
    echo "pwd: $(pwd)"
    echo "repo_path: {repo_path}"
    echo "Found requirements.txt, installing dependencies..."
    ${{VENV_PYTHON}} pip install -r {repo_path}/requirements.txt
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
                script.append(f"curl -L {from_source} -o ./{model_name}/{to_destination.replace('/home/content/', '')}")

    # Execute specified run commands
    run_vec = config.get("run", [])
    for run in run_vec:
        command, args = run.get("command"), run.get("args")
        if command and args:
            replaced_args = args.replace("/home/content/", f"./{model_name}/")
            script.append(f"{command} {replaced_args}")

    return "\n".join(script)
