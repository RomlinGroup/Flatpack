import requests
import sys
import toml

def parse_toml_to_pyenv_script(file_path: str, python_version="3.11.3", env_name="myenv") -> str:

    def check_command_availability(commands: list) -> list:
        checks = []
        for cmd in commands:
            cmd_check = f"""
if [[ $IS_COLAB -eq 0 ]] && ! command -v {cmd} >/dev/null; then
  echo "{cmd} not found. Please install {cmd}."
  exit 1
fi
"""
            checks.append(cmd_check.strip())
        return checks

    with open(file_path, 'r') as f:
        config = toml.load(f)

    model_name = config["environment"].get("model_name")
    if not model_name:
        raise ValueError("Missing model_name in flatpack.toml")

    script = ["#!/bin/bash"]

    colab_check = """
if [[ "${COLAB_GPU}" == "1" ]]; then
  echo "Running in Google Colab environment"
  IS_COLAB=1
else
  echo "Not running in Google Colab environment"
  IS_COLAB=0
fi
"""
    script.append(colab_check.strip())
    script.extend(check_command_availability(["pyenv", "wget", "git"]))

    pyenv_setup = f"""
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if [[ $IS_COLAB -eq 0 ]]; then
 if ! pyenv versions | grep -q {python_version}; then
  pyenv install {python_version}
fi
 if ! pyenv virtualenvs | grep -q {env_name}; then
  pyenv virtualenv {python_version} {env_name}
fi
 pyenv activate {env_name}
fi
"""
    script.append(pyenv_setup.strip())

    script.append(f"mkdir -p ./{model_name}")

    directories_map = config.get("directories")
    if directories_map:
        for directory_path in directories_map.values():
            formatted_directory_path = directory_path.lstrip('/')
            without_home_content = formatted_directory_path.replace("home/content/", "")
            script.append(f"mkdir -p ./{model_name}/{without_home_content}")

    script.append(f"export model_name={model_name}")

    packages = config.get("packages", {}).get("python", {})
    package_list = [f"{package}=={version}" if version != "*" and version else package for package, version in packages.items()]
    if package_list:
        script.append(f"python -m pip install {' '.join(package_list)}")

    for git in config.get("git", []):
        from_source = git.get("from_source")
        to_destination = git.get("to_destination")
        branch = git.get("branch")
        if from_source and to_destination and branch:
            repo_path = f"./{model_name}/{to_destination.replace('/home/content/', '')}"
            git_clone = f"""
echo 'Cloning repository from: {from_source}'
git clone -b {branch} {from_source} {repo_path}
if [ -f {repo_path}/requirements.txt ]; then
  echo 'Found requirements.txt, installing dependencies...'
  cd {repo_path} || exit
  python -m pip install -r requirements.txt
  cd - || exit
else
  echo 'No requirements.txt found.'
fi
"""
            script.append(git_clone.strip())

    for item_type in ["dataset", "file"]:
        for item in config.get(item_type, []):
            from_source = item.get("from_source")
            to_destination = item.get("to_destination")
            if from_source and to_destination:
                script.append(f"wget {from_source} -O ./{model_name}/{to_destination.replace('/home/content/', '')}")

    run_vec = config.get("run", [])
    for run in run_vec:
        command = run.get("command")
        args = run.get("args")
        if command and args:
            replaced_args = args.replace("/home/content/", f"./{model_name}/")
            script.append(f"{command} {replaced_args}")

    return "\n".join(script)

def fetch_flatpack_toml_from_dir(directory_name):
    base_url = "https://raw.githubusercontent.com/romlingroup/flatpack-ai/main/warehouse"
    toml_url = f"{base_url}/{directory_name}/flatpack.toml"
    response = requests.get(toml_url)
    if response.status_code != 200:
        return None
    return response.text

def fetch_github_dirs():
    url = "https://api.github.com/repos/romlingroup/flatpack-ai/contents/warehouse"
    response = requests.get(url)
    if response.status_code != 200:
        return ["Error fetching data from GitHub"]
    data = response.json()
    directories = [item['name'] for item in data if item['type'] == 'dir']
    return sorted(directories)

def install(directory_name):
    # Check if the directory exists
    existing_dirs = fetch_github_dirs()
    if directory_name not in existing_dirs:
        print(f"Error: The directory '{directory_name}' does not exist.")
        return

    toml_content = fetch_flatpack_toml_from_dir(directory_name)
    if toml_content:
        print(f"Contents of flatpack.toml in {directory_name}:\n{toml_content}\n")

        # Save the toml content to a temporary file
        with open('temp_flatpack.toml', 'w') as f:
            f.write(toml_content)

        # Generate the bash script using the toml file
        bash_script_content = parse_toml_to_pyenv_script('temp_flatpack.toml')

        # Save the bash script to a file
        with open('flatpack.sh', 'w') as f:
            f.write(bash_script_content)

        print("Bash script generated and saved as 'flatpack.sh'. Execute the script manually when ready.")

        # Remove the temporary toml file
        os.remove('temp_flatpack.toml')
    else:
        print(f"No flatpack.toml found in {directory_name}.\n")

def list_directories():
    dirs = fetch_github_dirs()
    return "\n".join(dirs)

def main():
    if len(sys.argv) < 2:
        print("Usage: flatpack.ai <command>")
        print("Available commands: help, install, list, test")
        return

    command = sys.argv[1]
    if command == "help":
        print("[HELP]")
    elif command == "install":
        if len(sys.argv) < 3:
            print("Please specify a flatpack for the install command.")
            return
        directory_name = sys.argv[2]
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
