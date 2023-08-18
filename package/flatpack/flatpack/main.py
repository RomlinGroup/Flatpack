import requests
import sys

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
    else:
        print(f"No flatpack.toml found in {directory_name}.\n")

def list_directories():
    dirs = fetch_github_dirs()
    return "\n".join(dirs)

def main():
    if len(sys.argv) < 2:
        print("Usage: flatpack.ai <command>")
        print("Available commands: help, list, test, install")
        return

    command = sys.argv[1]
    if command == "help":
        print("[HELP]")
    elif command == "install":
        if len(sys.argv) < 3:
            print("Please specify a directory name for the install command.")
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
