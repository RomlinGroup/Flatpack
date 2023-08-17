import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: flatpack.ai <command>")
        print("Available commands: help, list, test")
        return

    command = sys.argv[1]
    if command == "help":
        print("[HELP]")
    elif command == "list":
        print("[LIST]")
    elif command == "test":
        print("[TEST]")
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
