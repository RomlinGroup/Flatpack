import errno
import os
import resource
import selectors
import signal
import sys
import time
import traceback

MEMORY_LIMIT = 16 * 2 ** 30
CPU_LIMIT = 7200

GLOBAL_NAMESPACE = {}

input_pipe = None


def setup_resources():
    if sys.platform != 'darwin':
        try:
            resource.setrlimit(
                resource.RLIMIT_AS,
                (MEMORY_LIMIT, MEMORY_LIMIT)
            )
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (CPU_LIMIT, CPU_LIMIT)
            )
        except Exception as e:
            print(f"Warning: Could not set resource limits: {e}",
                  file=sys.stderr)


def setup_input_pipe():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        input_pipe_path = os.path.join(script_dir, "python_input")

        def open_with_timeout(path, mode, timeout=5):
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    return open(path, mode, buffering=1)
                except OSError as e:
                    if e.errno != errno.EINTR:
                        raise
                time.sleep(0.1)
            raise TimeoutError(f"Failed to open pipe {path} after {timeout}s")

        return open_with_timeout(input_pipe_path, 'r')
    except Exception as e:
        print(f"Failed to setup input pipe: {e}", file=sys.stderr)
        sys.exit(1)


def execute_code(code):
    old_stdout = sys.stdout
    old_stdin = sys.stdin
    try:
        sys.stdout = sys.__stdout__
        sys.stdin = sys.__stdin__
        os.setpgrp()

        def my_input(prompt=""):
            try:
                print("READY_FOR_INPUT")
                sys.stdout.flush()
                print(prompt, end="", flush=True)
                line = input_pipe.readline().strip()
                return line
            except Exception as e:
                print(f"Input error: {e}", file=sys.stderr)
                return ""

        GLOBAL_NAMESPACE["input"] = my_input
        exec(code, GLOBAL_NAMESPACE)

        sys.stdout.flush()
        sys.stderr.flush()
        print("EXECUTION_COMPLETE")
        return True

    except Exception as e:
        traceback.print_exc()
        return False

    finally:
        sys.stdout = old_stdout
        sys.stdin = old_stdin


def signal_handler(signum, frame):
    print(f"Python executor received signal {signum}. Exiting.")
    sys.exit(1)


def handle_stream_input(stream_type, line, code_block):
    if stream_type == "code":

        if line.strip() == "__EXIT_PYTHON_EXECUTOR__":
            print("Received exit signal. Cleaning up...")
            sys.stdout.flush()
            sys.exit(0)

        if line.strip() == "__END_CODE_BLOCK__":
            if code_block:
                try:
                    execute_code(''.join(code_block))
                except Exception as e:
                    print(f"Error executing code block: {e}", file=sys.stderr)
                    traceback.print_exc()
                return True

        elif line.strip() == "READY_FOR_INPUT":
            return True
        else:
            code_block.append(line)
            return False

    elif stream_type == "input":
        if "input" in GLOBAL_NAMESPACE:
            input_line = line.strip()

            try:
                GLOBAL_NAMESPACE["input"](input_line)
                return True
            except Exception as e:
                print(f"Error calling input handler: {e}", file=sys.stderr)
                traceback.print_exc()
                return False
        else:
            print("No input handler available")
            return False

    return False


def main():
    global input_pipe

    setup_resources()
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        input_pipe = setup_input_pipe()
        selector = selectors.DefaultSelector()

        with selector:
            selector.register(sys.stdin, selectors.EVENT_READ, "code")
            selector.register(input_pipe, selectors.EVENT_READ, "input")

            while True:
                try:
                    code_block = []

                    while True:
                        try:
                            events = selector.select(timeout=1.0)

                            if not events:
                                continue

                            for event, mask in events:
                                stream = event.fileobj
                                stream_type = event.data

                                try:
                                    line = stream.readline()
                                    if not line:
                                        continue

                                    if handle_stream_input(stream_type, line,
                                                           code_block):
                                        break

                                except Exception as e:
                                    print(f"Error handling stream: {e}",
                                          file=sys.stderr)

                        except Exception as e:
                            print(f"Error in inner loop: {e}", file=sys.stderr)
                            break

                except Exception as e:
                    print(f"Error in outer loop: {e}", file=sys.stderr)
                    time.sleep(1)

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
