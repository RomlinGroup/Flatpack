import signal
import sys
import traceback

from functools import wraps


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__stderr__.write("\nProgram interrupted by user. Exiting gracefully...\n")
    else:
        sys.__stderr__.write("An unexpected error occurred:\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
    sys.exit(1)


def setup_exception_handling():
    sys.excepthook = handle_exception


def sigint_handler(signum, frame):
    sys.exit(0)


def setup_signal_handling():
    signal.signal(signal.SIGINT, sigint_handler)


def safe_exit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SystemExit:
            sys.__stderr__.write("Exiting the program...\n")
            sys.exit(0)
        except Exception as e:
            sys.__stderr__.write(f"An unexpected error occurred: {str(e)}\n")
            sys.exit(1)

    return wrapper
