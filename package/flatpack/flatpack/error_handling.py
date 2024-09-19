import asyncio
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


def setup_exception_handling():
    sys.excepthook = handle_exception


def sigint_handler(signum, frame):
    raise KeyboardInterrupt()


def setup_signal_handling():
    signal.signal(signal.SIGINT, sigint_handler)


def safe_exit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            if asyncio.iscoroutinefunction(func):
                return asyncio.run(func(*args, **kwargs))
            else:
                return func(*args, **kwargs)
        except SystemExit:
            sys.__stderr__.write("Exiting the program...\n")
            sys.exit(0)
        except KeyboardInterrupt:
            sys.__stderr__.write("\nProgram interrupted by user. Exiting gracefully...\n")
            sys.exit(0)
        except Exception as e:
            sys.__stderr__.write(f"An unexpected error occurred: {str(e)}\n")
            traceback.print_exc()
            sys.exit(1)

    return wrapper


async def shutdown(signal, loop):
    """Cleanup tasks tied to the service's shutdown."""
    sys.__stderr__.write(f"Received exit signal {signal.name}...\n")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

    for task in tasks:
        task.cancel()

    sys.__stderr__.write(f"Cancelling {len(tasks)} outstanding tasks\n")
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


def handle_asyncio_exception(loop, context):
    msg = context.get("exception", context["message"])
    sys.__stderr__.write(f"Caught asyncio exception: {msg}\n")
