import asyncio
import signal
import sys
import traceback

from functools import wraps


def setup_signal_handling(loop=None):
    """Sets up signal handlers for SIGINT and SIGTERM."""
    if loop is None:
        loop = asyncio.get_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(s, loop)))


def setup_exception_handling():
    """Sets up a global exception handler."""
    sys.excepthook = handle_exception


def handle_exception(exc_type, exc_value, exc_traceback):
    """Handles uncaught exceptions."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__stderr__.write("\nProgram interrupted by user. Exiting gracefully...\n")
        sys.exit(0)
    else:
        sys.__stderr__.write("An unexpected error occurred:\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        sys.exit(1)


def handle_asyncio_exception(loop, context):
    """Handles exceptions in asyncio tasks."""
    msg = context.get("exception", context["message"])
    task = context.get("task")
    if task:
        sys.__stderr__.write(f"Caught asyncio exception in task {task.get_name()}: {msg}\n")
    else:
        sys.__stderr__.write(f"Caught asyncio exception: {msg}\n")


cleanup_callbacks = []


def register_cleanup(callback):
    """Registers a cleanup callback to be executed on shutdown."""
    cleanup_callbacks.append(callback)


async def shutdown(signal, loop):
    """Initiates a graceful shutdown process."""
    sys.__stderr__.write(f"Received exit signal {signal.name}. Shutting down...\n")
    for callback in cleanup_callbacks:
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback()
            else:
                callback()
        except Exception as e:
            sys.__stderr__.write(f"Error during cleanup: {e}\n")

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        sys.__stderr__.write(f"Cancelling {len(tasks)} outstanding tasks...\n")
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                sys.__stderr__.write(f"Exception during shutdown: {result}\n")
    loop.stop()


def safe_exit(func):
    """Decorator to ensure graceful exit on exceptions."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            sys.__stderr__.write(f"An unexpected error occurred: {str(e)}\n")
            traceback.print_exc()
            sys.exit(1)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            sys.__stderr__.write(f"An unexpected error occurred: {str(e)}\n")
            traceback.print_exc()
            sys.exit(1)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
