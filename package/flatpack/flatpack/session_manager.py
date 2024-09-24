import httpx
import threading


class SessionManager:
    def __init__(self):
        self._session = None
        self._lock = threading.Lock()

    def __enter__(self):
        """Create an HTTP session."""
        with self._lock:
            if self._session is None:
                self._session = httpx.Client()
            return self._session

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the HTTP session."""
        with self._lock:
            if self._session:
                try:
                    self._session.close()
                except Exception as e:
                    print(f"Error closing session: {e}")
                finally:
                    self._session = None

    def get_session(self):
        """Get the current session or create a new one if it doesn't exist."""
        with self._lock:
            if self._session is None:
                self._session = httpx.Client()
            return self._session
