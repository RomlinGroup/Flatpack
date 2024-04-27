import httpx


class SessionManager:
    def __enter__(self):
        """Create an HTTP session."""
        self.session = httpx.Client()
        return self.session

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the HTTP session."""
        self.session.close()
