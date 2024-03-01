import threading
import subprocess

threading.Thread(
    target=lambda: subprocess.run(['python3', '-m', 'llava.serve.controller', '--host', '0.0.0.0', '--port', '10000'],
                                  check=True), daemon=True).start()
