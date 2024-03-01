import threading
import subprocess

command = [
    'python3', '-m', 'llava.serve.model_worker',
    '--host', '0.0.0.0',
    '--controller', 'http://localhost:10000',
    '--port', '40000',
    '--worker', 'http://localhost:40000',
    '--model-path', 'NousResearch/Obsidian-3B-V0.5',
]

threading.Thread(target=lambda: subprocess.run(command, check=True, shell=False), daemon=True).start()
