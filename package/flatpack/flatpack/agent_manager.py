import json
import os
import psutil
import signal
import socket
import subprocess

from datetime import datetime
from pathlib import Path


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


class AgentManager:
    def __init__(self, filepath=os.path.join(os.path.expanduser("~"), ".fpk_agents.json")):
        self.processes = {}
        self.filepath = Path(filepath)
        self.load_processes()

    def save_processes(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.processes, f, default=str)

    def load_processes(self):
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                self.processes = json.load(f)

                for pid in list(self.processes.keys()):
                    if not psutil.pid_exists(int(pid)):
                        del self.processes[pid]
        else:
            self.processes = {}

    def spawn_agent(self, script_path):
        free_port = find_free_port()
        env = os.environ.copy()
        env['AGENT_PORT'] = str(free_port)

        process = subprocess.Popen(["python", script_path], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pid = process.pid

        self.processes[str(pid)] = {
            'process': process.pid,
            'start_time': datetime.now(),
            'script_path': script_path,
            'port': free_port
        }
        self.save_processes()
        print(f"Spawned agent {pid} running on port {free_port} with '{script_path}'")
        return pid

    def list_agents(self):
        if not self.processes:
            print("No active agents.")
        else:
            print("Active agents:")
            for pid, details in self.processes.items():
                print(
                    f"PID: {pid}, Script: {details['script_path']}, Start Time: {details['start_time']}, Port: {details['port']}")

    def terminate_agent(self, pid):
        pid = str(pid)
        if pid in self.processes:
            try:
                os.kill(int(pid), signal.SIGTERM)
                print(f"Terminated agent {pid}")
                del self.processes[pid]
                self.save_processes()
            except OSError as e:
                print(f"Failed to terminate agent {pid}: {e}")
        else:
            print(f"No agent with PID {pid} found.")
