import json
import os
import psutil
import signal
import subprocess

from datetime import datetime
from pathlib import Path


class AgentManager:
    _instance = None

    def __new__(cls, filepath='processes.json'):
        if cls._instance is None:
            cls._instance = super(AgentManager, cls).__new__(cls)
            cls._instance.processes = {}
            cls._instance.filepath = Path(filepath)
            cls._instance.load_processes()
        return cls._instance

    def save_processes(self):
        with self.filepath.open('w') as f:
            json.dump(self.processes, f, indent=4)

    def load_processes(self):
        if self.filepath.exists():
            with self.filepath.open('r') as f:
                self.processes = json.load(f)
        self.update_active_processes()

    def update_active_processes(self):
        active_processes = {}
        for pid, info in self.processes.items():
            if psutil.pid_exists(int(pid)):
                active_processes[pid] = info
            else:
                print(f"Process with PID {pid} no longer exists.")
        self.processes = active_processes
        self.save_processes()

    def spawn_agent(self, script_path):
        command = f"python {script_path}"
        pid = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).pid
        self.processes[str(pid)] = {
            'start_time': datetime.now().isoformat(),
            'script_path': script_path
        }
        self.save_processes()
        print(f"Spawned agent with PID {pid} running '{script_path}'")
        return pid

    def list_agents(self):
        self.update_active_processes()
        if not self.processes:
            print("No active agents.")
        else:
            print("Active agents:")
            for pid, details in self.processes.items():
                print(f"PID: {pid}, Script: {details['script_path']}, Start Time: {details['start_time']}")

    def terminate_agent(self, pid):
        try:
            os.kill(int(pid), signal.SIGTERM)
            print(f"Terminated agent with PID {pid}")
            del self.processes[str(pid)]
            self.save_processes()
        except OSError as e:
            print(f"Failed to terminate agent with PID {pid}: {e}")
        except KeyError:
            print(f"No agent with PID {pid} found.")
