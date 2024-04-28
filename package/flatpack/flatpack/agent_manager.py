import json
import os
import psutil
import signal
import subprocess

from datetime import datetime
from pathlib import Path


class AgentManager:
    def __init__(self, filepath='processes.json'):
        self.processes = {}
        self.filepath = Path(filepath)
        self.load_processes()

    def save_processes(self):
        """Save the current state of processes to a JSON file."""
        with self.filepath.open('w') as f:
            json.dump(self.processes, f, indent=4)

    def load_processes(self):
        """Load and update the state of processes from the JSON file."""
        if self.filepath.exists():
            with self.filepath.open('r') as f:
                self.processes = json.load(f)
            active_processes = {}
            for pid, info in self.processes.items():
                if psutil.pid_exists(int(pid)):
                    active_processes[pid] = info
                else:
                    print(f"Process with PID {pid} no longer exists.")
            self.processes = active_processes
            self.save_processes()

    def spawn_agent(self, script_path):
        """Spawn a new agent to run a given script."""
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
        """List all currently active agents."""
        if not self.processes:
            print("No active agents.")
        else:
            print("Active agents:")
            for pid, details in self.processes.items():
                print(f"PID: {pid}, Script: {details['script_path']}, Start Time: {details['start_time']}")

    def terminate_agent(self, pid):
        """Terminate an agent by PID."""
        try:
            os.kill(int(pid), signal.SIGTERM)
            print(f"Terminated agent with PID {pid}")
            del self.processes[str(pid)]
            self.save_processes()
        except OSError as e:
            print(f"Failed to terminate agent with PID {pid}: {e}")
        except KeyError:
            print(f"No agent with PID {pid} found.")
