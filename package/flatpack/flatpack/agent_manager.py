import json
import multiprocessing
import os
import psutil
import signal
import subprocess

from datetime import datetime
from pathlib import Path


class AgentManager:
    def __init__(self, filepath='/content/processes.json'):
        self.processes = {}
        self.filepath = filepath
        self.load_processes()

    def save_processes(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.processes, f, indent=4)
            f.flush()
            os.fsync(f.fileno())

    def load_processes(self):
        try:
            with open(self.filepath, 'r') as f:
                self.processes = json.load(f)
        except FileNotFoundError:
            self.processes = {}

        self.processes = {pid: details for pid, details in self.processes.items() if psutil.pid_exists(int(pid))}
        self.save_processes()

    def spawn_agent(self, script_path):
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
        process = multiprocessing.Process(target=self.run_script, args=(script_path,))
        process.daemon = True
        process.start()
        pid = process.pid
        self.processes[str(pid)] = {
            'process': process,
            'start_time': datetime.now().isoformat(),
            'script_path': script_path
        }
        self.save_processes()
        print(f"Spawned agent {pid} running '{script_path}'")
        return pid

    def run_script(self, script_path):
        os.system(f'python {script_path}')

    def list_agents(self):
        if not self.processes:
            print("No active agents.")
        else:
            print("Active agents:")
            for pid, details in self.processes.items():
                print(f"PID: {pid}, Script: {details['script_path']}, Start Time: {details['start_time']}")

    def terminate_agent(self, pid):
        process = self.processes.get(str(pid), {}).get('process')
        if process:
            process.terminate()
            process.join()
            print(f"Terminated agent {pid}")
            del self.processes[pid]
            self.save_processes()
        else:
            print(f"No agent with PID {pid} found.")
