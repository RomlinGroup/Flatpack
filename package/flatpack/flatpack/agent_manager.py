import json
import multiprocessing
import os
import psutil
import signal
import subprocess

from datetime import datetime
from pathlib import Path


class AgentManager:
    def __init__(self, filepath='processes.json'):
        self.processes = {}
        self.filepath = filepath
        self.load_processes()

    def save_processes(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.processes, f, default=str)

    def load_processes(self):
        try:
            with open(self.filepath, 'r') as f:
                self.processes = json.load(f)

                for pid in list(self.processes.keys()):
                    if not psutil.pid_exists(int(pid)):
                        del self.processes[pid]
        except FileNotFoundError:
            self.processes = {}

    def spawn_agent(self, script_path):
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
        process = multiprocessing.Process(target=self.run_script, args=(script_path))
        process.daemon = True
        process.start()
        pid = process.pid
        self.processes[str(pid)] = {
            'process': process,
            'start_time': datetime.now(),
            'script_path': script_path
        }
        self.save_processes()
        print(f"Spawned agent {pid} running '{script_path}'")
        return pid

    def run_script(self, script_path):
        try:
            result = subprocess.run(['python', script_path], capture_output=True, text=True)
            print(f"Script output: {result.stdout}")
            if result.stderr:
                print(f"Script error: {result.stderr}")
        except Exception as e:
            print(f"Error running script {script_path}: {e}")

    def list_agents(self):
        if not self.processes:
            print("No active agents.")
        else:
            print("Active agents:")
            for pid, details in self.processes.items():
                print(f"PID: {pid}, Script: {details['script_path']}, Start Time: {details['start_time']}")

    def terminate_agent(self, pid):
        pid = str(pid)
        if pid in self.processes:
            try:
                process = self.processes[pid]['process']
                process.terminate()
                process.join()
                print(f"Terminated agent {pid}")
                del self.processes[pid]
                self.save_processes()
            except Exception as e:
                print(f"Failed to terminate agent {pid}: {e}")
        else:
            print(f"No agent with PID {pid} found.")
