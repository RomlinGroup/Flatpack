import json
import logging
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

    def __init__(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def save_processes(self):
        try:
            with self.filepath.open('w') as f:
                json.dump(self.processes, f, indent=4)
            logging.info("Process data saved successfully.")
        except Exception as e:
            logging.error(f"Failed to save process data: {e}")

    def load_processes(self):
        if self.filepath.exists():
            try:
                with self.filepath.open('r') as f:
                    self.processes = json.load(f)
                logging.info("Process data loaded successfully.")
            except Exception as e:
                logging.error(f"Failed to load process data: {e}")
        self.update_active_processes()

    def update_active_processes(self):
        active_processes = {}
        for pid, info in self.processes.items():
            if psutil.pid_exists(int(pid)):
                active_processes[pid] = info
            else:
                logging.warning(f"Process with PID {pid} no longer exists.")
        self.processes = active_processes
        self.save_processes()

    def spawn_agent(self, script_path):
        try:
            command = ["python", script_path]
            pid = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).pid
            self.processes[str(pid)] = {
                'start_time': datetime.now().isoformat(),
                'script_path': script_path
            }
            self.save_processes()
            logging.info(f"Spawned agent with PID {pid} running '{script_path}'")
            return pid
        except Exception as e:
            logging.error(f"Failed to spawn agent: {e}")

    def list_agents(self):
        self.update_active_processes()
        if not self.processes:
            logging.info("No active agents.")
        else:
            logging.info("Active agents:")
            for pid, details in self.processes.items():
                logging.info(f"PID: {pid}, Script: {details['script_path']}, Start Time: {details['start_time']}")

    def terminate_agent(self, pid):
        try:
            os.kill(int(pid), signal.SIGTERM)
            logging.info(f"Terminated agent with PID {pid}")
            del self.processes[str(pid)]
            self.save_processes()
        except OSError as e:
            logging.error(f"Failed to terminate agent with PID {pid}: {e}")
        except KeyError:
            logging.error(f"No agent with PID {pid} found.")
