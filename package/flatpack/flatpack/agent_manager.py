import asyncio
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
        self.filepath = filepath
        self.load_processes()

    async def save_processes(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.processes, f, default=str)

    async def load_processes(self):
        try:
            with open(self.filepath, 'r') as f:
                self.processes = json.load(f)
                for pid in list(self.processes.keys()):
                    if not psutil.pid_exists(int(pid)):
                        del self.processes[pid]
        except FileNotFoundError:
            self.processes = {}

    async def spawn_agent(self, script_path):
        process = await asyncio.create_subprocess_exec(
            "python", script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        pid = process.pid
        self.processes[str(pid)] = {
            'process': process.pid,
            'start_time': datetime.now(),
            'script_path': script_path
        }
        await self.save_processes()
        print(f"Spawned agent {pid} running '{script_path}'")
        return pid

    def list_agents(self):
        if not self.processes:
            print("No active agents.")
        else:
            print("Active agents:")
            for pid, details in self.processes.items():
                print(f"PID: {pid}, Script: {details['script_path']}, Start Time: {details['start_time']}")

    async def terminate_agent(self, pid):
        pid = str(pid)
        if pid in self.processes:
            try:
                os.kill(int(pid), signal.SIGTERM)  # Send SIGTERM signal to terminate the process
                print(f"Terminated agent {pid}")
                del self.processes[pid]
                await self.save_processes()
            except OSError as e:
                print(f"Failed to terminate agent {pid}: {e}")
        else:
            print(f"No agent with PID {pid} found.")
