import logging
import os
import shlex
import shutil
import subprocess
import tempfile


class LlamaCPPEngine:
    def __init__(self, model_path, llama_cpp_dir="llama.cpp", n_ctx=4096, n_threads=8, verbose=False):
        self.model_path = model_path
        self.llama_cpp_dir = llama_cpp_dir
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.verbose = verbose
        self.setup_llama_cpp()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def log(self, message, level=logging.INFO):
        if self.verbose:
            print(message)
        else:
            logging.log(level, message)

    def setup_llama_cpp(self):
        if not os.path.exists(self.llama_cpp_dir):
            self.clone_llama_cpp_repo()

        main_executable = os.path.join(self.llama_cpp_dir, "main")
        if not os.path.exists(main_executable):
            self.run_make()

    def clone_llama_cpp_repo(self):
        git_executable = shutil.which("git")
        if not git_executable:
            self.log("[ERROR] The 'git' executable was not found in your PATH.", logging.ERROR)
            return

        try:
            self.log("Cloning llama.cpp repository...")
            subprocess.run(
                [
                    git_executable,
                    "clone",
                    "--depth",
                    "1",
                    "https://github.com/ggerganov/llama.cpp",
                    self.llama_cpp_dir
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.log(f"Finished cloning llama.cpp repository into '{self.llama_cpp_dir}'")
        except subprocess.CalledProcessError as e:
            self.log(f"Failed to clone the llama.cpp repository. Error: {e.stderr.decode()}", logging.ERROR)

    def run_make(self):
        makefile_path = os.path.join(self.llama_cpp_dir, "Makefile")
        if os.path.exists(makefile_path):
            make_executable = shutil.which("make")
            if not make_executable:
                self.log("[ERROR] 'make' executable not found in PATH.", logging.ERROR)
                return

            try:
                self.log("Running 'make' in the llama.cpp directory...")
                subprocess.run(
                    [make_executable],
                    cwd=self.llama_cpp_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.log("Finished running 'make' in the llama.cpp directory")
            except subprocess.CalledProcessError as e:
                self.log(f"Failed to run 'make' in the llama.cpp directory. Error: {e.stderr.decode()}", logging.ERROR)
        else:
            self.log("Makefile not found in the llama.cpp directory.", logging.ERROR)

    def generate_response(self, prompt, max_tokens):
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as prompt_file:
            prompt_file.write(prompt)
            prompt_file_path = prompt_file.name

        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as output_file:
            output_file_path = output_file.name

        command = [
            os.path.join(self.llama_cpp_dir, 'llama-cli'),
            '-m', self.model_path,
            '-n', str(max_tokens),
            '--ctx-size', str(self.n_ctx),
            '--temp', '1.0',
            '--repeat-penalty', '1.0',
            '--log-disable',
            '--file', prompt_file_path
        ]

        self.log(f"Executing command: {' '.join(shlex.quote(arg) for arg in command)}")

        try:
            with open(output_file_path, 'w', encoding='utf-8') as out_file:
                subprocess.run(command, stdout=out_file, stderr=subprocess.PIPE, check=True)

            with open(output_file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()

            response = content.replace(prompt.strip(), '', 1).strip()
            return response

        except subprocess.CalledProcessError as e:
            self.log(f"Error running model: {e.stderr.decode()}", logging.ERROR)
            return "I'm sorry, I don't have a response for that."
        finally:
            os.remove(prompt_file_path)
            os.remove(output_file_path)
