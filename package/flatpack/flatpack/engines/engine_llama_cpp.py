import os
import shlex
import shutil
import subprocess
import tempfile


class LlamaCPPEngine:
    def __init__(self, model_path, llama_cpp_dir="llama.cpp", n_ctx=4096, n_threads=8, verbose=True):
        self.model_path = model_path
        self.llama_cpp_dir = llama_cpp_dir
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.verbose = verbose
        self.setup_llama_cpp()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def print_debug(self, message):
        if self.verbose:
            print(message)

    def setup_llama_cpp(self):
        self.print_debug(f"Checking if llama.cpp directory '{self.llama_cpp_dir}' exists...")
        if not os.path.exists(self.llama_cpp_dir):
            self.print_debug(f"Directory '{self.llama_cpp_dir}' not found. Cloning repository...")
            self.clone_llama_cpp_repo()
        else:
            self.print_debug(f"Directory '{self.llama_cpp_dir}' already exists. Skipping cloning.")

        main_executable = os.path.join(self.llama_cpp_dir, "llama-cli")
        self.print_debug(f"Checking if main executable '{main_executable}' exists...")
        if not os.path.exists(main_executable):
            self.print_debug(f"Executable '{main_executable}' not found. Running 'make'...")
            self.run_make()
        else:
            self.print_debug(f"Executable '{main_executable}' found. Skipping 'make'.")

    def clone_llama_cpp_repo(self):
        git_executable = shutil.which("git")
        if not git_executable:
            print("[ERROR] The 'git' executable was not found in your PATH.")
            return

        try:
            self.print_debug("Running 'git clone' to clone the llama.cpp repository...")
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
            self.print_debug(f"Successfully cloned llama.cpp repository into '{self.llama_cpp_dir}'")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to clone the llama.cpp repository. Error: {e.stderr.decode()}")

    def run_make(self):
        makefile_path = os.path.join(self.llama_cpp_dir, "Makefile")
        self.print_debug(f"Checking if Makefile exists at '{makefile_path}'...")
        if os.path.exists(makefile_path):
            self.print_debug("Makefile found. Checking if 'make' is available...")
            make_executable = shutil.which("make")
            if not make_executable:
                print("[ERROR] 'make' executable not found in PATH.")
                return

            try:
                self.print_debug("Running 'make' to build the llama.cpp project...")
                subprocess.run(
                    [make_executable],
                    cwd=self.llama_cpp_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                self.print_debug("Successfully ran 'make' in the llama.cpp directory")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to run 'make' in the llama.cpp directory. Error: {e.stderr.decode()}")
        else:
            print("[ERROR] Makefile not found in the llama.cpp directory.")

    def generate_response(self, prompt, max_tokens):
        self.print_debug(f"Generating response for the prompt: {prompt}")
        self.print_debug(f"Maximum tokens set to: {max_tokens}")

        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as prompt_file:
            self.print_debug(f"Writing prompt to temporary file '{prompt_file.name}'...")
            prompt_file.write(prompt)
            prompt_file_path = prompt_file.name

        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as output_file:
            output_file_path = output_file.name
            self.print_debug(f"Output will be written to temporary file '{output_file_path}'")

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

        self.print_debug(f"Executing command: {' '.join(shlex.quote(arg) for arg in command)}")

        try:
            with open(output_file_path, 'w', encoding='utf-8') as out_file:
                subprocess.run(command, stdout=out_file, stderr=subprocess.PIPE, check=True)
                self.print_debug("Command executed successfully.")

            with open(output_file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                self.print_debug(f"Read content from output file: {content}")

            response = content.replace(prompt.strip(), '', 1).strip()
            self.print_debug(f"Generated response: {response}")
            return response

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error running model: {e.stderr.decode()}")
            return "I'm sorry, I don't have a response for that."
        finally:
            self.print_debug(f"Cleaning up temporary files: {prompt_file_path}, {output_file_path}")
            os.remove(prompt_file_path)
            os.remove(output_file_path)
