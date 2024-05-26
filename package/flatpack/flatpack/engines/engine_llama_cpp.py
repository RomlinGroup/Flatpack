import os
import re
import shlex
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

    def setup_llama_cpp(self):
        if not os.path.exists(self.llama_cpp_dir):
            try:
                print("📥 Cloning llama.cpp repository...")
                clone_result = subprocess.run(
                    ["git", "clone", "https://github.com/ggerganov/llama.cpp", self.llama_cpp_dir],
                    check=True
                )
                print(f"📥 Finished cloning llama.cpp repository into '{self.llama_cpp_dir}'")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to clone the llama.cpp repository. Error: {e}")
                return

        main_executable = os.path.join(self.llama_cpp_dir, "main")
        if not os.path.exists(main_executable):
            makefile_path = os.path.join(self.llama_cpp_dir, "Makefile")
            if os.path.exists(makefile_path):
                try:
                    print("🔨 Running 'make' in the llama.cpp directory...")
                    make_result = subprocess.run(["make"], cwd=self.llama_cpp_dir, check=True)
                    print("🔨 Finished running 'make' in the llama.cpp directory")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to run 'make' in the llama.cpp directory. Error: {e}")
                    return
            else:
                print("❌ Makefile not found in the llama.cpp directory.")

    def generate_response(self, prompt, max_tokens):
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as prompt_file:
            prompt_file.write(prompt)
            prompt_file_path = prompt_file.name

        with tempfile.NamedTemporaryFile(delete=False, mode='r', encoding='utf-8') as output_file:
            output_file_path = output_file.name

        # Build the command as a list
        command = [
            os.path.join(self.llama_cpp_dir, 'main'),
            '-m', self.model_path,
            '-n', str(max_tokens),
            '--ctx-size', str(self.n_ctx),
            '--temp', '1.0',
            '--repeat-penalty', '1.0',
            '--log-disable',
            '--file', prompt_file_path
        ]

        if self.verbose:
            print(f"Executing command: {' '.join(shlex.quote(arg) for arg in command)}")

        try:
            with open(output_file_path, 'w') as out_file:
                result = subprocess.run(command, stdout=out_file, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            print("Error running model:", e.stderr.decode())
            return "I'm sorry, I don't have a response for that."

        # Read the output from the temporary file
        with open(output_file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()

        # Clean up temporary files
        os.remove(prompt_file_path)
        os.remove(output_file_path)

        # Remove the prompt from the output
        response = content.replace(prompt.strip(), '', 1).strip()

        return response
