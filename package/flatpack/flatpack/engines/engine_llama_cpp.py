import os
import re
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

    def setup_llama_cpp(self):
        if not os.path.exists(self.llama_cpp_dir):
            try:
                print(f"📥 Cloning llama.cpp repository...")
                clone_result = subprocess.run(
                    ["git", "clone", "https://github.com/ggerganov/llama.cpp", self.llama_cpp_dir]
                )
                if clone_result.returncode != 0:
                    print(
                        "❌ Failed to clone the llama.cpp repository. Please check your internet connection and try again.")
                    return
                print(f"📥 Finished cloning llama.cpp repository into '{self.llama_cpp_dir}'")
            except Exception as e:
                print(f"❌ Failed to clone the llama.cpp repository. Error: {e}")
                return

        main_executable = os.path.join(self.llama_cpp_dir, "main")
        if not os.path.exists(main_executable):
            makefile_path = os.path.join(self.llama_cpp_dir, "Makefile")
            if os.path.exists(makefile_path):
                try:
                    print(f"🔨 Running 'make' in the llama.cpp directory...")
                    make_result = subprocess.run(["make"], cwd=self.llama_cpp_dir)
                    if make_result.returncode != 0:
                        print(
                            f"❌ Failed to run 'make' in the llama.cpp directory. Please check the logs for more details.")
                        return
                    print(f"🔨 Finished running 'make' in the llama.cpp directory")
                except Exception as e:
                    print(f"❌ Failed to run 'make' in the llama.cpp directory. Error: {e}")
                    return
            else:
                print(
                    f"❌ Makefile not found in the llama.cpp directory. Please ensure the repository is cloned correctly.")

    def generate_response(self, context, question, max_tokens):
        prompt = f"""
Context: {context}
Question: {question}
        """

        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as prompt_file:
            prompt_file.write(prompt)
            prompt_file_path = prompt_file.name

        with tempfile.NamedTemporaryFile(delete=False, mode='r', encoding='utf-8') as output_file:
            output_file_path = output_file.name

        # Build and run the command
        command = (
            f"{os.path.join(self.llama_cpp_dir, 'main')} -m {self.model_path} "
            f"-n {max_tokens} --ctx-size {self.n_ctx} --temp 1.0 --repeat-penalty 1.0 "
            f"--file {prompt_file_path} > {output_file_path} 2>/dev/null"
        )

        if self.verbose:
            print(f"Executing command: {command}")

        result = subprocess.run(command, shell=True, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("Error running model:", result.stderr.decode())
            return "I'm sorry, I don't have a response for that."

        # Read the output from the temporary file
        with open(output_file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()

        # Clean up temporary files
        os.remove(prompt_file_path)
        os.remove(output_file_path)

        # Remove the prompt from the output
        response = content.replace(prompt.strip(), '', 1).strip()

        # List of tags to remove
        tags_to_remove = ['<|assistant|>', '<|end|>', '<s>']

        # Remove specific tags from the content
        for tag in tags_to_remove:
            response = response.replace(tag, '').strip()

        return response


# Example usage
engine = LlamaCPPEngine(
    model_path="./Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=8,
    verbose=False
)

response = engine.generate_response(
    context="You are a helpful assistant.",
    question="What is the capital of Sweden?",
    max_tokens=256
)

print(response)
