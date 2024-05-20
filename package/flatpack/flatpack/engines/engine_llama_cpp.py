import time

from llama_cpp import Llama


class LlamaCPPEngine:
    def __init__(self, repo_id=None, filename=None, n_ctx=4096, n_threads=8, verbose=False):
        start_time = time.time()
        print("Initializing LlamaCPPEngine...")

        if repo_id:
            print(f"Loading model from repo_id: {repo_id}")
            self.model = Llama.from_pretrained(
                echo=False,
                filename=filename,
                n_ctx=n_ctx,
                n_threads=n_threads,
                repeat_penalty=1.0,
                repo_id=repo_id,
                seed=-1,
                streaming=True,
                temp=1.0,
                verbose=verbose
            )
        else:
            print(f"Loading model from file: {filename}")
            self.model = Llama(
                echo=False,
                model_path=filename,
                n_ctx=n_ctx,
                n_threads=n_threads,
                repeat_penalty=1.0,
                seed=-1,
                streaming=True,
                temp=1.0,
                verbose=verbose
            )

        print(f"Model loaded in {time.time() - start_time:.2f} seconds")

    def generate_response(self, context, question, max_tokens):
        prompt = f"""
        Context: {context}\n
        Question: {question}\n
        """

        messages = [
            {"role": "user", "content": prompt}
        ]

        start_time = time.time()
        print("Generating response...")

        response_chunks = self.model.create_chat_completion(
            max_tokens=max_tokens,
            messages=messages,
            stop=[""],
            stream=True
        )

        print(f"Response generation started in {time.time() - start_time:.2f} seconds")

        for chunk in response_chunks:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                print(f"Yielding response chunk: {delta['content']}")
                yield delta['content']

        print(f"Response generation completed in {time.time() - start_time:.2f} seconds")
