from llama_cpp import Llama


class LlamaCPPEngine:
    def __init__(self, repo_id=None, filename=None, n_ctx=4096, n_threads=8, verbose=False):
        if repo_id:
            print(repo_id)
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
            print(filename)
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

    def generate_response(self, context, question, max_tokens):
        prompt = f"""
        Context: {context}\n
        Question: {question}\n
        """

        messages = [
            {"role": "user", "content": prompt}
        ]

        response_chunks = self.model.create_chat_completion(
            max_tokens=max_tokens,
            messages=messages,
            stop=["<|end|>"],
            stream=True
        )

        for chunk in response_chunks:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                yield delta['content']
