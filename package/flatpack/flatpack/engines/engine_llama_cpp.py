from llama_cpp import Llama


class LlamaCPPEngine:
    def __init__(self, repo_id=None, filename=None, n_ctx=4096, n_threads=6, verbose=False):
        if repo_id:
            self.model = Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_ctx=n_ctx,
                n_threads=n_threads,
                temp=1.0,
                repeat_penalty=1.0,
                verbose=verbose
            )
        else:
            self.model = Llama.from_local(
                model_path=filename,
                n_ctx=n_ctx,
                n_threads=n_threads,
                temp=1.0,
                repeat_penalty=1.0,
                verbose=verbose
            )

    def generate_response(self, context, question):
        prompt = f"""
        Context: {context}\n
        Question: {question}\n
        """
        output = self.model(
            f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
            max_tokens=256,
            stop=["<|end|>"],
            echo=False
        )
        return output['choices'][0]['text']
