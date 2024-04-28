from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="*q4.gguf",
    n_ctx=4096,
    n_threads=8,
    verbose=False
)

input = {
    "context": "I live in Uppsala, Sweden.",
    "question": "Where do you live?"
}

prompt = f"Context: {input['context']} \nQuestion: {input['question']}\nPlease provide your response in one complete sentence."

output = llm(
    f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
    max_tokens=256,
    stop=["<|end|>"],
    echo=False
)

response = output['choices'][0]['text']
print(response)
