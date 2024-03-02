from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

chat_handler = Llava15ChatHandler(clip_model_path="mmproj-obsidian-f16.gguf")

llm = Llama(
    model_path="obsidian-q6.gguf",
    chat_handler=chat_handler,
    n_ctx=2048,
    logits_all=True
)

llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are an assistant who perfectly describes images."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://romlin.com/temp/tiger.png"}},
                {"type": "text", "text": "Describe this image in detail please."}
            ]
        }
    ]
)
