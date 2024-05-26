import asyncio
import ngrok
import os
import re
import sys
import uvicorn

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from flatpack import load_engines
from flatpack.vector_manager import VectorManager
from pydantic import BaseModel

app = FastAPI()
vm = VectorManager(directory="vector")

ngrok_auth_token = os.environ.get('NGROK_AUTHTOKEN')
if not ngrok_auth_token:
    print("❌ Error: NGROK_AUTHTOKEN is not set. Please set it using:")
    print("export NGROK_AUTHTOKEN='your_ngrok_auth_token'")
    sys.exit(1)
else:
    print("NGROK_AUTHTOKEN is set.")

engine = load_engines.LlamaCPPEngine(
    model_path="./gemma-1.1-2b-it-Q4_K_M.gguf",
    n_ctx=8192,
    n_threads=8,
    verbose=False
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    prompt: str
    max_tokens: int


executor = ThreadPoolExecutor(max_workers=4)


async def generate_response_async(engine, prompt, max_tokens):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, engine.generate_response, prompt, max_tokens)


@app.post("/generate-response/")
async def generate_response(query: Query):
    try:
        context = ""
        if vm.is_index_ready():
            results = vm.search_vectors(query.prompt)
            if results:
                context = "\n".join(result['text'] for result in results[:5])

        prompt = f"""
        You are a question-answering assistant. Keep answers brief and factual. Use only the provided context.\n
        Context: {context}\n
        Question: {query.prompt}\n
        Answer:
        """

        try:
            response = await asyncio.wait_for(
                generate_response_async(engine, prompt, query.max_tokens),
                timeout=30.0
            )
        except FuturesTimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out. Please try again.")

        cleaned_response = re.sub(r'<[^>]+>', '', response).replace('\n\n', ' ').strip()

        return {"response": cleaned_response}

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def read_root():
    pid = os.getpid()
    return {"message": f"Hello from Agent {pid}"}


if __name__ == "__main__":
    try:
        port = int(os.environ.get("AGENT_PORT", 8000))
        listener = ngrok.forward(port, authtoken_from_env=True)
        print(f"Ingress established at {listener.url()}")
        uvicorn.run(app, host="127.0.0.1", port=port)
    except KeyboardInterrupt:
        print("❌ FastAPI server has been stopped.")
    except Exception as e:
        print(f"❌ An unexpected error occurred during server run: {e}")
    finally:
        ngrok.disconnect(listener.url())
