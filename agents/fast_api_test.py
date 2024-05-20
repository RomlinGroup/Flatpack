import os
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from flatpack.vector_manager import VectorManager
from flatpack import load_engines
from pydantic import BaseModel

app = FastAPI()
vm = VectorManager(directory="vector")

# Load the LlamaCPPEngine
engine = load_engines.LlamaCPPEngine(
    filename="./Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    n_ctx=4096,
    n_threads=6,
    verbose=False
)

origins = [
    "http://127.0.0.1:8080",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    context: str
    question: str
    max_tokens: int


@app.post("/generate-response/")
async def generate_response(query: Query):
    try:
        async def response_generator():
            context = query.context

            if vm.is_index_ready():
                results = vm.search_vectors(query.question)
                if results:
                    context = "\n".join(result['text'] for result in results[:10])

            for chunk in engine.generate_response(context=context, question=query.question,
                                                  max_tokens=query.max_tokens):
                yield chunk

        return StreamingResponse(response_generator(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def read_root():
    pid = os.getpid()
    return {"message": f"Hello from Agent {pid}"}


if __name__ == "__main__":
    port = int(os.environ.get("AGENT_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
