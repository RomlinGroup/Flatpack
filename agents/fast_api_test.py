import os
import re
import uvicorn

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from flatpack import load_engines
from flatpack.vector_manager import VectorManager
from pydantic import BaseModel

app = FastAPI()
vm = VectorManager(directory="vector")

engine = load_engines.LlamaCPPEngine(
    model_path="./gemma-1.1-2b-it-Q4_K_M.gguf",
    n_ctx=8192,
    n_threads=8,
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
    prompt: str
    max_tokens: int


@app.post("/generate-response/")
async def generate_response(query: Query):
    try:
        context = ""
        if vm.is_index_ready():
            results = vm.search_vectors(query.prompt)
            if results:
                context = "\n".join(result['text'] for result in results[:5])

        response = engine.generate_response(
            prompt=(f"""
            Context: {context}\n
            Question: {query.prompt}\n
            Answer in one short sentence using only the context:
            """),
            max_tokens=query.max_tokens
        )

        cleaned_response = re.sub(r'<[^>]+>', '', response).replace('\n\n', ' ').strip()

        return {"response": cleaned_response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def read_root():
    pid = os.getpid()
    return {"message": f"Hello from Agent {pid}"}


if __name__ == "__main__":
    port = int(os.environ.get("AGENT_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
