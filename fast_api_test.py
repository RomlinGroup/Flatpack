import os
import uvicorn

from fastapi import FastAPI, HTTPException
from flatpack import load_engines
from pydantic import BaseModel

app = FastAPI()

engine = load_engines.LlamaCPPEngine(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="*q4.gguf",
    n_ctx=4096,
    n_threads=8,
    verbose=False
)


class Query(BaseModel):
    context: str
    question: str


@app.post("/generate-response/")
async def generate_response(query: Query):
    try:
        response = engine.generate_response(context=query.context, question=query.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def read_root():
    pid = os.getpid()
    return {"message": f"Hello from Agent {pid}"}


if __name__ == "__main__":
    port = int(os.environ.get("AGENT_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
