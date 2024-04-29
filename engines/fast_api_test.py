import os
import uvicorn

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello from Agent"}

if __name__ == "__main__":
    port = int(os.environ.get("AGENT_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
