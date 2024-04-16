import uvicorn
from fastapi import FastAPI
from router import retriever
from extension.bm25_algo import *

app = FastAPI()
app.include_router(retriever.router, prefix="/api/question")

if __name__ == "__main__":
    uvicorn.run("main:app", port=5000, reload=True)
