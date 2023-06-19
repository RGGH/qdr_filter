from fastapi import FastAPI

# The file where NeuralSearcher is stored
from neural_searcher import NeuralSearcher

app = FastAPI()

# Create a neural searcher instance
neural_searcher = NeuralSearcher(collection_name="startupsX")


@app.get("/api/search")
def search_startup(q: str, city:str):
    return {"result": neural_searcher.search(text=q, city="London")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
