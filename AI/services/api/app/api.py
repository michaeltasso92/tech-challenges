from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from .recommend import Recommender

class Neighbor(BaseModel):
    item: str
    confidence: float

class RecOut(BaseModel):
    item_id: str
    left: List[Neighbor]
    right: List[Neighbor]

app = FastAPI(title="IWD Recommender API")
rec = None

@app.on_event("startup")
def _startup():
    global rec
    rec = Recommender()

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/recommend/{item_id}", response_model=RecOut)
def recommend(item_id: str):
    return {"item_id": item_id, **rec.get(item_id)}
