import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
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

@app.get("/name/{item_id}")
def name(item_id: str):
    return {"item_id": item_id, "name": rec.name_of(item_id)}

@app.get("/names")
def names(ids: str = Query(..., description="Comma-separated item ids")):
    id_list = [x.strip() for x in ids.split(",") if x.strip()]
    return {"names": {i: rec.name_of(i) for i in id_list}}

@app.get("/debug/model")
def debug_model():
    return {
        "model_dir": rec.MODEL_DIR,
        "faiss_ok": getattr(rec, "faiss_ok", False),
        "num_vocab_items": len(getattr(rec, "vocab", {})) if getattr(rec, "faiss_ok", False) else 0,
        "has_left_index": os.path.exists(os.path.join(rec.MODEL_DIR, "left.index")),
        "has_right_index": os.path.exists(os.path.join(rec.MODEL_DIR, "right.index")),
        "fallback_counts": {
            "left_json": len(getattr(rec, "left", {})),
            "right_json": len(getattr(rec, "right", {})),
        },
    }
