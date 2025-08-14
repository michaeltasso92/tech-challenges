import os
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from .recommend import Recommender, MODEL_DIR

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

@app.get("/recommend/{item_id}")
def recommend(item_id: str, k: int = Query(10, ge=1, le=100)):
    try:
        out = rec.get(item_id, k=k)
        return {"item_name": rec.name_of(item_id), **out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/name/{item_id}")
def name(item_id: str):
    try:
        return {"item_id": item_id, "name": rec.name_of(item_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/items")
def get_items():
    """Get all available items with their names"""
    items = []
    for item_id, name in rec.names.items():
        items.append({"id": item_id, "name": name})
    return items

@app.get("/names")
def get_names(ids: Optional[str] = Query(None, description="Comma-separated item ids")):
    if ids is not None:
        id_list = [x.strip() for x in ids.split(",") if x.strip()]
        return {"names": {i: rec.name_of(i) for i in id_list}}
    # no ids -> return full mapping {item_id: name}
    return rec.names

@app.get("/images/{item_id}")
def get_images(item_id: str):
    """Get image URLs for a specific item"""
    try:
        print(f"API DEBUG: Getting images for {item_id}")
        image_urls = rec.get_image_urls(item_id)
        print(f"API DEBUG: Got image_urls: {image_urls}")
        print(f"API DEBUG: Type of image_urls: {type(image_urls)}")
        result = {"item_id": item_id, "image_urls": image_urls}
        print(f"API DEBUG: Returning result: {result}")
        return result
    except Exception as e:
        print(f"API DEBUG: Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug/model")
def debug_model():
    return {
        "model_dir": MODEL_DIR,
        "faiss_ok": getattr(rec, "faiss_ok", False),
        "num_vocab_items": len(getattr(rec, "vocab", {})) if getattr(rec, "faiss_ok", False) else 0,
        "has_left_index": os.path.exists(os.path.join(MODEL_DIR, "left.index")),
        "has_right_index": os.path.exists(os.path.join(MODEL_DIR, "right.index")),
        "gnn": {
            "use_gnn": getattr(rec, "use_gnn", False),
            "gnn_ok": getattr(rec, "gnn_ok", False),
            "has_gnn_index": os.path.exists(os.path.join(MODEL_DIR, "gnn.index")),
            "has_gnn_embed": os.path.exists(os.path.join(MODEL_DIR, "gnn_embed.npy")),
        },
        "fallback_counts": {
            "left_json": len(getattr(rec, "left", {})),
            "right_json": len(getattr(rec, "right", {})),
        },
        "seen_counts": {
            "train": len(getattr(rec,"seen_train", set())),
            "val":   len(getattr(rec,"seen_val", set())),
            "test":  len(getattr(rec,"seen_test", set())),
        }
    }
