import os, json
from typing import Dict, List
from .util import load_artifacts

TOP_K=int(os.getenv("TOP_K","10"))
MODEL_DIR=os.getenv("MODEL_DIR","/app/models/artifacts")

class Recommender:
    def __init__(self):
        self.left, self.right, self.fb = load_artifacts(MODEL_DIR)
        names_path = os.path.join(MODEL_DIR, "item_names.json")
        self.names = json.load(open(names_path)) if os.path.exists(names_path) else {}

    def name_of(self, item_id: str) -> str | None:
        return self.names.get(item_id)

    def enrich_neighbors(self, lst, item_id):
        out=[]
        for d in lst:
            if d.get("item") == item_id:  # skip self
                continue
            out.append({
                "item": d["item"],
                "name": d.get("name") or self.name_of(d["item"]),
                "confidence": d.get("confidence", 0.0),
            })
        return out

    def get(self, item_id: str, k: int = TOP_K):
        l = self.left.get(item_id, [{"item": x, "confidence": 0.01} for x in self.fb["left"][:k]])
        r = self.right.get(item_id, [{"item": x, "confidence": 0.01} for x in self.fb["right"][:k]])

        # filter out self-adjacency
        l = [d for d in l if d.get("item") != item_id]
        r = [d for d in r if d.get("item") != item_id]

        # enrich with names
        def enrich(lst):
            out=[]
            for d in lst[:k]:
                nid = d.get("item")
                out.append({
                    "item": nid,
                    "name": d.get("name") or self.name_of(nid),
                    "confidence": float(d.get("confidence", 0.0))
                })
            return out

        return {"left": enrich(l), "right": enrich(r)}
