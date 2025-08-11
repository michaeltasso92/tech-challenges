import os
from typing import Dict, List
from .util import load_artifacts

TOP_K=int(os.getenv("TOP_K","10"))
MODEL_DIR=os.getenv("MODEL_DIR","/app/models/artifacts")

class Recommender:
    def __init__(self):
        self.left,self.right,self.fb=load_artifacts(MODEL_DIR)

    def get(self, item_uuid: str, k: int = TOP_K) -> Dict[str,List[Dict[str,float]]]:
        l=self.left.get(item_uuid,[{"item":x,"confidence":0.01} for x in self.fb["left"][:k]])
        r=self.right.get(item_uuid,[{"item":x,"confidence":0.01} for x in self.fb["right"][:k]])
        return {"left":l[:k],"right":r[:k]}
