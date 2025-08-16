import argparse
import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

p=argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--out", required=True)
p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
a=p.parse_args()
os.makedirs(a.out, exist_ok=True)

names_path=os.path.join(a.inp,"item_names.parquet")
if not os.path.exists(names_path):
    raise FileNotFoundError("item_names.parquet not found; run parser first")

# Load names + brand/folders if you persisted them; else name-only
names=pd.read_parquet(names_path)        # index=item_id, col: "name"
names=names.fillna("")
try:
    meta=pd.read_parquet(os.path.join(a.inp,"item_meta.parquet"))  # optional: brand, folders
    df=names.join(meta, how="left")
except Exception:
    df=names
    df["brand"]=""; df["folders"]=""

def row_text(r):
    brand = (r.get("brand") or "").strip()
    name  = (r.get("name")  or "").strip()
    flds  = r.get("folders")
    if isinstance(flds, (list, tuple, np.ndarray)):
        seq = flds.tolist() if isinstance(flds, np.ndarray) else list(flds)
        flds = " > ".join([str(x) for x in seq if x is not None and str(x)])
    else:
        if flds is None or (isinstance(flds, float) and np.isnan(flds)):
            flds = ""
        else:
            flds = str(flds)
    # optional: type and codes
    typ = (r.get("type") or "").strip()
    code = (r.get("code") or r.get("code2") or r.get("code3") or "").strip()

    parts = [brand, name]
    if flds: parts.append(flds)
    if typ:  parts.append(typ)
    if code: parts.append(code)
    return " | ".join([p for p in parts if p])


df=df.reset_index().rename(columns={"index":"item_id"})
texts=[row_text(r) for r in df.to_dict(orient="records")]

model=SentenceTransformer(a.model)
emb=model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
emb=np.asarray(emb, dtype=np.float32)

# Save: matrix + index map
np.save(os.path.join(a.out,"item_text_emb.npy"), emb)
with open(os.path.join(a.out,"item_index.json"),"w") as f:
    json.dump({iid:i for i,iid in enumerate(df["item_id"].tolist())}, f)

print({"items": len(df), "dim": int(emb.shape[1])})
