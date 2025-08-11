import os, json, logging
from typing import Dict, List, Optional
import numpy as np

TOP_K = int(os.getenv("TOP_K", "10"))
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models/artifacts")

class Recommender:
    def __init__(self):
        def _load_json(path, default):
            try:
                with open(path, "r") as f: return json.load(f)
            except Exception: return default

        # Fallback (rule-based)
        self.MODEL_DIR = MODEL_DIR
        self.left  = _load_json(os.path.join(MODEL_DIR, "left.json"), {})
        self.right = _load_json(os.path.join(MODEL_DIR, "right.json"), {})
        self.fb    = _load_json(os.path.join(MODEL_DIR, "fallback.json"), {"left": [], "right": []})
        self.names = _load_json(os.path.join(MODEL_DIR, "item_names.json"), {})

        # FAISS + embeddings
        self.faiss_ok = False
        self.vocab = {}; self.rev_vocab = {}
        self.idx_l = None; self.idx_r = None
        self.E_left = None; self.E_right = None

        try:
            import faiss
            vocab_path = os.path.join(MODEL_DIR, "item_vocab.json")
            left_idx_p  = os.path.join(MODEL_DIR, "left.index")
            right_idx_p = os.path.join(MODEL_DIR, "right.index")
            left_emb_p  = os.path.join(MODEL_DIR, "embed_left.npy")
            right_emb_p = os.path.join(MODEL_DIR, "embed_right.npy")

            if all(os.path.exists(p) for p in [vocab_path, left_idx_p, right_idx_p, left_emb_p, right_emb_p]):
                with open(vocab_path) as f:
                    self.vocab = json.load(f)                # item_id -> row
                self.rev_vocab = {v: k for k, v in self.vocab.items()}

                # load indexes
                self.idx_l = faiss.read_index(left_idx_p)
                self.idx_r = faiss.read_index(right_idx_p)

                # load embeddings (already L2-normalized from trainer; re-normalize just in case)
                self.E_left  = np.load(left_emb_p).astype("float32")
                self.E_right = np.load(right_emb_p).astype("float32")
                for E in (self.E_left, self.E_right):
                    n = np.linalg.norm(E, axis=1, keepdims=True); n[n==0]=1; E /= n

                # sanity: sizes must match
                ok = (self.idx_l.ntotal == self.E_left.shape[0] ==
                      self.idx_r.ntotal == self.E_right.shape[0] ==
                      len(self.vocab))
                self.faiss_ok = bool(ok)
                if not ok:
                    logging.warning(f"FAISS size mismatch: "
                                    f"vocab={len(self.vocab)}, "
                                    f"Lidx={self.idx_l.ntotal}, Ridx={self.idx_r.ntotal}, "
                                    f"LE={self.E_left.shape}, RE={self.E_right.shape}")
        except Exception as e:
            logging.warning(f"FAISS not loaded, falling back to rule-based. Reason: {e}")

        logging.info(f"MODEL_DIR: {MODEL_DIR}")
        logging.info(f"FAISS loaded: {self.faiss_ok} | items: {len(self.vocab) if self.faiss_ok else 0}")
        logging.info(f"Rule-based fallback loaded: left_json={len(self.left)} right_json={len(self.right)}")

    def name_of(self, item_id: str) -> Optional[str]:
        return self.names.get(item_id)

    def _faiss_neighbors(self, item_id: str, side: str, k: int):
        if not self.faiss_ok: return None
        i = self.vocab.get(item_id)
        if i is None: return None
        q = (self.E_left[i:i+1] if side == "left" else self.E_right[i:i+1])  # (1, dim)
        index = self.idx_l if side == "left" else self.idx_r
        D, I = index.search(q, k + 5)
        out, seen = [], {item_id}
        for idx, score in zip(I[0], D[0]):
            if idx < 0: continue
            nid = self.rev_vocab.get(int(idx))
            if not nid or nid in seen: continue
            seen.add(nid)
            out.append({"item": nid, "confidence": float(max(score, 0.0))})
            if len(out) >= k: break
        return out

    def _pad_with_fallback(self, lst, side: str, item_id: str, k: int):
        have = {d["item"] for d in lst}
        pool = [x for x in self.fb.get(side, []) if x not in have and x != item_id]
        need = max(0, k - len(lst))
        return lst + [{"item": x, "confidence": 0.01} for x in pool[:need]]

    def _enrich(self, lst):
        return [{"item": d["item"], "name": self.name_of(d["item"]), "confidence": float(d["confidence"])} for d in lst]

    def get(self, item_id: str, k: int = TOP_K):
        l = self._faiss_neighbors(item_id, "left",  k) or self.left.get(item_id, [])
        r = self._faiss_neighbors(item_id, "right", k) or self.right.get(item_id, [])
        l = [d for d in l if d["item"] != item_id]
        r = [d for d in r if d["item"] != item_id]
        l = self._pad_with_fallback(l, "left",  item_id, k)[:k]
        r = self._pad_with_fallback(r, "right", item_id, k)[:k]
        return {"left": self._enrich(l), "right": self._enrich(r)}
