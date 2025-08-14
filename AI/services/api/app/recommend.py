import os, json, logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

TOP_K = int(os.getenv("TOP_K", "10"))
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models/artifacts")
USE_GNN = os.getenv("USE_GNN", "0") == "1"
GNN_RERANK_CANDIDATES = int(os.getenv("GNN_RERANK_CANDIDATES", "0"))  # if >0, rerank bi-encoder seeds with GNN

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
        # Images from item_meta.parquet
        self.images: Dict[str, List[str]] = {}
        self._meta_parquet_candidates = [
            os.path.join(MODEL_DIR, "item_meta.parquet"),
            "/app/data/interim/item_meta.parquet",
        ]

        # FAISS + embeddings (bi-encoder)
        self.faiss_ok = False
        self.vocab = {}; self.rev_vocab = {}
        self.idx_l = None; self.idx_r = None
        self.E_left = None; self.E_right = None
        # Optional GNN embeddings/index
        self.use_gnn = USE_GNN
        self.gnn_ok = False
        self.gnn_idx = None
        self.E_gnn = None
        self.gnn_rerank_k = max(0, int(GNN_RERANK_CANDIDATES))

        try:
            import faiss
            vocab_path = os.path.join(MODEL_DIR, "item_vocab.json")
            left_idx_p  = os.path.join(MODEL_DIR, "left.index")
            right_idx_p = os.path.join(MODEL_DIR, "right.index")
            left_emb_p  = os.path.join(MODEL_DIR, "embed_left.npy")
            right_emb_p = os.path.join(MODEL_DIR, "embed_right.npy")
            gnn_idx_p   = os.path.join(MODEL_DIR, "gnn.index")
            gnn_emb_p   = os.path.join(MODEL_DIR, "gnn_embed.npy")
            meta_p      = None  # defer to candidate list below

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
            # Load images from first available meta parquet
            for cand in self._meta_parquet_candidates:
                if not os.path.exists(cand):
                    continue
                try:
                    meta_df = pd.read_parquet(cand)
                    if "item_id" in meta_df.columns:
                        meta_df = meta_df.set_index("item_id")
                    if "image_urls" in meta_df.columns:
                        imgs: Dict[str, List[str]] = {}
                        for iid, row in meta_df["image_urls"].items():
                            try:
                                urls = []
                                if isinstance(row, (list, tuple)):
                                    urls = [str(u) for u in row if u]
                                elif isinstance(row, np.ndarray):
                                    urls = [str(u) for u in row.tolist() if u]
                                elif isinstance(row, str) and row:
                                    urls = [row]
                                if urls:
                                    imgs[str(iid)] = urls
                            except Exception:
                                continue
                        if imgs:
                            self.images = imgs
                            logging.info(f"Loaded images for {len(self.images)} items from {cand}")
                            break
                except Exception as e:
                    logging.warning(f"Failed to load images from {cand}: {e}")
            # Try to load optional GNN artifacts
            if all(os.path.exists(p) for p in [vocab_path, gnn_idx_p, gnn_emb_p]):
                # vocab already loaded if bi-encoder present; if not, load it
                if not self.vocab:
                    with open(vocab_path) as f:
                        self.vocab = json.load(f)
                    self.rev_vocab = {v: k for k, v in self.vocab.items()}

                self.gnn_idx = faiss.read_index(gnn_idx_p)
                self.E_gnn = np.load(gnn_emb_p).astype("float32")
                # normalize
                n = np.linalg.norm(self.E_gnn, axis=1, keepdims=True); n[n==0]=1; self.E_gnn /= n
                # size check
                ok = (self.gnn_idx.ntotal == self.E_gnn.shape[0] == len(self.vocab))
                self.gnn_ok = bool(ok)
                if not ok:
                    logging.warning(f"GNN size mismatch: vocab={len(self.vocab)}, idx={self.gnn_idx.ntotal}, E={self.E_gnn.shape}")
        except Exception as e:
            logging.warning(f"FAISS not loaded, falling back to rule-based. Reason: {e}")

        logging.info(f"MODEL_DIR: {MODEL_DIR}")
        logging.info(f"FAISS loaded: {self.faiss_ok} | items: {len(self.vocab) if self.vocab else 0}")
        logging.info(f"Rule-based fallback loaded: left_json={len(self.left)} right_json={len(self.right)}")
        logging.info(f"GNN available: {self.gnn_ok} | use_gnn={self.use_gnn} | gnn_rerank_k={self.gnn_rerank_k}")

        # Seen/cold-start info
        self.seen = {"train_items": [], "val_items": [], "test_items": []}
        seen_p = os.path.join(MODEL_DIR, "seen_items.json")
        try:
            if os.path.exists(seen_p):
                with open(seen_p) as f:
                    self.seen = json.load(f)
        except Exception as e:
            logging.warning(f"Could not load seen_items.json: {e}")
        self.seen_train = set(self.seen.get("train_items", []))
        self.seen_val   = set(self.seen.get("val_items", []))
        self.seen_test  = set(self.seen.get("test_items", []))

    def name_of(self, item_id: str) -> Optional[str]:
        return self.names.get(item_id)

    def get_image_urls(self, item_id: str) -> List[str]:
        imgs = self.images.get(item_id)
        if imgs:
            return imgs
        # Lazy-load single item from parquet if available
        try:
            for cand in self._meta_parquet_candidates:
                if not os.path.exists(cand):
                    continue
                meta_df = pd.read_parquet(cand)
                if "item_id" in meta_df.columns:
                    meta_df = meta_df.set_index("item_id")
                if item_id not in meta_df.index:
                    continue
                cell = meta_df.loc[item_id]["image_urls"] if "image_urls" in meta_df.columns else None
                if isinstance(cell, (list, tuple)):
                    urls = [str(u) for u in cell if u]
                elif isinstance(cell, np.ndarray):
                    urls = [str(u) for u in cell.tolist() if u]
                elif isinstance(cell, str) and cell:
                    urls = [cell]
                else:
                    urls = []
                if urls:
                    self.images[item_id] = urls
                    return urls
        except Exception as e:
            logging.warning(f"lazy image load failed for {item_id}: {e}")
        return []

    def seen_status(self, item_id: str):
        if item_id in self.seen_train: return "train"
        if item_id in self.seen_val:   return "val"
        if item_id in self.seen_test:  return "test"
        return "unseen"

    def _faiss_neighbors(self, item_id: str, side: str, k: int):
        if not self.faiss_ok: return None
        i = self.vocab.get(item_id)
        if i is None: return None
        q = (self.E_left[i:i+1] if side == "left" else self.E_right[i:i+1])  # (1, dim)
        index = self.idx_l if side == "left" else self.idx_r
        D, I = index.search(q, k + 10)  # Get more candidates to filter duplicates
        out, seen = [], {item_id}
        seen_names = {self.name_of(item_id)}  # Track product names to avoid duplicates
        
        for idx, score in zip(I[0], D[0]):
            if idx < 0: continue
            nid = self.rev_vocab.get(int(idx))
            if not nid or nid in seen: continue
            
            # Check if this is a duplicate product name
            neighbor_name = self.name_of(nid)
            if neighbor_name in seen_names:
                continue  # Skip if we already have this product name
                
            seen.add(nid)
            seen_names.add(neighbor_name)
            out.append({"item": nid, "confidence": float(max(score, 0.0))})
            if len(out) >= k: break
        return out

    def _gnn_neighbors(self, item_id: str, k: int):
        if not (self.gnn_ok and self.use_gnn):
            return None
        i = self.vocab.get(item_id)
        if i is None:
            return None
        q = self.E_gnn[i:i+1]
        D, I = self.gnn_idx.search(q, k + 10)
        out, seen = [], {item_id}
        seen_names = {self.name_of(item_id)}
        for idx, score in zip(I[0], D[0]):
            if idx < 0: continue
            nid = self.rev_vocab.get(int(idx))
            if not nid or nid in seen: continue
            neighbor_name = self.name_of(nid)
            if neighbor_name in seen_names:
                continue
            seen.add(nid)
            seen_names.add(neighbor_name)
            out.append({"item": nid, "confidence": float(max(score, 0.0))})
            if len(out) >= k: break
        return out

    def _gnn_rerank(self, item_id: str, seeds: list, k: int):
        """Rerank candidate item_ids using cosine with GNN embeddings."""
        if not (self.gnn_ok and self.gnn_rerank_k > 0):
            return None
        i = self.vocab.get(item_id)
        if i is None:
            return None
        q = self.E_gnn[i]
        # Collect candidate indices and keep a mapping
        cand_ids = []
        seen = set([item_id])
        for d in seeds:
            cid = d.get("item") if isinstance(d, dict) else d
            if not cid or cid in seen: continue
            seen.add(cid)
            cand_ids.append(cid)
        if not cand_ids:
            return None
        cidx = [self.vocab.get(cid) for cid in cand_ids]
        cidx = [x for x in cidx if x is not None]
        if not cidx:
            return None
        C = self.E_gnn[cidx]
        # cosine since E_gnn is L2-normalized
        scores = (C * q).sum(axis=1)
        order = np.argsort(-scores)[:k]
        out = []
        for oi in order:
            nid = self.rev_vocab.get(int(cidx[oi]))
            if not nid: continue
            out.append({"item": nid, "confidence": float(max(float(scores[oi]), 0.0))})
        return out

    def _pad_with_fallback(self, lst, side: str, item_id: str, k: int):
        have = {d["item"] for d in lst}
        pool = [x for x in self.fb.get(side, []) if x not in have and x != item_id]
        need = max(0, k - len(lst))
        return lst + [{"item": x, "confidence": 0.01} for x in pool[:need]]

    def _enrich(self, lst):
        return [{"item": d["item"], "name": d.get("name") or self.name_of(d["item"]), "confidence": float(d["confidence"])} for d in lst]

    def _deduplicate_by_name(self, recommendations, item_id: str):
        """Remove duplicate product names from recommendations"""
        seen_names = {self.name_of(item_id)}
        deduplicated = []
        
        for rec in recommendations:
            if rec["item"] == item_id:
                continue  # Skip self
                
            neighbor_name = self.name_of(rec["item"])
            if neighbor_name in seen_names:
                continue  # Skip duplicate product names
                
            seen_names.add(neighbor_name)
            deduplicated.append(rec)
            
        return deduplicated

    def get(self, item_id: str, k: int = TOP_K):
        # If rerank is enabled, pull larger candidate pools from bi-encoder then rerank with GNN
        if self.gnn_rerank_k > 0 and self.gnn_ok:
            seeds_l = self._faiss_neighbors(item_id, "left",  max(k, self.gnn_rerank_k)) or []
            seeds_r = self._faiss_neighbors(item_id, "right", max(k, self.gnn_rerank_k)) or []
            rerank_l = self._gnn_rerank(item_id, seeds_l, k) or seeds_l[:k]
            rerank_r = self._gnn_rerank(item_id, seeds_r, k) or seeds_r[:k]
            l, r = rerank_l, rerank_r
        elif self.use_gnn and self.gnn_ok:
            g = self._gnn_neighbors(item_id, k) or []
            g = self._deduplicate_by_name(g, item_id)[:k]
            l = g[:]
            r = g[:]
        else:
            l = self._faiss_neighbors(item_id, "left",  k) or self.left.get(item_id, [])
            r = self._faiss_neighbors(item_id, "right", k) or self.right.get(item_id, [])
        
        # Deduplicate by product name
        l = self._deduplicate_by_name(l, item_id)[:k]
        r = self._deduplicate_by_name(r, item_id)[:k]
        
        l = self._pad_with_fallback(l, "left",  item_id, k)[:k]
        r = self._pad_with_fallback(r, "right", item_id, k)[:k]
        result = {"left": self._enrich(l), "right": self._enrich(r)}
        # annotate query status
        result["query_seen"] = self.seen_status(item_id)
        # annotate each neighbor (helpful in debug)
        for side in ("left","right"):
            for d in result[side]:
                d["seen"] = self.seen_status(d["item"])
        return result
