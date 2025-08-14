import os, json, argparse, logging, random, hashlib
import numpy as np, pandas as pd
from tqdm import tqdm
import dgl
import torch
import torch.nn as nn
from typing import Dict, Tuple
import mlflow
from dgl.nn import SAGEConv, HeteroGraphConv
import faiss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class GNNGraphBuilder:
    def __init__(self, interim_dir: str, artifacts_dir: str, seed: int = 42, experiment_name: str = "gnn-graph-building",
                 hidden_dim: int = 512, out_dim: int = 256, epochs: int = 15, lr: float = 1e-3, neg_ratio: int = 5,
                 use_edge_weights: bool = True):
        self.interim = interim_dir
        self.artifacts = artifacts_dir
        self.seed = seed
        self.experiment_name = experiment_name
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.epochs = epochs
        self.lr = lr
        self.neg_ratio = max(1, int(neg_ratio))
        self.use_edge_weights = use_edge_weights
        os.makedirs(self.artifacts, exist_ok=True)
        self.item2idx: Dict[str,int] = {}
        self.idx2item: Dict[int,str] = {}
        self.num_nodes = 0
        
        # Setup MLflow
        mlflow.set_experiment(self.experiment_name)
        # Reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def load_vocab(self):
        vocab_p = os.path.join(self.artifacts, "item_vocab.json")
        if not os.path.exists(vocab_p):
            raise FileNotFoundError(f"item_vocab.json not found in {self.artifacts}")
        with open(vocab_p) as f:
            self.item2idx = {k:int(v) for k,v in json.load(f).items()}
        # idx2item of length = max(idx)+1 (aligns with embed rows)
        n = max(self.item2idx.values()) + 1 if self.item2idx else 0
        self.idx2item = {v:k for k,v in self.item2idx.items()}
        self.num_nodes = n
        logging.info(f"Vocab loaded: {len(self.item2idx):,} items, nodes={self.num_nodes}")

    def _hash_index(self, text: str, dim: int) -> int:
        h = hashlib.md5(text.encode("utf-8")).hexdigest()
        return int(h, 16) % dim

    def _build_meta_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (misc_features, brand_ids) aligned to vocab rows.
        misc_features includes hashed folders/markets, type flags, and numeric stats.
        """
        meta_p = os.path.join(self.interim, "item_meta.parquet")
        parsed_p = os.path.join(self.interim, "parsed.parquet")
        folders_dim = 128
        markets_dim = 64

        # Initialize
        misc = np.zeros((self.num_nodes, folders_dim + markets_dim + 2 + 6), dtype=np.float32)
        brand_ids = np.zeros((self.num_nodes,), dtype=np.int64)

        # Build brand id mapping
        brand_to_id = {"<UNK>": 0}

        # Load meta if exists
        if os.path.exists(meta_p):
            meta = pd.read_parquet(meta_p)
            if "item_id" in meta.columns:
                meta = meta.set_index("item_id")
        else:
            meta = pd.DataFrame()

        # Vectors per item
        for item, idx in self.item2idx.items():
            if item not in meta.index:
                brand_ids[idx] = 0
                continue
            row = meta.loc[item]
            # brand id
            brand = str(row.get("brand", "")).strip() if isinstance(row.get("brand"), str) else ""
            if brand not in brand_to_id:
                brand_to_id[brand] = len(brand_to_id)
            brand_ids[idx] = brand_to_id.get(brand, 0)

            # type flags
            typ = str(row.get("type", "")).lower()
            is_product = 1.0 if typ == "product" else 0.0
            is_tester = 1.0 if typ == "tester" else 0.0
            misc[idx, 0] = is_product
            misc[idx, 1] = is_tester

            off = 2
            # folders hashed bag
            fl = row.get("folders")
            if isinstance(fl, (list, tuple)):
                for f in fl:
                    if not f: continue
                    j = self._hash_index(str(f), folders_dim)
                    misc[idx, off + j] += 1.0
            off += folders_dim
            # markets hashed bag
            mk = row.get("markets")
            if isinstance(mk, (list, tuple)):
                for m in mk:
                    if not m: continue
                    j = self._hash_index(str(m), markets_dim)
                    misc[idx, off + j] += 1.0

        # Add planogram stats from parsed.parquet
        if os.path.exists(parsed_p):
            df = pd.read_parquet(parsed_p)
            # basic counts
            counts = df.groupby("item_id").size()
            left_deg = df[~df["left_neighbor"].isna()].groupby("item_id").size()
            right_deg = df[~df["right_neighbor"].isna()].groupby("item_id").size()
            # seq lengths for normalized pos
            seq_len = df.groupby(["guideline_id","group_seq"]).agg(seq_len=("pos", lambda s: int(s.max())+1)).reset_index()
            df2 = df.merge(seq_len, on=["guideline_id","group_seq"], how="left")
            df2["pos_norm"] = df2["pos"] / df2["seq_len"].clip(lower=1)
            pos_stats = df2.groupby("item_id")["pos_norm"].agg(["mean","std"]).fillna(0.0)
            num_guides = df.groupby("item_id")["guideline_id"].nunique()
            num_seqs = df.groupby("item_id")["group_seq"].nunique()

            # Fill misc columns
            # layout: [is_product,is_tester] [folders 128] [markets 64] [pop,deg_L,deg_R,avg_pos,std_pos,diversity]
            for item, idx in self.item2idx.items():
                off = 2 + folders_dim + markets_dim
                misc[idx, off + 0] = np.log1p(float(counts.get(item, 0)))
                misc[idx, off + 1] = np.log1p(float(left_deg.get(item, 0)))
                misc[idx, off + 2] = np.log1p(float(right_deg.get(item, 0)))
                misc[idx, off + 3] = float(pos_stats.loc[item, "mean"]) if item in pos_stats.index else 0.0
                misc[idx, off + 4] = float(pos_stats.loc[item, "std"]) if item in pos_stats.index else 0.0
                misc[idx, off + 5] = float(num_guides.get(item, 0) + num_seqs.get(item, 0))

        # Standardize non-binary columns (excluding first two type flags)
        start_std = 2
        mu = misc[:, start_std:].mean(axis=0)
        sigma = misc[:, start_std:].std(axis=0)
        sigma[sigma == 0] = 1.0
        misc[:, start_std:] = (misc[:, start_std:] - mu) / sigma

        logging.info(f"Meta features: misc_dim={misc.shape[1]} brands={len(brand_to_id)}")
        return misc.astype(np.float32), brand_ids.astype(np.int64)

    def build_node_features(self) -> np.ndarray:
        # Base text features
        el_p = os.path.join(self.artifacts, "embed_left.npy")
        er_p = os.path.join(self.artifacts, "embed_right.npy")
        if not (os.path.exists(el_p) and os.path.exists(er_p)):
            raise FileNotFoundError("embed_left.npy or embed_right.npy not found; run bi-encoder trainer first")
        E_left  = np.load(el_p).astype("float32")
        E_right = np.load(er_p).astype("float32")
        if E_left.shape[0] != self.num_nodes or E_right.shape[0] != self.num_nodes:
            raise ValueError(f"Embedding rows mismatch vocab: {E_left.shape[0]} vs {self.num_nodes}")
        X_text = np.concatenate([E_left, E_right], axis=1)
        # Additional meta features
        misc, brand_ids = self._build_meta_features()
        X = np.concatenate([X_text, misc], axis=1)
        # L2 normalize full feature vector
        n = np.linalg.norm(X, axis=1, keepdims=True); n[n==0]=1.0; X = X / n
        # Persist brand ids for potential future trainable embeddings
        np.save(os.path.join(self.artifacts, "gnn_brand_ids.npy"), brand_ids)
        logging.info(f"Node features: X shape={X.shape} (text + meta); saved gnn_brand_ids.npy")
        return X

    def load_edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return directed edges and weights for left and right as (src,dst,weight) per relation."""
        parsed_p = os.path.join(self.interim, "parsed.parquet")
        if not os.path.exists(parsed_p):
            raise FileNotFoundError("parsed.parquet not found; run parser first")
        df = pd.read_parquet(parsed_p)
        # Drop self and nulls
        df = df[(df["item_id"].notna())]
        df = df[(df["item_id"] != df["left_neighbor"]) & (df["item_id"] != df["right_neighbor"])]
        # Edge frequencies
        left_pairs = df.dropna(subset=["left_neighbor"]).groupby(["item_id","left_neighbor"]).size().rename("w").reset_index()
        right_pairs = df.dropna(subset=["right_neighbor"]).groupby(["item_id","right_neighbor"]).size().rename("w").reset_index()

        def to_idx_pairs(sub: pd.DataFrame, col: str):
            rows = []
            ws = []
            for r in sub.itertuples(index=False):
                u = self.item2idx.get(r.item_id, None)
                v = self.item2idx.get(getattr(r, col), None)
                if u is None or v is None: continue
                rows.append((u, v))
                w = getattr(r, "w", 1)
                ws.append(float(np.log1p(w)))
            if not rows:
                return (np.empty((0,),dtype=np.int64), np.empty((0,),dtype=np.int64), np.empty((0,),dtype=np.float32))
            a = np.asarray(rows, dtype=np.int64)
            w = np.asarray(ws, dtype=np.float32)
            return a[:,0], a[:,1], w

        l_src, l_dst, l_w = to_idx_pairs(left_pairs,  "left_neighbor")
        r_src, r_dst, r_w = to_idx_pairs(right_pairs, "right_neighbor")
        logging.info(f"Edges: LEFT {len(l_src):,} | RIGHT {len(r_src):,}")
        return l_src, l_dst, r_src, r_dst, l_w, r_w

    def build_heterograph(self, l_src, l_dst, r_src, r_dst, l_w, r_w) -> dgl.DGLHeteroGraph:
        data_dict = {
            ('item','left','item'):  (torch.as_tensor(l_src), torch.as_tensor(l_dst)),
            ('item','right','item'): (torch.as_tensor(r_src), torch.as_tensor(r_dst)),
        }
        g = dgl.heterograph(data_dict, num_nodes_dict={'item': self.num_nodes})
        # attach weights for potential future use
        if len(l_w): g.edges['left'].data['w'] = torch.as_tensor(l_w, dtype=torch.float32)
        if len(r_w): g.edges['right'].data['w'] = torch.as_tensor(r_w, dtype=torch.float32)
        logging.info(g)
        return g

    # ------------------------------ GNN model ---------------------------------
    class _HeteroSAGE(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
            super().__init__()
            self.layers = nn.ModuleList([
                HeteroGraphConv(
                    {
                        'left':  SAGEConv(in_dim, hidden_dim, aggregator_type='mean'),
                        'right': SAGEConv(in_dim, hidden_dim, aggregator_type='mean'),
                    },
                    aggregate='mean'
                ),
                HeteroGraphConv(
                    {
                        'left':  SAGEConv(hidden_dim, out_dim, aggregator_type='mean'),
                        'right': SAGEConv(hidden_dim, out_dim, aggregator_type='mean'),
                    },
                    aggregate='mean'
                )
            ])
            self.act = nn.ReLU()

        def forward(self, g: dgl.DGLHeteroGraph, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
            h = x_dict
            for i, layer in enumerate(self.layers):
                h = layer(g, h)
                # Only one node type: 'item'
                h['item'] = self.act(h['item']) if i == 0 else h['item']
            return h['item']  # (num_nodes, out_dim)

    @staticmethod
    def _dot_score(h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return (h[src] * h[dst]).sum(dim=1)

    @staticmethod
    def _neg_sample(num_nodes: int, num_pos: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, num_nodes, (num_pos,), device=device)

    def save_pretrain_dump(self, X: np.ndarray, l_src, l_dst, r_src, r_dst):
        # Save as numpy for quick reload in the trainer step
        np.save(os.path.join(self.artifacts, "gnn_X.npy"), X)
        np.save(os.path.join(self.artifacts, "gnn_left_src.npy"),  l_src)
        np.save(os.path.join(self.artifacts, "gnn_left_dst.npy"),  l_dst)
        np.save(os.path.join(self.artifacts, "gnn_right_src.npy"), r_src)
        np.save(os.path.join(self.artifacts, "gnn_right_dst.npy"), r_dst)
        logging.info("Saved: gnn_X.npy, gnn_left/right_{src,dst}.npy")

    def train_gnn(self, g: dgl.DGLHeteroGraph, X: np.ndarray, l_src, l_dst, r_src, r_dst,
                  l_w=None, r_w=None) -> np.ndarray:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        g = g.to(device)
        feat = torch.from_numpy(X).to(device)
        # Assign node feature
        g.nodes['item'].data['feat'] = feat

        model = self._HeteroSAGE(in_dim=feat.shape[1], hidden_dim=self.hidden_dim, out_dim=self.out_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss()

        # Pre-build positive edges tensors
        l_src_t = torch.as_tensor(l_src, device=device)
        l_dst_t = torch.as_tensor(l_dst, device=device)
        r_src_t = torch.as_tensor(r_src, device=device)
        r_dst_t = torch.as_tensor(r_dst, device=device)
        l_w_t = torch.as_tensor(l_w, device=device, dtype=torch.float32) if l_w is not None else None
        r_w_t = torch.as_tensor(r_w, device=device, dtype=torch.float32) if r_w is not None else None

        for epoch in range(1, self.epochs+1):
            model.train()
            opt.zero_grad()
            h = model(g, {'item': g.nodes['item'].data['feat']})
            # LEFT relation loss
            l_pos_score = self._dot_score(h, l_src_t, l_dst_t)
            l_neg_dst = self._neg_sample(self.num_nodes, len(l_src_t) * self.neg_ratio, device)
            l_neg_src = l_src_t.repeat_interleave(self.neg_ratio)
            l_neg_score = self._dot_score(h, l_neg_src, l_neg_dst)
            l_logits = torch.cat([l_pos_score, l_neg_score], dim=0)
            l_labels = torch.cat([torch.ones_like(l_pos_score), torch.zeros_like(l_neg_score)], dim=0)
            if self.use_edge_weights and l_w_t is not None and len(l_w_t) == len(l_pos_score):
                # normalize weights to mean ~1
                lw = l_w_t / (l_w_t.mean() + 1e-6)
                l_weight = torch.cat([lw, torch.ones_like(l_neg_score)], dim=0)
                left_loss = nn.functional.binary_cross_entropy_with_logits(l_logits, l_labels, weight=l_weight)
            else:
                left_loss = bce(l_logits, l_labels)
            # RIGHT relation loss
            r_pos_score = self._dot_score(h, r_src_t, r_dst_t)
            r_neg_dst = self._neg_sample(self.num_nodes, len(r_src_t) * self.neg_ratio, device)
            r_neg_src = r_src_t.repeat_interleave(self.neg_ratio)
            r_neg_score = self._dot_score(h, r_neg_src, r_neg_dst)
            r_logits = torch.cat([r_pos_score, r_neg_score], dim=0)
            r_labels = torch.cat([torch.ones_like(r_pos_score), torch.zeros_like(r_neg_score)], dim=0)
            if self.use_edge_weights and r_w_t is not None and len(r_w_t) == len(r_pos_score):
                rw = r_w_t / (r_w_t.mean() + 1e-6)
                r_weight = torch.cat([rw, torch.ones_like(r_neg_score)], dim=0)
                right_loss = nn.functional.binary_cross_entropy_with_logits(r_logits, r_labels, weight=r_weight)
            else:
                right_loss = bce(r_logits, r_labels)

            loss = left_loss + right_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()

            mlflow.log_metrics({
                'epoch': epoch,
                'loss': float(loss.item()),
                'left_loss': float(left_loss.item()),
                'right_loss': float(right_loss.item())
            }, step=epoch)
            logging.info(f"Epoch {epoch}: loss={loss.item():.4f} left={left_loss.item():.4f} right={right_loss.item():.4f}")

        # Final embeddings
        model.eval()
        with torch.no_grad():
            h = model(g, {'item': g.nodes['item'].data['feat']}).detach().cpu().numpy().astype('float32')
        # L2 normalize
        n = np.linalg.norm(h, axis=1, keepdims=True); n[n==0]=1.0; h = h / n
        np.save(os.path.join(self.artifacts, 'gnn_embed.npy'), h)
        # Build FAISS index (inner product on normalized vectors = cosine similarity)
        index = faiss.IndexFlatIP(h.shape[1])
        index.add(h)
        faiss.write_index(index, os.path.join(self.artifacts, 'gnn.index'))
        logging.info(f"Saved: gnn_embed.npy shape={h.shape} and gnn.index with {index.ntotal} vectors")
        return h

    def run(self):
        with mlflow.start_run(run_name="gnn-graph-building"):
            # Log parameters
            mlflow.log_params({
                "interim_dir": self.interim,
                "artifacts_dir": self.artifacts,
                "seed": self.seed
            })
            
            self.load_vocab()
            X = self.build_node_features()
            l_src, l_dst, r_src, r_dst, l_w, r_w = self.load_edges()
            g = self.build_heterograph(l_src, l_dst, r_src, r_dst, l_w, r_w)  # topology + weights
            self.save_pretrain_dump(X, l_src, l_dst, r_src, r_dst)
            # Train GNN to refine embeddings via link prediction
            h = self.train_gnn(g, X, l_src, l_dst, r_src, r_dst, l_w=l_w, r_w=r_w)
            
            # Log metrics
            mlflow.log_metrics({
                "num_nodes": self.num_nodes,
                "num_items": len(self.item2idx),
                "left_edges": len(l_src),
                "right_edges": len(r_src),
                "total_edges": len(l_src) + len(r_src),
                "node_features_dim": X.shape[1],
                "gnn_embed_dim": h.shape[1]
            })
            
            # Log graph metadata
            mlflow.log_dict({
                "graph_info": {
                    "num_nodes": self.num_nodes,
                    "num_edges": len(l_src) + len(r_src),
                    "edge_types": ["left", "right"],
                    "node_features_shape": list(X.shape)
                }
            }, "graph_metadata.json")
            
            # Log artifacts
            mlflow.log_artifact(self.artifacts, "gnn_artifacts")
            
            logging.info(f"GNN graph building completed and logged to MLflow")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim",   required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--experiment-name", default="gnn-graph-building", help="MLflow experiment name")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--neg-ratio", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-edge-weights", action="store_true")
    args = ap.parse_args()
    GNNGraphBuilder(
        args.interim,
        args.artifacts,
        seed=args.seed,
        experiment_name=args.experiment_name,
        hidden_dim=args.hidden,
        out_dim=256,
        epochs=args.epochs,
        lr=args.lr,
        neg_ratio=args.neg_ratio,
        use_edge_weights=not args.no_edge_weights
    ).run()
