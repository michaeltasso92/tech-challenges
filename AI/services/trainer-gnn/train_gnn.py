import os, json, argparse, logging
import numpy as np, pandas as pd
from tqdm import tqdm
import dgl
import torch
from typing import Dict, Tuple
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

class GNNGraphBuilder:
    def __init__(self, interim_dir: str, artifacts_dir: str, seed: int = 42, experiment_name: str = "gnn-graph-building"):
        self.interim = interim_dir
        self.artifacts = artifacts_dir
        self.seed = seed
        self.experiment_name = experiment_name
        os.makedirs(self.artifacts, exist_ok=True)
        self.item2idx: Dict[str,int] = {}
        self.idx2item: Dict[int,str] = {}
        self.num_nodes = 0
        
        # Setup MLflow
        mlflow.set_experiment(self.experiment_name)

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

    def load_embeddings(self) -> np.ndarray:
        # Concatenate left & right bi-encoder embeddings as node features
        el_p = os.path.join(self.artifacts, "embed_left.npy")
        er_p = os.path.join(self.artifacts, "embed_right.npy")
        if not (os.path.exists(el_p) and os.path.exists(er_p)):
            raise FileNotFoundError("embed_left.npy or embed_right.npy not found; run bi-encoder trainer first")
        E_left  = np.load(el_p).astype("float32")
        E_right = np.load(er_p).astype("float32")
        if E_left.shape[0] != self.num_nodes or E_right.shape[0] != self.num_nodes:
            raise ValueError(f"Embedding rows mismatch vocab: {E_left.shape[0]} vs {self.num_nodes}")
        X = np.concatenate([E_left, E_right], axis=1)
        # L2 normalize
        n = np.linalg.norm(X, axis=1, keepdims=True); n[n==0]=1.0; X = X / n
        logging.info(f"Node features: X shape={X.shape} (concat of left/right bi-encoder)")
        return X

    def load_edges(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return directed edges for left and right as (src,dst) per relation."""
        parsed_p = os.path.join(self.interim, "parsed.parquet")
        if not os.path.exists(parsed_p):
            raise FileNotFoundError("parsed.parquet not found; run parser first")
        df = pd.read_parquet(parsed_p)
        # Drop self and nulls
        df = df[(df["item_id"].notna())]
        df = df[(df["item_id"] != df["left_neighbor"]) & (df["item_id"] != df["right_neighbor"])]
        left_df  = df.dropna(subset=["left_neighbor"])[["item_id","left_neighbor"]].drop_duplicates()
        right_df = df.dropna(subset=["right_neighbor"])[["item_id","right_neighbor"]].drop_duplicates()

        def to_idx_pairs(sub: pd.DataFrame, col: str):
            rows = []
            for r in sub.itertuples(index=False):
                u = self.item2idx.get(r.item_id, None)
                v = self.item2idx.get(getattr(r, col), None)
                if u is None or v is None: continue
                rows.append((u, v))
            if not rows: return np.empty((0,),dtype=np.int64), np.empty((0,),dtype=np.int64)
            a = np.asarray(rows, dtype=np.int64)
            return a[:,0], a[:,1]

        l_src, l_dst = to_idx_pairs(left_df,  "left_neighbor")
        r_src, r_dst = to_idx_pairs(right_df, "right_neighbor")
        logging.info(f"Edges: LEFT {len(l_src):,} | RIGHT {len(r_src):,}")
        return l_src, l_dst, r_src, r_dst

    def build_heterograph(self, l_src, l_dst, r_src, r_dst) -> dgl.DGLHeteroGraph:
        data_dict = {
            ('item','left','item'):  (torch.as_tensor(l_src), torch.as_tensor(l_dst)),
            ('item','right','item'): (torch.as_tensor(r_src), torch.as_tensor(r_dst)),
        }
        g = dgl.heterograph(data_dict, num_nodes_dict={'item': self.num_nodes})
        logging.info(g)
        return g

    def save_pretrain_dump(self, X: np.ndarray, l_src, l_dst, r_src, r_dst):
        # Save as numpy for quick reload in the trainer step
        np.save(os.path.join(self.artifacts, "gnn_X.npy"), X)
        np.save(os.path.join(self.artifacts, "gnn_left_src.npy"),  l_src)
        np.save(os.path.join(self.artifacts, "gnn_left_dst.npy"),  l_dst)
        np.save(os.path.join(self.artifacts, "gnn_right_src.npy"), r_src)
        np.save(os.path.join(self.artifacts, "gnn_right_dst.npy"), r_dst)
        logging.info("Saved: gnn_X.npy, gnn_left/right_{src,dst}.npy")

    def run(self):
        with mlflow.start_run(run_name="gnn-graph-building"):
            # Log parameters
            mlflow.log_params({
                "interim_dir": self.interim,
                "artifacts_dir": self.artifacts,
                "seed": self.seed
            })
            
            self.load_vocab()
            X = self.load_embeddings()
            l_src, l_dst, r_src, r_dst = self.load_edges()
            g = self.build_heterograph(l_src, l_dst, r_src, r_dst)  # just to validate topology
            self.save_pretrain_dump(X, l_src, l_dst, r_src, r_dst)
            
            # Log metrics
            mlflow.log_metrics({
                "num_nodes": self.num_nodes,
                "num_items": len(self.item2idx),
                "left_edges": len(l_src),
                "right_edges": len(r_src),
                "total_edges": len(l_src) + len(r_src),
                "node_features_dim": X.shape[1]
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
    args = ap.parse_args()
    GNNGraphBuilder(args.interim, args.artifacts, args.seed, args.experiment_name).run()
