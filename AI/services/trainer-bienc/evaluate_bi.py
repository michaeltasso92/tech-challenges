import os
import json
import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_artifacts(artifacts_dir: str):
    import faiss

    vocab_path = os.path.join(artifacts_dir, "item_vocab.json")
    left_idx_p = os.path.join(artifacts_dir, "left.index")
    right_idx_p = os.path.join(artifacts_dir, "right.index")
    left_emb_p = os.path.join(artifacts_dir, "embed_left.npy")
    right_emb_p = os.path.join(artifacts_dir, "embed_right.npy")
    seen_p = os.path.join(artifacts_dir, "seen_items.json")
    gnn_idx_p = os.path.join(artifacts_dir, "gnn.index")
    gnn_emb_p = os.path.join(artifacts_dir, "gnn_embed.npy")

    if not all(os.path.exists(p) for p in [vocab_path, left_idx_p, right_idx_p, left_emb_p, right_emb_p, seen_p]):
        missing = [p for p in [vocab_path, left_idx_p, right_idx_p, left_emb_p, right_emb_p, seen_p] if not os.path.exists(p)]
        raise FileNotFoundError(f"Missing required artifact files: {missing}")

    with open(vocab_path) as f:
        vocab = json.load(f)
    rev_vocab = {int(v): k for k, v in vocab.items()}

    idx_l = faiss.read_index(left_idx_p)
    idx_r = faiss.read_index(right_idx_p)
    E_left = np.load(left_emb_p).astype("float32")
    E_right = np.load(right_emb_p).astype("float32")
    for E in (E_left, E_right):
        n = np.linalg.norm(E, axis=1, keepdims=True)
        n[n == 0] = 1
        E /= n

    with open(seen_p) as f:
        seen = json.load(f)

    # Optional GNN artifacts
    idx_g, E_gnn = None, None
    if os.path.exists(gnn_idx_p) and os.path.exists(gnn_emb_p):
        idx_g = faiss.read_index(gnn_idx_p)
        E_gnn = np.load(gnn_emb_p).astype("float32")
        n = np.linalg.norm(E_gnn, axis=1, keepdims=True)
        n[n == 0] = 1
        E_gnn /= n

    return vocab, rev_vocab, idx_l, idx_r, E_left, E_right, seen, idx_g, E_gnn


def build_test_pairs(parsed_path: str, test_items: List[str], use_all: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(parsed_path):
        raise FileNotFoundError(f"parsed.parquet not found at {parsed_path}")

    df = pd.read_parquet(parsed_path)
    df = df[(df["item_id"].notna())]
    df = df[(df["item_id"] != df["left_neighbor"]) & (df["item_id"] != df["right_neighbor"])].copy()

    if not use_all and test_items:
        df = df[df["item_id"].isin(set(test_items))]

    left_df = df.dropna(subset=["left_neighbor"])[["item_id", "left_neighbor"]].rename(columns={"left_neighbor": "nbr"})
    right_df = df.dropna(subset=["right_neighbor"])[["item_id", "right_neighbor"]].rename(columns={"right_neighbor": "nbr"})
    return left_df, right_df


def evaluate_side(index, embeddings: np.ndarray, vocab: Dict[str, int], rev_vocab: Dict[int, str], pairs_df: pd.DataFrame, ks: List[int]) -> Dict[str, float]:
    if len(pairs_df) == 0:
        return {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0, "num_pairs": 0}

    hits_at_k = {k: 0 for k in ks}
    mrr_total = 0.0
    evaluated = 0

    # Group by query item to avoid duplicate searches
    for item_id, group in pairs_df.groupby("item_id"):
        row = vocab.get(item_id)
        if row is None:
            continue

        q = embeddings[row:row + 1]
        D, I = index.search(q, max(max(ks) + 10, 20))
        ranks = {}
        seen = {item_id}
        r = 0
        for idx in I[0]:
            if idx < 0:
                continue
            cand = rev_vocab.get(int(idx))
            if not cand or cand in seen:
                continue
            r += 1
            seen.add(cand)
            ranks[cand] = r
            if r >= max(ks) + 5:
                break

        for nbr in group["nbr"].tolist():
            if nbr not in ranks:
                continue
            evaluated += 1
            rank = ranks[nbr]
            mrr_total += 1.0 / rank
            for k in ks:
                if rank <= k:
                    hits_at_k[k] += 1

    if evaluated == 0:
        return {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0, "num_pairs": 0}

    metrics = {f"recall@{k}": hits_at_k[k] / evaluated for k in ks}
    metrics["mrr"] = mrr_total / evaluated
    metrics["num_pairs"] = evaluated
    return metrics


def evaluate_gnn(index, embeddings: np.ndarray, vocab: Dict[str, int], rev_vocab: Dict[int, str], left_df: pd.DataFrame, right_df: pd.DataFrame, ks: List[int]) -> Dict[str, float]:
    # Merge left/right pairs for symmetric evaluation
    pairs_df = pd.concat([left_df, right_df], axis=0, ignore_index=True)
    return evaluate_side(index, embeddings, vocab, rev_vocab, pairs_df, ks)


def maybe_log_to_mlflow(metrics: Dict[str, float], params: Dict[str, str]):
    try:
        import mlflow
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "bi-encoder-eval"))
        with mlflow.start_run(run_name="bi-encoder-eval"):
            mlflow.log_params(params)
            # Sanitize metric names for MLflow (no '@')
            safe_metrics = {}
            for k, v in metrics.items():
                safe_k = k.replace('@', '_at_')
                safe_metrics[safe_k] = v
            mlflow.log_metrics(safe_metrics)
    except Exception as e:
        logging.warning(f"MLflow logging skipped: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True, help="Path to artifacts directory produced by training")
    ap.add_argument("--in", dest="inp", required=True, help="Interim directory containing parsed.parquet")
    ap.add_argument("--k", default="1,3,5,10", help="Comma-separated list of K values for recall@K")
    ap.add_argument("--no-test-split", action="store_true", help="Ignore seen_items test split and evaluate on all data in parsed.parquet")
    ap.add_argument("--register-model", dest="register_model", default=os.getenv("REGISTER_MODEL_NAME"), help="If set, register this artifacts dir as a model version")
    ap.add_argument("--register-alias", dest="register_alias", default=os.getenv("REGISTER_MODEL_ALIAS","Staging"))
    args = ap.parse_args()

    ks = [int(x) for x in args.k.split(",")]
    vocab, rev_vocab, idx_l, idx_r, E_left, E_right, seen, idx_g, E_gnn = load_artifacts(args.artifacts)
    test_items = seen.get("test_items", [])
    left_df, right_df = build_test_pairs(os.path.join(args.inp, "parsed.parquet"), test_items, use_all=args.no_test_split)

    # Fallback: if split yielded no pairs (e.g., fixture mismatch), evaluate on full data
    if (len(left_df) + len(right_df)) == 0 and not args.no_test_split:
        print("No pairs found for test split; falling back to full dataset evaluation")
        left_df, right_df = build_test_pairs(os.path.join(args.inp, "parsed.parquet"), test_items=[], use_all=True)

    logging.info(f"Evaluating on {len(test_items)} test items | left_pairs={len(left_df)} right_pairs={len(right_df)}")

    left_metrics = evaluate_side(idx_l, E_left, vocab, rev_vocab, left_df, ks)
    right_metrics = evaluate_side(idx_r, E_right, vocab, rev_vocab, right_df, ks)
    gnn_metrics = None
    if idx_g is not None and E_gnn is not None:
        gnn_metrics = evaluate_gnn(idx_g, E_gnn, vocab, rev_vocab, left_df, right_df, ks)

    # Aggregate by averaging left/right
    agg = {}
    for k in ks:
        agg[f"recall@{k}"] = (left_metrics.get(f"recall@{k}", 0.0) + right_metrics.get(f"recall@{k}", 0.0)) / 2.0
    agg["mrr"] = (left_metrics.get("mrr", 0.0) + right_metrics.get("mrr", 0.0)) / 2.0
    agg["num_pairs"] = left_metrics.get("num_pairs", 0) + right_metrics.get("num_pairs", 0)

    print("Left:", left_metrics)
    print("Right:", right_metrics)
    print("Aggregate:", agg)
    if gnn_metrics is not None:
        print("GNN:", gnn_metrics)

    ml_metrics = {f"left_{k}": v for k, v in left_metrics.items()} | {f"right_{k}": v for k, v in right_metrics.items()} | {f"agg_{k}": v for k, v in agg.items()}
    if gnn_metrics is not None:
        ml_metrics |= {f"gnn_{k}": v for k, v in gnn_metrics.items()}
    maybe_log_to_mlflow(ml_metrics,
                        {"artifacts": args.artifacts, "interim": args.inp, "ks": args.k})

    # Optional: register this artifacts directory as a model version
    if args.register_model:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()
            try:
                client.create_registered_model(args.register_model)
            except Exception:
                pass
            # Create an ad-hoc run to tie registration
            with mlflow.start_run(run_name="bi-encoder-eval-register") as run:
                mlflow.log_artifacts(args.artifacts, artifact_path="artifacts")
                mv = client.create_model_version(name=args.register_model,
                                                 source=f"runs:/{run.info.run_id}/artifacts/artifacts",
                                                 run_id=run.info.run_id)
                if args.register_alias:
                    try:
                        client.set_registered_model_alias(args.register_model, args.register_alias, mv.version)
                    except Exception:
                        pass
                print(f"Registered {args.register_model} v{mv.version} with alias {args.register_alias}")
        except Exception as e:
            print(f"Model registry registration failed: {e}")


if __name__ == "__main__":
    main()


