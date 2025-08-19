# Product Recommendation System — Architecture & Technical Decisions

## Executive summary
- Goal: recommend left/right neighbors for products from hierarchical guideline JSON.
- Pipeline: parse → clean adjacency → bi-encoder training (Sentence-Transformers) → evaluate (Recall@K/MRR) → serve API → CI gates + MLflow promotion.
- Current best: `intfloat/e5-small-v2` fine-tuned; Aggregate R@1 ≈ 0.320, MRR ≈ 0.507 on full-dataset eval.

## Problem framing
- Input: ~6k guideline JSONs (bays → shelves → items).
- Output: for a given `item_id`, top-K neighbors for left and right, with confidences.
- Constraints: robust, reproducible, time-bounded; focus on engineering quality.

## Data parsing & cleaning (AI/services/parser/parse_guidelines.py)
Decisions:
- Keep only valid retail products (exclude display/infrastructure). Legacy `accessory` recorded for metadata but not used for neighbor slots.
- Detect shelf-like 1D sequences (shelf-like nodes or majority-item children).
- Expand items by `facing` (≥1), preserve order.
- Skip sequences with <2 valid products.
Artifacts:
- `AI/data/interim/parsed.parquet` with `guideline_id, group_seq, pos, item_id, left_neighbor, right_neighbor`.
- `item_names.parquet`, `item_meta.parquet` (brand/type/folders/markets/codes/image_urls).
Rationale: cleaner adjacency improves generalization and avoids noise.

## Features
- Text per item: `brand | name | folders | type | code` (robust join with fallbacks to id).

## Model & training (AI/services/trainer-bienc/train_bi.py)
- Dual-encoder (left/right) using Sentence-Transformers + MultipleNegativesRankingLoss.
- Backbones: `intfloat/e5-small-v2` (384d) and `intfloat/e5-base-v2` (768d).
- Split by `guideline_id` into train/val/test for stability.
- Save: `embed_left.npy`, `embed_right.npy`, `left.index`, `right.index`, `item_vocab.json`, `seen_items.json`.
- MLflow: log metrics/artifacts; optional registry registration with alias.
- Tunables (CLI): `--epochs`, `--batch`, `--max-seq-len`, `--lr`, `--warmup-ratio`, `--scheduler`.
Suggested defaults:
- e5-small-v2 (CPU-friendly): epochs 6, batch 32, lr 3e-5, max_len 128, warmup 0.1, cosine.
- e5-base-v2 (GPU): epochs 8–12, batch 32, lr 2e-5, max_len 128, warmup 0.1, cosine.

## Evaluation (AI/services/trainer-bienc/evaluate_bi.py)
- Build pairs from `parsed.parquet` (full dataset in CI via `--no-test-split`).
- For each query, search deeper than K; compute Recall@{1,3,5,10} and MRR; aggregate left/right.
- Gates in CI target realistic thresholds for current data/model.

## Serving (AI/services/api)
- FastAPI endpoints for recommendations, names, images, and model debug.
- Loads FAISS indices + vocab; graceful fallbacks.

## CI/CD ( .github/workflows/ci.yml )
- test: lint → tests → prepare data → download artifacts from MLflow (models:/ or runs:/) → evaluate → gate; hard-fail if gate fails.
- docker-build: build/push images; pruning is best-effort.
- promote-model (push to master only, after test success): alias-based promotion `Staging → Production` with no‑op guard.
- Secrets: `MLFLOW_URL`, `MLFLOW_USER`, `MLFLOW_PASS`, `MLFLOW_ALLOW_PROMOTE='true'`.

## MLflow portability
- Copy both `AI/mlflow/` (DB) and `AI/mlruns/` (artifacts) to migrate local server; or log to a remote server.

## Results snapshot
- e5-small-v2: Aggregate ≈ R@1 0.320, MRR 0.507.
- e5-base-v2 (initial GPU fine‑tune): Aggregate ≈ R@1 0.302, MRR 0.497 → undertrained; improve with longer training.
Recommendation: present small model as current best; outline plan to surpass with base on GPU.

## Data enhancement opportunities
- Add spatial signals:
  - Bounding boxes per item (x, y, width, height) at shelf coordinates; normalize within shelf/bay; derive adjacency, distance, and relative height features.
  - 3D/planogram coordinates if available; convert to 2D shelf-space metrics.
- Store and market context:
  - Store type (flagship, travel retail, boutique), region/country, cluster, seasonality.
  - Geolocation rollups to market/region; use to stratify splits and as features.
- Commercial signals:
  - Sales velocity, facing count, stock-outs; promotions/launch windows.
  - Co-purchase or basket co-occurrence if accessible.
- Visual and textual enrichment:
  - Image embeddings (e.g., CLIP/SigLIP) for product visuals; fuse with text embeddings.
  - OCR of packaging for richer names/codes when missing.
- Data quality:
  - Deduplicate near-identical SKUs, standardize codes, resolve aliasing across markets.
  - Explicit handling of accessories/infrastructure when they affect layout, kept as typed nodes not neighbors.

## Alternative modeling approaches
- Cross-encoder reranking: re-score top-K from bi-encoder with a cross-encoder for higher precision at low K.
- Graph-based ranking without training: personalized PageRank or random-walk scores over the item-neighbor graph as a baseline/feature.
- Matrix factorization / implicit feedback: factorize item–neighbor co-occurrence; combine with text via hybrid scoring.
- Sequence modeling of shelves: sequence-to-sequence or CRF-style local ordering model per shelf row.
- Hard-negative mining: mine confusable neighbors within same brand/category/shelf to improve discrimination.
- Distillation: train small bi-encoder from a strong cross-encoder teacher for better small-model accuracy.

## GNN extensions and input features
- Build an item-shelf graph and train a GNN to predict left/right adjacency or edge strengths.
- Candidate features for node/edge inputs:
  - Node (item): text embedding, image embedding, brand, type, category, sales/velocity, facing count, average position (shelf index, height), store-type frequency, regional popularity.
  - Edge (item–item): co-occurrence count, mean shelf-distance, relative height difference, same-brand flag, same-category flag, historical left/right directionality.
  - Context (store/bay/shelf): store type/region, bay layout descriptors, density, number of shelves, planogram template id.
- Use GNN outputs to rerank top-K from the bi-encoder or as a standalone recommender; optionally directional GNNs (left vs right) or edge classification.

## Key trade‑offs
- Favor clean adjacency over short‑term metric inflation.
- Evaluate on full dataset in CI; allow test-split for research.
- Alias‑based promotion; skip no‑ops.
- Keep serving artifacts lightweight; avoid logging full models by default.

## Quickstart
Parsing:
```bash
cd AI
docker compose up --build parser
```
Training:
```bash
docker compose up --build trainer-bienc
```
Evaluation:
```bash
docker compose run --rm --no-deps trainer-bienc \
  python evaluate_bi.py --in /app/data/interim --artifacts /app/models/artifacts \
  --k 1,3,5,10 --no-test-split
```
API:
```bash
docker compose up --build api
```
MLflow UI:
```bash
docker compose up -d mlflow
```

## Troubleshooting
- Artifacts not downloadable: ensure serve‑artifacts or remote store; CI skips localhost URIs.
- Promotion didn’t run: must be push to master, gates pass, and `MLFLOW_ALLOW_PROMOTE='true'`.
- Repeat promotions: guarded; skipped if Production==Staging.
- Empty training batches: reduce `--batch`.
- FAISS import errors: install `faiss-cpu`.

## Future work
- Longer GPU training for base/large models; hard negative mining; cross‑encoder rerank; calibrated scores; dataset A/B to quantify rule changes.
