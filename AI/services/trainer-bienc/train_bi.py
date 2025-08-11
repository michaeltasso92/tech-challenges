import os, json, logging, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = ""               # hard-disable CUDA
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np, pandas as pd
from typing import Dict, Tuple, List
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch; torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS","4")))
from sentence_transformers import SentenceTransformer, InputExample, losses, LoggingHandler
import faiss  # faiss-cpu

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                    level=logging.INFO, handlers=[LoggingHandler()])

class BiEncoderTrainer:
    def __init__(self, base_model: str, interim_dir: str, artifacts_dir: str,
                 epochs: int = 2, batch_size: int = 256, dim: int = 384, seed: int = 42):
        self.base_model = base_model
        self.interim_dir = interim_dir
        self.artifacts_dir = artifacts_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.dim = dim
        self.seed = seed
        self.text_map: Dict[str,str] = {}
        self.item2idx: Dict[str,int] = {}
        self.items: List[str] = []
        os.makedirs(self.artifacts_dir, exist_ok=True)

    @staticmethod
    def _row_text(name: str, brand: str, folders, typ: str, code: str) -> str:
        brand = (brand or "").strip() if isinstance(brand, str) else ""
        name = (name or "").strip() if isinstance(name, str) else ""
        
        if isinstance(folders, (list, tuple, set)):
            folders_str = " > ".join(str(x) for x in folders if x)
        elif isinstance(folders, np.ndarray):
            folders_str = " > ".join(str(x) for x in folders.tolist() if x)
        elif isinstance(folders, str):
            folders_str = folders.strip()
        else:
            folders_str = ""

        typ = (typ or "").strip() if isinstance(typ, str) else ""
        code = (code or "").strip() if isinstance(code, str) else ""

        parts = [brand, name]
        if folders_str: parts.append(folders_str)
        if typ: parts.append(typ)
        if code: parts.append(code)

        return " | ".join(p for p in parts if p)


    def build_text_map(self) -> Dict[str,str]:
        names_p = os.path.join(self.interim_dir, "item_names.parquet")
        meta_p  = os.path.join(self.interim_dir, "item_meta.parquet")
        names = pd.read_parquet(names_p) if os.path.exists(names_p) else pd.DataFrame(columns=["name"])
        meta  = pd.read_parquet(meta_p)  if os.path.exists(meta_p)  else pd.DataFrame()
        df = names.join(meta, how="outer").reset_index().rename(columns={"index":"item_id"})
        records = df.to_dict(orient="records")
        txt = { r["item_id"]: self._row_text(r.get("name"), r.get("brand"), r.get("folders"),
                                             r.get("type"), r.get("code") or r.get("code2") or r.get("code3") or "")
                for r in records }
        for k,v in list(txt.items()):
            if not v or not v.strip(): txt[k]=k
        logging.info(f"text_map built for {len(txt):,} items")
        return txt

    def mine_pairs_from_df(self, parsed_path: str) -> Tuple[pd.DataFrame,pd.DataFrame]:
        df = pd.read_parquet(parsed_path)
        df = df[(df["item_id"] != df["left_neighbor"]) & (df["item_id"] != df["right_neighbor"])]
        left  = df.dropna(subset=["left_neighbor"])[["item_id","left_neighbor"]].drop_duplicates().rename(columns={"left_neighbor":"nbr"})
        right = df.dropna(subset=["right_neighbor"])[["item_id","right_neighbor"]].drop_duplicates().rename(columns={"right_neighbor":"nbr"})
        logging.info(f"LEFT pairs: {len(left):,} | RIGHT pairs: {len(right):,}")
        return left, right

    @staticmethod
    def to_examples(pairs_df: pd.DataFrame, text_map: Dict[str,str]) -> List[InputExample]:
        ex=[]; append=ex.append
        for r in pairs_df.itertuples(index=False):
            a=text_map.get(r.item_id, r.item_id); b=text_map.get(r.nbr, r.nbr)
            if a and b: append(InputExample(texts=[a,b]))
        return ex

    def train_model(self, examples: List[InputExample]) -> SentenceTransformer:
        model = SentenceTransformer(self.base_model, device="cpu")
        model.max_seq_length = 64
        examples = examples[:2000]
        loader = DataLoader(examples, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0)
        loss = losses.MultipleNegativesRankingLoss(model)
        warmup = int(0.06 * self.epochs * len(loader))
        logging.info(f"Training {self.base_model} (CPU) epochs={self.epochs} batch={self.batch_size} warmup={warmup}")
        model.fit(train_objectives=[(loader, loss)], epochs=self.epochs, warmup_steps=warmup, show_progress_bar=True, use_amp=False)
        return model

    def encode_all(self, model: SentenceTransformer) -> np.ndarray:
        texts = [self.text_map[it] for it in self.items]
        emb = model.encode(texts, batch_size=256, normalize_embeddings=True, show_progress_bar=True, device="cpu")
        return emb.astype(np.float32)

    def save_artifacts(self, emb_left: np.ndarray, emb_right: np.ndarray):
        np.save(os.path.join(self.artifacts_dir,"embed_left.npy"),  emb_left)
        np.save(os.path.join(self.artifacts_dir,"embed_right.npy"), emb_right)
        with open(os.path.join(self.artifacts_dir,"item_vocab.json"),"w") as f: json.dump(self.item2idx, f)
        idx_l = faiss.IndexFlatIP(emb_left.shape[1]); idx_r = faiss.IndexFlatIP(emb_right.shape[1])
        idx_l.add(emb_left); idx_r.add(emb_right)
        faiss.write_index(idx_l, os.path.join(self.artifacts_dir,"left.index"))
        faiss.write_index(idx_r, os.path.join(self.artifacts_dir,"right.index"))
        logging.info("Saved artifacts: embed_left.npy, embed_right.npy, item_vocab.json, left.index, right.index")

    def split_by_guideline(self, df, train_ratio=0.8, val_ratio=0.1, seed=42):
        guids = df["guideline_id"].dropna().unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(guids)

        n_train = int(len(guids) * train_ratio)
        n_val = int(len(guids) * val_ratio)

        train_guids = set(guids[:n_train])
        val_guids   = set(guids[n_train:n_train+n_val])
        test_guids  = set(guids[n_train+n_val:])

        return (
            df[df["guideline_id"].isin(train_guids)],
            df[df["guideline_id"].isin(val_guids)],
            df[df["guideline_id"].isin(test_guids)]
        )


    def run(self):
        parsed_p = os.path.join(self.interim_dir,"parsed.parquet")
        if not os.path.exists(parsed_p): raise FileNotFoundError("parsed.parquet not found; run parser first")

        self.text_map = self.build_text_map()
        # Full parsed data
        df_all = pd.read_parquet(parsed_p)
        df_all = df_all[(df_all["item_id"] != df_all["left_neighbor"]) &
                        (df_all["item_id"] != df_all["right_neighbor"])]

        # Split by guideline
        df_train, df_val, df_test = self.split_by_guideline(df_all)

        # Mine train pairs
        left_df, right_df = self.mine_pairs_from_df(df_train)  # new helper

        # Save seen items for debug
        seen = {
            "train_items": sorted(set(left_df["item_id"]) | set(right_df["item_id"])),
            "val_items": sorted(set(df_val["item_id"].unique())),
            "test_items": sorted(set(df_test["item_id"].unique()))
        }
        with open(os.path.join(self.artifacts_dir, "seen_items.json"), "w") as f:
            json.dump(seen, f)

        logging.info(f"Train items: {len(seen['train_items'])}, Val items: {len(seen['val_items'])}, Test items: {len(seen['test_items'])}")


        left_ex  = self.to_examples(left_df,  self.text_map)
        right_ex = self.to_examples(right_df, self.text_map)
        if len(left_ex) < 10 or len(right_ex) < 10:
            raise RuntimeError("Not enough training pairs; check parser output")

        self.items = sorted(self.text_map.keys()); self.item2idx = {it:i for i,it in enumerate(self.items)}

        logging.info("Training LEFT bi-encoder (CPU)…")
        model_left  = self.train_model(left_ex)
        logging.info("Training RIGHT bi-encoder (CPU)…")
        model_right = self.train_model(right_ex)

        logging.info("Encoding corpus with LEFT encoder (CPU)…")
        emb_left  = self.encode_all(model_left)
        logging.info("Encoding corpus with RIGHT encoder (CPU)…")
        emb_right = self.encode_all(model_right)

        self.save_artifacts(emb_left, emb_right)

        model_left.save(os.path.join(self.artifacts_dir,"bienc_left_model"))
        model_right.save(os.path.join(self.artifacts_dir,"bienc_right_model"))
        logging.info({"items": len(self.items), "left_dim": emb_left.shape[1], "right_dim": emb_right.shape[1]})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dim", type=int, default=384)
    args = ap.parse_args()
    BiEncoderTrainer(args.base, args.inp, args.artifacts, args.epochs, args.batch, args.dim).run()
