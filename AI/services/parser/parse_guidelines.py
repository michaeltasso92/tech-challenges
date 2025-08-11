import argparse, os, json, glob
import pandas as pd
from tqdm import tqdm

CATALOG_TYPES = {
    "product","tester","visual","object3D","compositeObject3D",
    "pointOfSaleRepresentation","shoes","cloth","textileAccessory",
    "storeComponent","makeupGrid"
}
ACCESSORY_TYPE = "accessory"
SHELF_LIKE_TYPES = {
    "shelf","verticalSeparator","bayHeader","bayFooter",
    "bayRightSide","bayLeftSide","bayBackPanel","hook","itemHolder",
    # makeupTray parts act like shelves (they contain shelfLikeChildren)
    "makeupTrayBackPanel","makeupTrayFrameBottom","makeupTrayFrameLeft",
    "makeupTrayFrameRight","makeupTrayFrameTop"
}

def is_item_node(x: dict) -> bool:
    t = x.get("type")
    return t in CATALOG_TYPES or t == ACCESSORY_TYPE

def is_shelf_like(x: dict) -> bool:
    t = x.get("type")
    return t in SHELF_LIKE_TYPES

def expand_facings(child: dict):
    """Return a list of item_ids expanded by facing count (>=1)."""
    # accessories don't have 'facing' in schema; treat as 1
    facing = child.get("facing", 1)
    try:
        facing = int(facing) if facing is not None else 1
    except: facing = 1
    facing = max(1, facing)
    iid = child.get("id")
    # keep only items with a non-empty id
    if not isinstance(iid, str) or not iid: return []
    return [iid] * facing

def iter_sequences(node):
    """Yield ordered sequences (lists of item dicts) that represent a single shelf-like row."""
    if not isinstance(node, dict): return
    children = node.get("children")
    if isinstance(children, list) and children:
        # If node is 'shelf-like' or its children are mostly items, treat as a sequence
        child_dicts = [c for c in children if isinstance(c, dict)]
        item_like = [c for c in child_dicts if is_item_node(c)]
        if is_shelf_like(node) or (len(item_like) >= max(2, len(child_dicts)//2)):
            # sequence found
            yield child_dicts
        # Recurse
        for c in child_dicts:
            yield from iter_sequences(c)

def parse_file(path):
    try:
        g = json.load(open(path))
    except Exception:
        return []
    gid = g.get("id") or os.path.basename(path)
    rows = []
    names = {}

    seq_idx = 0
    for seq in iter_sequences(g):
        # Turn child dicts into a 1D list of item_ids, expanded by facing, preserving order
        slot_ids = []
        for child in seq:
            if is_item_node(child):
                slot_ids.extend(expand_facings(child))
                iid = child.get("id")
                if iid and iid not in names:
                    names[iid] = child.get("name") or child.get("name2") or ""
        if len(slot_ids) < 2:
            continue
        seq_idx += 1
        for i, iid in enumerate(slot_ids):
            left_id  = slot_ids[i-1] if i > 0 else None
            right_id = slot_ids[i+1] if i < len(slot_ids)-1 else None
            rows.append({
                "guideline_id": gid,
                "group_seq": seq_idx,
                "pos": i,
                "item_id": iid,
                "left_neighbor": left_id,
                "right_neighbor": right_id,
            })
    return rows, names

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = glob.glob(os.path.join(args.raw, "**/*.json"), recursive=True)
    all_rows, all_names , empty = [], {}, 0
    for f in tqdm(files, desc="parse"):
        r, nm = parse_file(f)
        if not r: empty += 1
        all_rows.extend(r)
        all_names.update(nm)

    df = pd.DataFrame(all_rows)
    outp = os.path.join(args.out, "parsed.parquet")
    if len(df): 
        df.to_parquet(outp)
        pd.Series(all_names, name="name").to_frame().to_parquet(
            os.path.join(args.out, "item_names.parquet")
        )
    print(f"files={len(files)} empty={empty} rows={len(df)} out={outp if len(df) else 'n/a'}")
