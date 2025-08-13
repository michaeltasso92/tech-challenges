import argparse
import os
import json
import glob
import logging
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Types that represent actual retail products for recommendation
PRODUCT_TYPES = {
    "product", "tester", "shoes", "cloth", "textileAccessory"
}

# Types that are display/marketing items (not actual products)
DISPLAY_TYPES = {
    "visual", "object3D", "compositeObject3D", "pointOfSaleRepresentation"
}

# Types that are infrastructure/store components (not products)
INFRASTRUCTURE_TYPES = {
    "storeComponent", "makeupGrid"
}

# Legacy support - keeping accessory for backward compatibility but filtering it out
ACCESSORY_TYPE = "accessory"
SHELF_LIKE_TYPES = {
    "shelf","verticalSeparator","bayHeader","bayFooter",
    "bayRightSide","bayLeftSide","bayBackPanel","hook","itemHolder",
    # makeupTray parts act like shelves (they contain shelfLikeChildren)
    "makeupTrayBackPanel","makeupTrayFrameBottom","makeupTrayFrameLeft",
    "makeupTrayFrameRight","makeupTrayFrameTop"
}

def is_item_node(x: dict) -> bool:
    """Check if a node represents an actual retail product for recommendation."""
    t = x.get("type")
    # Treat legacy "accessory" as item as well (tests expect this)
    return t in PRODUCT_TYPES or t == ACCESSORY_TYPE

def is_valid_product_for_recommendation(x: dict) -> bool:
    """Check if a node is a valid product that should be included in recommendations."""
    # Accessories, display objects and infrastructure are not valid for recommendation
    t = x.get("type")
    if t == ACCESSORY_TYPE or t in DISPLAY_TYPES or t in INFRASTRUCTURE_TYPES:
        return False
    if not is_item_node(x):
        return False
    
    # Must have a valid ID
    iid = x.get("id")
    if not isinstance(iid, str) or not iid.strip():
        return False
    
    # Must have some identifying information (name, brand, or code)
    has_name = bool(x.get("name") or x.get("name2"))
    has_brand = bool(x.get("brand"))
    has_code = bool(x.get("code") or x.get("code2") or x.get("code3"))
    
    return has_name or has_brand or has_code

def is_shelf_like(x: dict) -> bool:
    t = x.get("type")
    return t in SHELF_LIKE_TYPES

def expand_facings(child: dict):
    """Return a list of item_ids expanded by facing count (>=1)."""
    # Keep this utility lenient for unit tests: expand solely based on id/facing
    facing = child.get("facing", 1)
    try:
        facing = int(facing) if facing is not None else 1
    except: 
        facing = 1
    facing = max(1, facing)
    
    iid = child.get("id")
    if not isinstance(iid, str) or not iid.strip():
        return []
    
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
    metas = defaultdict(lambda: {"folders": set(), "markets": set()})
    
    # Statistics for data cleaning
    stats = {
        "total_sequences": 0,
        "valid_sequences": 0,
        "total_items_processed": 0,
        "valid_items_processed": 0,
        "items_by_type": defaultdict(int)
    }

    seq_idx = 0
    for seq in iter_sequences(g):
        stats["total_sequences"] += 1
        
        # Count all items by type for statistics
        for child in seq:
            if isinstance(child, dict):
                stats["total_items_processed"] += 1
                item_type = child.get("type", "unknown")
                stats["items_by_type"][item_type] += 1
        
        # Turn child dicts into a 1D list of item_ids, expanded by facing, preserving order
        slot_ids = []
        for child in seq:
            # Only process valid products for recommendation for neighbor slots
            if is_valid_product_for_recommendation(child):
                stats["valid_items_processed"] += 1
                slot_ids.extend(expand_facings(child))
                iid = child.get("id")
                if iid and iid not in names:
                    names[iid] = child.get("name") or child.get("name2") or ""
                add_meta(metas, child)
            else:
                # Record metadata/names only for accessories (but not other invalid items)
                if child.get("type") == ACCESSORY_TYPE:
                    iid = child.get("id")
                    if iid and iid not in names:
                        names[iid] = child.get("name") or child.get("name2") or ""
                    add_meta(metas, child)
        
        # Skip sequences with too few valid products (need at least 2 for neighbor relationships)
        if len(slot_ids) < 2:
            continue
        
        stats["valid_sequences"] += 1
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
    
    # Log cleaning statistics if there's significant data
    if stats["total_items_processed"] > 0:
        logging.info(f"File {gid}: {stats['valid_items_processed']}/{stats['total_items_processed']} items kept "
                    f"({stats['valid_items_processed']/stats['total_items_processed']*100:.1f}%)")
        logging.info(f"File {gid}: {stats['valid_sequences']}/{stats['total_sequences']} sequences kept "
                    f"({stats['valid_sequences']/stats['total_sequences']*100:.1f}%)")
        if stats["items_by_type"]:
            type_summary = ", ".join([f"{k}:{v}" for k, v in sorted(stats["items_by_type"].items())])
            logging.info(f"File {gid}: Item types found: {type_summary}")
    
    return rows, names, metas

def add_meta(meta_acc, item):
    iid = item.get("id")
    if not iid: return
    m = meta_acc[iid]
    # scalars: keep first non-empty
    for k in ("brand","type","code","code2","code3","name","name2"):
        v = item.get(k)
        if v and k not in m:
            m[k] = v
    # lists: merge unique
    if isinstance(item.get("folders"), list):
        m.setdefault("folders", set()).update([str(x) for x in item["folders"] if x])
    if isinstance(item.get("markets"), list):
        ids = []
        for mk in item["markets"]:
            if isinstance(mk, dict) and mk.get("id"):
                ids.append(str(mk["id"]))
            elif isinstance(mk, str):
                ids.append(mk)
        if ids: m.setdefault("markets", set()).update(ids)
    # Extract image URLs
    if isinstance(item.get("files"), list):
        image_urls = []
        for file_info in item["files"]:
            if isinstance(file_info, dict) and file_info.get("fileURL"):
                image_urls.append(file_info["fileURL"])
        if image_urls:
            m["image_urls"] = image_urls


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = glob.glob(os.path.join(args.raw, "**/*.json"), recursive=True)
    all_rows, all_names , all_meta, empty = [], {}, {}, 0
    
    # Overall statistics
    total_files = len(files)
    total_items_found = 0
    
    for f in tqdm(files, desc="parse"):
        r, nm, mt = parse_file(f)
        if not r: empty += 1
        all_rows.extend(r)
        total_items_found += len(nm)
        
        # prefer first non-empty name seen
        for k,v in nm.items():
            all_names.setdefault(k, v)
        # merge meta
        for iid, m in mt.items():
            dst = all_meta.setdefault(iid, {})
            for k, v in m.items():
                if isinstance(v, set):
                    dst.setdefault(k, set()).update(v)
                elif k not in dst and v:
                    dst[k] = v

    df = pd.DataFrame(all_rows)
    outp = os.path.join(args.out, "parsed.parquet")
    if len(df):
        df.to_parquet(outp)

    # names parquet (index=item_id)
    if all_names:
        pd.Series(all_names, name="name").to_frame().to_parquet(
            os.path.join(args.out, "item_names.parquet")
        )

    # meta parquet (index=item_id)
    if all_meta:
        # normalize: convert sets to sorted lists, choose best name and type/brand fallbacks
        recs = []
        for iid, m in all_meta.items():
            folders = sorted(list(m.get("folders", set())))
            markets = sorted(list(m.get("markets", set())))
            recs.append({
                "item_id": iid,
                "brand": m.get("brand",""),
                "type": m.get("type",""),
                "folders": folders,
                "markets": markets,
                "code": m.get("code",""),
                "code2": m.get("code2",""),
                "code3": m.get("code3",""),
                "image_urls": m.get("image_urls", []),
            })
        meta_df = pd.DataFrame.from_records(recs).set_index("item_id")
        meta_df.to_parquet(os.path.join(args.out, "item_meta.parquet"))

    # Final summary
    print(f"files={len(files)} empty={empty} rows={len(df)}")
    print(f"Total unique items found: {len(all_names)}")
    print(f"Data cleaning summary:")
    print(f"  - Files processed: {total_files}")
    print(f"  - Empty files: {empty}")
    print(f"  - Valid neighbor relationships: {len(df)}")
    print(f"  - Unique products: {len(all_names)}")
    
    if len(df) > 0:
        # Show some sample data
        print(f"\nSample of parsed data:")
        print(df.head(10).to_string())
        
        # Show item frequency distribution
        item_counts = df['item_id'].value_counts()
        print(f"\nTop 10 most frequent items:")
        print(item_counts.head(10).to_string())
