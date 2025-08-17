import os
import unicodedata
import requests
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

API_BASE = os.getenv("API_BASE_URL","http://localhost:8000")
TOP_K = 10
SEEN_COLORS = {"train":"#2563eb","val":"#16a34a","test":"#f59e0b","unseen":"#6b7280"}

# Target items from the README (top 10 most frequent items)
# NOTE: These items don't exist in the current dataset, so we're using the actual top 10
TARGET_ITEMS = [
    "b82c22c1-5b7d-11ed-b2f6-028cc0c9a267",  # Actual #1 (freq: 863)
    "25c7c4cf-5b7e-11ed-b2f6-028cc0c9a267",  # Actual #2 (freq: 844)
    "1772df19-d902-11e4-a7e5-0025904e7aec",  # Actual #3 (freq: 812)
    "1e4361a6-d902-11e4-a7e5-0025904e7aec",  # Actual #4 (freq: 764)
    "dbadc572-7277-11ee-aee6-028cc0c9a267",  # Actual #5 (freq: 754)
    "0e16abdc-612a-11ee-aee6-028cc0c9a267",  # Actual #6 (freq: 672)
    "fde1a0b2-6129-11ee-aee6-028cc0c9a267",  # Actual #7 (freq: 648)
    "10c467dc-5b7e-11ed-b2f6-028cc0c9a267",  # Actual #8 (freq: 647)
    "11f5b934-4500-11ef-bf5c-0ae80689e703",  # Actual #9 (freq: 643)
    "2daed28e-ceff-11ed-b2f6-028cc0c9a267"   # Actual #10 (freq: 634)
]

# Fallback defaults if /names isn't available
DEFAULT_ITEMS = TARGET_ITEMS

st.set_page_config(page_title="IWD Recommender", page_icon="🛍️", layout="wide")
st.title("🛍️ IWD Product Placement Recommender")
st.caption("Search by product name or pick a popular item. Returns top-10 left/right neighbors with confidence and seen/cold-start flags.")

if "name_cache" not in st.session_state: st.session_state["name_cache"] = {}

def norm(s:str)->str:
    if not s: return ""
    return unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii").lower()

def seen_badge(label:str)->str:
    c=SEEN_COLORS.get(label,"#6b7280")
    return f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;background:{c};color:#fff;font-size:12px">{label.upper()}</span>'

@st.cache_data(show_spinner=False)
def fetch_names():
    try:
        r=requests.get(f"{API_BASE}/names",timeout=15); r.raise_for_status()
        data = r.json()
        return data.get("names", data) 
    except Exception:
        return {}

def fetch_name(item_id:str):
    cache=st.session_state["name_cache"]
    if item_id in cache: return cache[item_id]
    try:
        r=requests.get(f"{API_BASE}/name/{item_id}",timeout=10)
        if r.ok:
            n=r.json().get("name")
            cache[item_id]=n
            return n
    except Exception:
        pass
    cache[item_id]=None
    return None

def label_for(item_id:str, names:dict)->str:
    n = names.get(item_id) or fetch_name(item_id)
    if n:
        return f"{n} — {item_id}"
    else:
        # Show a shortened UUID for items without names
        short_id = item_id[:8] + "..." if len(item_id) > 8 else item_id
        return f"Unknown Product ({short_id}) — {item_id}"

def get_recs(item_id:str):
    r=requests.get(f"{API_BASE}/recommend/{item_id}",params={"k":TOP_K},timeout=30)
    r.raise_for_status()
    return r.json()

def get_image_urls(item_id:str):
    """Get image URLs for a specific item"""
    try:
        r = requests.get(f"{API_BASE}/images/{item_id}", timeout=10)
        if r.ok:
            data = r.json()
            return data.get("image_urls", [])
    except Exception:
        pass
    return []

def _neighbor_card_html(name: str, item_id: str, score: float, seen: str, image_url: str = "") -> str:
    img = (
        f'<img src="{image_url}" alt="{name}" '
        f'style="width:auto; height:80px; border-radius:8px; margin-bottom:8px; background:white; padding:5px; object-fit:contain;"/>'
        if image_url else ""
    )
    return (
        f'<div style="background:white; border-radius:12px; padding:14px; margin-bottom:12px; '
        f'box-shadow:0 2px 6px rgba(0,0,0,0.08); border-left:4px solid #3b82f6; text-align:center; '
        f'display:flex; flex-direction:column; align-items:center; gap:6px; box-sizing:border-box; '
        f'min-height:220px; overflow:hidden;">'
        f'{img}'
        f'<div style="font-weight:600; color:#1e293b; font-size:14px; margin:0; '
        f'max-height:40px; line-height:20px; overflow:hidden; text-overflow:ellipsis; '
        f'display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; word-break:break-word;">{name}</div>'
        f'<div style="font-size:11px; color:#64748b; font-family:monospace; margin:0; word-break:break-all;">{item_id}</div>'
        f'<div style="display:flex; justify-content:space-between; align-items:center; gap:8px; font-size:12px;">'
        f'<span style="background:#f1f5f9; padding:2px 8px; border-radius:12px; color:#475569; font-weight:500;">Score: {float(score):.3f}</span>'
        f'{seen_badge(str(seen))}'
        f'</div>'
        f'</div>'
    )

def _selected_card_html(name: str, item_id: str, image_url: str | None, seen: str) -> str:
    img = (
        f'<img src="{image_url}" alt="{name}" '
        f'style="width:auto; height:150px; border-radius:10px; margin-bottom:12px; background:white; padding:10px; object-fit:contain;"/>'
        if image_url else ""
    )
    return (
        f'<div style="background:linear-gradient(135deg,#3b82f6 0%,#1d4ed8 100%); border-radius:20px; padding:24px; text-align:center; '
        f'color:white; box-shadow:0 10px 15px -3px rgba(0,0,0,0.1); margin:20px 0; min-height:360px; box-sizing:border-box; overflow:hidden;">'
        f'{img}'
        f'<h3 style="margin:0 0 8px 0; font-size:20px; line-height:22px; max-height:44px; overflow:hidden; text-overflow:ellipsis; '
        f'display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; word-break:break-word;">{name}</h3>'
        f'<div style="margin-bottom:12px;">{seen_badge(seen)}</div>'
        f'<div style="font-size:12px; opacity:0.9; font-family:monospace; word-break:break-all;">{item_id}</div>'
        f'<div style="margin-top:12px; font-size:14px; opacity:0.85;">Selected Product</div>'
        f'</div>'
    )

# Check for missing product names
names = fetch_names()
if names:
    missing_count = len([item for item in TARGET_ITEMS if item not in names])
    if missing_count > 0:
        st.warning(f"⚠️ {missing_count}/10 target items are missing product names. Some recommendations may show as 'Unknown Product'.")

all_ids = list(names.keys())

# Create popular items list with target items prioritized at the top
# Always include target items, even if they don't have names
available_targets = [item for item in TARGET_ITEMS if item in all_ids]
missing_targets = [item for item in TARGET_ITEMS if item not in all_ids]

# Get remaining items sorted alphabetically (excluding target items)
remaining_items = sorted([item for item in all_ids if item not in TARGET_ITEMS], 
                       key=lambda x: (names.get(x) or x))[:290]  # Leave room for targets

# Combine: available targets first, then missing targets, then remaining items
popular_ids = available_targets + missing_targets + remaining_items

# ---------- Controls (search + popular + paste) ----------
cL, cR = st.columns([2,1])

with cL:
    st.subheader("Search by name")
    q = st.text_input("Type part of a product name (e.g., lip, serum, black)")
    search_matches = []
    if q and names:
        nq = norm(q)
        # simple substring match on normalized names
        search_matches = [{"id": iid, "name": names[iid]} for iid in all_ids if nq in norm(names.get(iid,""))]
        if search_matches:
            choice = st.selectbox(
                "Matches",
                options=search_matches,
                format_func=lambda x: f"{x['name']} — {x['id']}",
                key="search_select"
            )
        else:
            st.warning("No products found. Try a different keyword.")
            choice = None
    elif q and not names:
        st.info("Name search requires the /names endpoint. Falling back to popular/paste.")
        choice = None
    else:
        choice = None

with cR:
    st.subheader("Target Items & Popular Products")
    selected_pop = st.selectbox(
        "Select from target items (top 10) or popular products",
        options=popular_ids,
        format_func=lambda x: label_for(x,names),
        index=0,
        key="popular_select"
    )
    available_target_count = len([item for item in TARGET_ITEMS if item in all_ids])
    st.caption(f"📋 {available_target_count}/10 target items available (actual top 10 most frequent items) • {len(popular_ids)} total products")
    st.caption("Or paste an Item ID (UUID or code):")
    typed_id = st.text_input("Item ID", key="typed_id")

run = st.button("Recommend", use_container_width=True)
st.divider()

# Resolve selection priority: search > paste > popular
selected_id = None
if run:
    if choice: selected_id = choice["id"]
    elif typed_id and typed_id.strip(): selected_id = typed_id.strip()
    else: selected_id = selected_pop

# ---------- Results ----------
if run and selected_id:
    with st.spinner("Querying API..."):
        try:
            payload = get_recs(selected_id)
        except Exception as e:
            st.error(f"API error: {e}"); st.stop()

    # Get the product name, with fallback for missing names
    qname = names.get(selected_id) or fetch_name(selected_id)
    if not qname:
        # Show a shortened UUID for items without names
        short_id = selected_id[:8] + "..." if len(selected_id) > 8 else selected_id
        qname = f"Unknown Product ({short_id})"
    qseen = payload.get("query_seen","unseen")
    
    # Get image URLs for the selected product
    selected_images = get_image_urls(selected_id)
    
    def to_df(items:list)->pd.DataFrame:
        if not items: return pd.DataFrame(columns=["neighbor","item","confidence","seen"])
        df=pd.DataFrame(items)
        if "name" not in df.columns:
            df["name"]=[names.get(x) or fetch_name(x) for x in df["item"]]
        df["neighbor"]=df.apply(lambda r: r.get("name") or r["item"],axis=1)
        if "seen" not in df.columns: df["seen"]="unseen"
        if "confidence" not in df.columns: df["confidence"]=0.0
        # Add image URLs for each item
        df["image_urls"] = [get_image_urls(item_id) for item_id in df["item"]]
        return df[["neighbor","item","confidence","seen","image_urls"]]

    left_df  = to_df(payload.get("left",[])[:TOP_K])
    right_df = to_df(payload.get("right",[])[:TOP_K])

    # Shelf-like layout using Streamlit native components
    st.markdown("### Product Recommendations")
    
    # Create the main shelf container
    left_col, center_col, right_col = st.columns([1, 0.8, 1])
    
    with left_col:
        st.markdown("#### ⬅️ Left Shelf")
        if left_df.empty:
            st.info("No left neighbors found")
        else:
            for _, row in left_df.iterrows():
                with st.container():
                    img_url = row['image_urls'][0] if row['image_urls'] else ""
                    html = _neighbor_card_html(row['neighbor'], row['item'], float(row['confidence']), str(row['seen']), img_url)
                    st_html(html, height=230)
    
    with center_col:
        # Display product image if available
        if selected_images:
            # Use the first image (usually the main product image)
            main_image_url = selected_images[0]
            st_html(_selected_card_html(qname, selected_id, main_image_url, qseen), height=380)
        else:
            # Fallback without image
            st_html(_selected_card_html(qname, selected_id, None, qseen), height=280)
    
    with right_col:
        st.markdown("#### Right Shelf ➡️")
        if right_df.empty:
            st.info("No right neighbors found")
        else:
            for _, row in right_df.iterrows():
                with st.container():
                    img_url = row['image_urls'][0] if row['image_urls'] else ""
                    html = _neighbor_card_html(row['neighbor'], row['item'], float(row['confidence']), str(row['seen']), img_url)
                    st_html(html, height=230)

    # Additional info below the shelf
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Left Neighbors", len(left_df))
    with col2:
        st.metric("Right Neighbors", len(right_df))
    with col3:
        total_recs = len(left_df) + len(right_df)
        st.metric("Total Recommendations", total_recs)
