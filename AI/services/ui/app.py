import os, unicodedata, requests, pandas as pd, streamlit as st

API_BASE = os.getenv("API_BASE_URL","http://localhost:8000")
TOP_K = 10
SEEN_COLORS = {"train":"#2563eb","val":"#16a34a","test":"#f59e0b","unseen":"#6b7280"}

# Fallback defaults if /names isn't available
DEFAULT_ITEMS = [
    "1771c4a1-d902-11e4-a7e5-0025904e7aec","1797ef88-d902-11e4-a7e5-0025904e7aec",
    "17c45bee-d902-11e4-a7e5-0025904e7aec","31377edb-bc97-11ec-80d8-028cc0c9a267",
    "1964dca0-bc97-11ec-80d8-028cc0c9a267","1797e209-d902-11e4-a7e5-0025904e7aec",
    "d5d8df12-c3a1-11eb-a03c-064c87f59bd9","9df3b27c-c3a1-11eb-a03c-064c87f59bd9",
    "b82c22c1-5b7d-11ed-b2f6-028cc0c9a267","5b65d592-6041-11ed-b2f6-028cc0c9a267",
]

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
    return f"{n} — {item_id}" if n else item_id

def get_recs(item_id:str):
    r=requests.get(f"{API_BASE}/recommend/{item_id}",params={"k":TOP_K},timeout=30)
    r.raise_for_status()
    return r.json()

names = fetch_names()
all_ids = list(names.keys())
popular_ids = sorted(all_ids, key=lambda x: (names.get(x) or x))[:300] if all_ids else DEFAULT_ITEMS

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
    st.subheader("Popular items")
    selected_pop = st.selectbox(
        "Select a popular item",
        options=popular_ids,
        format_func=lambda x: label_for(x,names),
        index=0,
        key="popular_select"
    )
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

    qname = names.get(selected_id) or fetch_name(selected_id) or selected_id
    qseen = payload.get("query_seen","unseen")
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:10px'><h3 style='margin:0'>{qname}</h3>{seen_badge(qseen)}</div>",
        unsafe_allow_html=True
    )
    st.caption(f"Item ID: {selected_id}")

    def to_df(items:list)->pd.DataFrame:
        if not items: return pd.DataFrame(columns=["neighbor","item","confidence","seen"])
        df=pd.DataFrame(items)
        if "name" not in df.columns:
            df["name"]=[names.get(x) or fetch_name(x) for x in df["item"]]
        df["neighbor"]=df.apply(lambda r: r.get("name") or r["item"],axis=1)
        if "seen" not in df.columns: df["seen"]="unseen"
        if "confidence" not in df.columns: df["confidence"]=0.0
        return df[["neighbor","item","confidence","seen"]]

    left_df  = to_df(payload.get("left",[])[:TOP_K])
    right_df = to_df(payload.get("right",[])[:TOP_K])

    def render_list(title, df:pd.DataFrame):
        st.subheader(title)
        if df.empty:
            st.info("No recommendations."); return
        for _,r in df.iterrows():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"border:1px solid #e5e7eb;border-radius:10px;padding:8px 12px;margin-bottom:6px'>"
                f"<div style='display:flex;flex-direction:column'>"
                f"<div style='font-weight:600'>{r['neighbor']}</div>"
                f"<div style='font-size:12px;color:#6b7280'>{r['item']}</div>"
                f"</div>"
                f"<div style='display:flex;align-items:center;gap:10px'>"
                f"<div style='font-size:12px;color:#374151'>score: {float(r['confidence']):.3f}</div>"
                f"{seen_badge(str(r['seen']))}"
                f"</div>"
                f"</div>", unsafe_allow_html=True
            )

    colL,colR=st.columns(2)
    with colL: render_list("⬅️ Left neighbors", left_df)
    with colR: render_list("➡️ Right neighbors", right_df)
