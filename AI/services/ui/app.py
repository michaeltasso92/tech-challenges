import os, requests, pandas as pd, streamlit as st

API_BASE = os.getenv("API_BASE_URL","http://localhost:8000")
DEFAULT_ITEMS = [
    "1771c4a1-d902-11e4-a7e5-0025904e7aec",
    "1797ef88-d902-11e4-a7e5-0025904e7aec",
    "17c45bee-d902-11e4-a7e5-0025904e7aec",
    "31377edb-bc97-11ec-80d8-028cc0c9a267",
    "1964dca0-bc97-11ec-80d8-028cc0c9a267",
    "1797e209-d902-11e4-a7e5-0025904e7aec",
    "d5d8df12-c3a1-11eb-a03c-064c87f59bd9",
    "9df3b27c-c3a1-11eb-a03c-064c87f59bd9",
    "b82c22c1-5b7d-11ed-b2f6-028cc0c9a267",
    "5b65d592-6041-11ed-b2f6-028cc0c9a267",
]

st.set_page_config(page_title="IWD Recommender", page_icon="🛍️🏬", layout="wide")
st.title("🛍️🏬 IWD Product Placement Recommender")
st.caption("Select a product to get left/right placement suggestions with confidence scores.")

# --- Simple name cache in session state
if "name_cache" not in st.session_state: st.session_state["name_cache"] = {}

def fetch_name(item_id: str) -> str | None:
    if not item_id: return None
    cache = st.session_state["name_cache"]
    if item_id in cache: return cache[item_id]
    # Try a dedicated API (if you added it)
    try:
        r = requests.get(f"{API_BASE}/name/{item_id}", timeout=10)
        if r.status_code == 200:
            name = r.json().get("name") or None
            cache[item_id] = name
            return name
    except Exception:
        pass
    # As a fallback, try /recommend and hope the API returns item_name
    try:
        r = requests.get(f"{API_BASE}/recommend/{item_id}", timeout=10)
        if r.status_code == 200:
            name = r.json().get("item_name") or None
            cache[item_id] = name
            return name
    except Exception:
        pass
    cache[item_id] = None
    return None

def label_for(item_id: str) -> str:
    name = fetch_name(item_id)
    return f"{name} — {item_id}" if name else item_id

# Prewarm labels for default items (non-blocking best effort)
for _iid in DEFAULT_ITEMS:
    try: fetch_name(_iid)
    except Exception: pass

# --- Controls
col1, col2 = st.columns([2,1])
with col1:
    selected_id = st.selectbox(
        "Product",
        options=DEFAULT_ITEMS,
        format_func=label_for,
        index=0
    )
with col2:
    k = st.number_input("Top-K", min_value=1, max_value=50, value=10, step=1)

run = st.button("Recommend", use_container_width=True)
st.divider()

if run:
    with st.spinner("Querying API..."):
        try:
            r = requests.get(f"{API_BASE}/recommend/{selected_id}", timeout=20)
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    # Resolve display name for the selected product
    selected_name = payload.get("item_name") or fetch_name(selected_id) or selected_id
    st.subheader(f"Results for: **{selected_name}**")

    # Build neighbor dataframes; use names if present, else look them up
    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        # If API already included names, keep them; otherwise fetch
        if "name" not in df.columns:
            df["name"] = [fetch_name(x) for x in df["item"]]
        df["neighbor"] = df.apply(lambda r: r["name"] if r["name"] else r["item"], axis=1)
        return df[["neighbor","item","confidence"]]

    left_df  = enrich(pd.DataFrame(payload.get("left",  [])[:k]))
    right_df = enrich(pd.DataFrame(payload.get("right", [])[:k]))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**⬅️ Left neighbors**")
        if len(left_df):
            st.dataframe(left_df.rename(columns={"item":"neighbor ID"}), use_container_width=True)
            # Use neighbor names on the chart if available
            chart_series = left_df.set_index("neighbor")["confidence"]
            st.bar_chart(chart_series)
        else:
            st.info("No left neighbors found (fallback used).")

    with c2:
        st.markdown("**➡️ Right neighbors**")
        if len(right_df):
            st.dataframe(right_df.rename(columns={"item":"neighbor ID"}), use_container_width=True)
            chart_series = right_df.set_index("neighbor")["confidence"]
            st.bar_chart(chart_series)
        else:
            st.info("No right neighbors found (fallback used).")
