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

st.set_page_config(page_title="IWD Recommender", page_icon="🧠", layout="wide")
st.title("🧠 IWD Product Placement Recommender")
st.caption("Enter an item ID to get left/right placement suggestions with confidence scores.")

col1, col2, col3 = st.columns([2,1,1])
with col1:
    item = st.text_input("Item ID", value=DEFAULT_ITEMS[0], help="Paste a catalog item id")
with col2:
    k = st.number_input("Top-K", min_value=1, max_value=50, value=10, step=1)
with col3:
    if st.button("Recommend", use_container_width=True):
        st.session_state["trigger"] = True

st.divider()

if "trigger" in st.session_state and st.session_state["trigger"]:
    with st.spinner("Querying API..."):
        try:
            r = requests.get(f"{API_BASE}/recommend/{item}", timeout=15)
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    st.subheader(f"Results for: `{payload.get('item_id', item)}`")
    left_df = pd.DataFrame(payload["left"][:k])
    right_df = pd.DataFrame(payload["right"][:k])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**⬅️ Left neighbors**")
        if len(left_df):
            if "name" in left_df.columns:
                left_df = left_df[["item","name","confidence"]]
            st.dataframe(left_df.rename(columns={"item":"neighbor ID","name":"neighbor Name"}))
            st.bar_chart(left_df.set_index("item")["confidence"])
        else:
            st.info("No left neighbors found (fallback used).")

    with c2:
        st.markdown("**➡️ Right neighbors**")
        if len(right_df):
            if "name" in right_df.columns:
                right_df = right_df[["item","name","confidence"]]
            st.dataframe(right_df.rename(columns={"item":"neighbor ID","name":"neighbor Name"}))
            st.bar_chart(right_df.set_index("item")["confidence"])
        else:
            st.info("No right neighbors found (fallback used).")

st.sidebar.header("Quick picks")
for iid in DEFAULT_ITEMS:
    if st.sidebar.button(iid):
        st.session_state["trigger"] = True
        st.experimental_rerun()
