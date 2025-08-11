import argparse, os, json
import pandas as pd
import numpy as np

p=argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True)
p.add_argument("--out", required=True)
a=p.parse_args()
os.makedirs(a.out, exist_ok=True)

df=pd.read_parquet(os.path.join(a.inp,"parsed.parquet"))
names_path = os.path.join(a.inp, "item_names.parquet")
names = pd.read_parquet(names_path).to_dict()["name"]


left=df.dropna(subset=["left_neighbor"]).groupby(
    ["item_id","left_neighbor"]).size().rename("cnt").reset_index()
right=df.dropna(subset=["right_neighbor"]).groupby(
    ["item_id","right_neighbor"]).size().rename("cnt").reset_index()

def score_side(df_pairs, item_col, nei_col, cnt_col="cnt", alpha=5.0):
    item_tot=df_pairs.groupby(item_col)[cnt_col].sum()
    nei_tot=df_pairs.groupby(nei_col)[cnt_col].sum()
    total=df_pairs[cnt_col].sum()
    out={}
    for r in df_pairs.itertuples(index=False):
        i=getattr(r,item_col); n=getattr(r,nei_col); c=getattr(r,cnt_col)
        p_xy=(c+alpha)/(total+alpha*len(item_tot))
        p_x=item_tot.loc[i]/item_tot.sum()
        p_y=nei_tot.loc[n]/nei_tot.sum()
        s=np.log(p_xy/(p_x*p_y + 1e-12) + 1e-12)
        out.setdefault(i,[]).append((n,s))
    scored={}
    for i,lst in out.items():
        arr=np.array([s for _,s in lst])
        w=np.exp(arr-arr.max()); w=w/w.sum()
        scored[i]=sorted(
            [{"item":n,"confidence":float(w[k])} for k,(n,_) in enumerate(lst)],
            key=lambda x:-x["confidence"])
    return scored

left_scores=score_side(left,"item_id","left_neighbor")
right_scores=score_side(right,"item_id","right_neighbor")
scored[i] = sorted(
    [
        {"item": n, "name": names.get(n, ""), "confidence": float(w[k])}
        for k, (n, _) in enumerate(lst)
    ],
    key=lambda x: -x["confidence"]
)

with open(os.path.join(a.out,"left.json"),"w") as f: json.dump(left_scores,f)
with open(os.path.join(a.out,"right.json"),"w") as f: json.dump(right_scores,f)

glob_left=(left.groupby("left_neighbor")["cnt"].sum()
           .sort_values(ascending=False).head(20).index.tolist())
glob_right=(right.groupby("right_neighbor")["cnt"].sum()
           .sort_values(ascending=False).head(20).index.tolist())
with open(os.path.join(a.out,"fallback.json"),"w") as f:
    json.dump({"left":glob_left,"right":glob_right},f)

print("artifacts written to", a.out)
