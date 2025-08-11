import os, json

def load_artifacts(model_dir):
    with open(os.path.join(model_dir,"left.json")) as f: left=json.load(f)
    with open(os.path.join(model_dir,"right.json")) as f: right=json.load(f)
    with open(os.path.join(model_dir,"fallback.json")) as f: fb=json.load(f)
    return left,right,fb
