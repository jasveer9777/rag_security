# src/test_retrieve.py
"""
Quick local retrieval test using the FAISS index.
Usage:
  python src/test_retrieve.py --query "When did CRPC come into effect?" --k 5
"""
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse
from pathlib import Path

INDEX_DIR = "index"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_index():
    idx_path = Path(INDEX_DIR) / "index.faiss"
    metas_path = Path(INDEX_DIR) / "metas.pkl"
    if not idx_path.exists() or not metas_path.exists():
        raise FileNotFoundError("Index not found; run embed_index.py first.")
    index = faiss.read_index(str(idx_path))
    metas = pickle.load(open(str(metas_path), "rb"))
    return index, metas

def retrieve(index, metas, model, query, k=5):
    qv = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(qv)
    dists, ids = index.search(qv, k)
    results = []
    for pos, i in enumerate(ids[0]):
        results.append((metas[i], float(dists[0][pos])))
    return results

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, required=True)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    index, metas = load_index()
    model = SentenceTransformer(EMB_MODEL)
    res = retrieve(index, metas, model, args.query, k=args.k)
    for meta, score in res:
        print("SCORE:", score, "SOURCE:", meta.get("source_path"), "CHUNK_IDX:", meta.get("chunk_index"))
        print((meta.get("chunk_text") or "")[:400].replace("\n"," "), "...\n")
