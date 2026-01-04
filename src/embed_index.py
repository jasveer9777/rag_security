# src/embed_index.py
"""
Create embeddings for chunks.jsonl and build a FAISS index.
Usage:
  python src/embed_index.py --chunks chunks.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --index_dir index --batch 64
"""
import json
import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from tqdm import tqdm

DEFAULT_CHUNKS = "chunks.jsonl"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_INDEX_DIR = "index"
DEFAULT_BATCH = 64

def load_chunks(chunks_file):
    texts = []
    metas = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item.get("chunk_text") or item.get("text") or item.get("chunk")
            if not text or not text.strip():
                continue
            texts.append(text.strip())
            metas.append(item)
    return texts, metas

def build_embeddings(model_name, texts, batch_size=DEFAULT_BATCH):
    model = SentenceTransformer(model_name)
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    return embeddings

def build_faiss_index(embeddings, index_dir):
    # normalize and use IndexFlatIP for cosine-like similarity
    faiss.normalize_L2(embeddings)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(Path(index_dir) / "index.faiss"))
    return index

def save_metas(metas, index_dir):
    with open(Path(index_dir) / "metas.pkl", "wb") as f:
        pickle.dump(metas, f)

def main(args):
    chunks_file = args.chunks
    model_name = args.model
    index_dir = args.index_dir
    batch = args.batch

    if not Path(chunks_file).exists():
        print(f"[ERROR] Chunks file not found: {chunks_file}")
        return

    print("[STEP] Loading chunks...")
    texts, metas = load_chunks(chunks_file)
    print(f"[INFO] Loaded {len(texts)} chunks")

    print(f"[STEP] Building embeddings with model: {model_name}")
    embeddings = build_embeddings(model_name, texts, batch_size=batch)
    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    print("[STEP] Building FAISS index")
    build_faiss_index(embeddings, index_dir)
    save_metas(metas, index_dir)
    print(f"[DONE] Index saved to '{index_dir}/index.faiss' and metas to '{index_dir}/metas.pkl'")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=str, default=DEFAULT_CHUNKS)
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--index_dir", type=str, default=DEFAULT_INDEX_DIR)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    args = p.parse_args()
    main(args)
