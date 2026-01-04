# src/ingest_with_metadata.py
"""
Ingest PDFs and paired metadata .txt files, chunk text, and emit chunks.jsonl
Each chunk JSON includes any parsed metadata fields (snake_cased).
Usage:
  python src/ingest_with_metadata.py --data_dir data --out_file chunks.jsonl --chunk_size 1200 --overlap 300
"""

import json
import uuid
import re
from pathlib import Path
from pypdf import PdfReader
import argparse
import sys
from typing import Dict

# ---------- Defaults (tweakable) ----------
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_OVERLAP = 300
DEFAULT_DATA_DIR = "data"
DEFAULT_OUT_FILE = "chunks.jsonl"
METADATA_EXT = ".txt"
ALLOWED_EXT = {".pdf", ".txt", ".md"}
# ------------------------------------------

def text_from_pdf(path: Path) -> str:
    """Extract text from a PDF, adding simple [PAGE n] markers."""
    try:
        reader = PdfReader(str(path))
        pages = []
        for i, p in enumerate(reader.pages):
            txt = p.extract_text()
            if txt:
                pages.append(f"[PAGE {i+1}]\n" + txt)
        return "\n".join(pages)
    except Exception as e:
        print(f"[WARN] Failed to read PDF {path}: {e}")
        return ""

def text_from_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return ""

def parse_metadata_file(meta_path: Path) -> Dict[str,str]:
    """
    Parse a simple metadata file into a dict.
    Lines with "Key: Value" or "Key - Value" become {snake_key: value}.
    Other text is appended to 'summary'.
    """
    meta = {}
    if not meta_path.exists():
        return meta
    txt = text_from_file(meta_path)
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([^:-]+)\s*[:\-]\s*(.+)$", line)
        if m:
            key = m.group(1).strip().lower()
            key = re.sub(r"[^\w]+", "_", key).strip("_")
            value = m.group(2).strip()
            meta[key] = value
        else:
            # fallback: append to `summary` field
            if "summary" not in meta:
                meta["summary"] = line
            else:
                meta["summary"] += " " + line
    return meta

def chunk_text(text: str, chunk_size:int, overlap:int):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        slice_text = text[start:end]
        chunks.append({"start": start, "end": min(end, length), "text": slice_text.strip()})
        start = end - overlap
        if start < 0:
            start = 0
        if start >= length:
            break
    return chunks

def ingest(data_dir: str, out_file: str, chunk_size: int, overlap: int):
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ERROR] Data directory '{data_dir}' does not exist.")
        sys.exit(1)

    out_items = []
    for p in sorted(data_path.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ALLOWED_EXT:
            # skip irrelevant files
            continue

        # skip standalone metadata files if paired with PDF
        if p.suffix.lower() == METADATA_EXT and any(p.with_suffix(ext).exists() for ext in [".pdf"]):
            continue

        if p.suffix.lower() == ".pdf":
            text = text_from_pdf(p)
        else:
            text = text_from_file(p)

        if not text or len(text.strip()) == 0:
            print(f"[WARN] No text extracted from {p.name}. Skipping.")
            continue

        # try to find metadata file with same stem
        meta_file = p.with_suffix(METADATA_EXT)
        metadata = {}
        if meta_file.exists():
            metadata = parse_metadata_file(meta_file)
            print(f"[INFO] Loaded metadata for {p.name}: {list(metadata.keys())}")
        else:
            # also try lowercase/underscore variant
            alt = p.with_name(p.stem.lower().replace(" ", "_") + METADATA_EXT)
            if alt.exists():
                metadata = parse_metadata_file(alt)
                print(f"[INFO] Loaded metadata for {p.name} via alt file: {alt.name}")

        doc_id = str(uuid.uuid4())
        chunks = chunk_text(text, chunk_size, overlap)
        print(f"[INFO] {p.name} -> {len(chunks)} chunks")
        for idx, c in enumerate(chunks):
            item = {
                "id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "source_path": str(p),
                "chunk_index": idx,
                "start_char": c["start"],
                "end_char": c["end"],
                "chunk_text": c["text"]
            }
            # attach metadata fields (avoid overwriting core keys)
            for k, v in metadata.items():
                if k not in item:
                    item[k] = v
            out_items.append(item)

    if not out_items:
        print("[ERROR] No chunks created. Add documents to data/ and try again.")
        sys.exit(1)

    with open(out_file, "w", encoding="utf-8") as f:
        for it in out_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"[DONE] Wrote {len(out_items)} chunks to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest docs and output chunked JSONL with metadata")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out_file", default=DEFAULT_OUT_FILE)
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()
    ingest(args.data_dir, args.out_file, args.chunk_size, args.overlap)
