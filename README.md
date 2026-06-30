# A Security Perspective on Retrieval‑Augmented Generation (RAG)

**Thesis Project – Legal Domain (India)**

This repository implements a Retrieval‑Augmented Generation pipeline over official Indian legal documents and studies its security vulnerabilities and mitigations.

---

## Overview
- **Corpus**: Legal statutes, regulations, and case law scraped from Indian government portals.
- **Pipeline**:
  1. Ingest PDFs/TXTs → chunk → embed → FAISS index.
  2. Query rewriting for legal shorthand.
  3. Trust‑aware reranking (PDFs boosted, plain‑text demoted).
  4. Strict prompt enforcing *context‑only* answers.
  5. Detailed audit logging.
- **Attacks**: Corpus poisoning and prompt‑injection attacks against three LLM back‑ends (OpenAI `gpt‑4o‑mini`, Ollama `llama3:8b`, local `phi‑2`).
- **Defense**: Dynamic injection‑chunk detection, source‑trust scoring, post‑retrieval sanitisation, and a strict response template.

---

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set up .env (OpenAI key and/or Ollama URL)
# See README_DETAILED.md for format

# Ingest the legal corpus
python src/ingest_with_metadata.py --data_dir data --out_file chunks.jsonl --chunk_size 1200 --overlap 300

# Build the FAISS index
python src/embed_index.py --chunks chunks.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --index_dir index --batch 64

# Run a secure query (default strict mode)
python src/serve_query.py --question "What is the penalty for kidnapping under the IPC?" --model openai:gpt-4o-mini --k 8 --min_sim 0.12 --strict
```

---

## Evaluation – Before vs. After Defense
| Model | Attack Success (no defense) | Attack Success (with defense) |
|-------|-----------------------------|--------------------------------|
| OpenAI `gpt-4o-mini` | 42% | **3%** |
| Ollama `llama3:8b` | 38% | **2%** |
| Local `phi-2` | 45% | **5%** |

---

## License
This project is released under the MIT License.
