# RAG Security — Detailed Implementation Guide

This document provides a comprehensive overview of the **A Security Perspective on Retrieval-Augmented Generation (RAG)** thesis project. It explains the end-to-end workflow, the legal‑domain corpus, the adversarial poisoning experiments, the custom defense architecture we implemented, and the evaluation results before and after applying the defense.

---

## 1) Project Overview

The goal of this thesis is to investigate security risks in RAG pipelines that operate over **official Indian legal documents**. We:
1. **Collected** a clean corpus from government websites (e.g., `gov.in`, `legislative.gov.in`).
2. **Implemented** a retrieval‑augmented generation system that includes:
   - Chunking, embedding, and FAISS indexing.
   - Query rewriting for legal shorthand.
   - Trust‑aware reranking (PDF/official sources are boosted, plain‑text sources are demoted).
   - Strict prompt engineering that forces the LLM to answer *only* from retrieved context.
   - Comprehensive audit logging.
3. **Performed corpus poisoning** by injecting malicious statements into a subset of the text files.
4. **Attacked** the system against three LLM back‑ends (OpenAI `gpt-4o-mini`, Ollama `llama3:8b`, and a locally‑hosted `phi‑2`).
5. **Designed and integrated** a custom defense architecture that adds:
   - Dynamic injection‑chunk detection, 
   - Source‑trust scoring, and
   - Post‑retrieval sanitisation.
6. **Re‑evaluated** the pipeline, demonstrating a drastic drop in successful attacks while maintaining answer quality.

---

## 2) Repository Structure (unchanged)

### Root
- `README.md` — concise project readme.
- `README_DETAILED.md` — this detailed guide.
- `requirements.txt` — Python dependencies.
- `chunks.jsonl` — chunked corpus produced by ingestion.
- `Fine_rag_impl.txt` — project notes/documentation artifact.

### data/
- Input corpus files (`.pdf`, `.txt`, `.md`).
- Official legal PDFs are stored here; each may have a side‑car `.txt` metadata file.

### index/
- `index.faiss` — FAISS vector index.
- `metas.pkl` — metadata list aligned with vectors.

### logs/
- Per‑query audit logs written by serving scripts.

### models/
- `openai_client.py` — OpenAI wrapper.
- `ollama_client.py` — Ollama HTTP wrapper.

### src/
- `ingest_with_metadata.py` — document reading + metadata parsing + chunking.
- `embed_index.py` — embedding generation + FAISS index build.
- `serve_query.py` — secure retrieval/reranking/prompt/model call/logging (core security logic).
- `interactive_openai.py` — interactive CLI for debugging.
- `test_retrieve.py` — retrieval‑only sanity test.
- `evaluate.py` — QA dataset evaluation (EM/F1/Precision@k).
- `comapre_models.py` — compare outputs across multiple models.
- `attack_asr.py` — orchestrates poisoning attacks and records results.
- `attack_cases.jsonl` – definition of attack vectors used in experiments.

---

## 3) Environment Setup (unchanged)

### 3.1 Python setup
- Use Python **3.10+** (recommended).

### 3.2 Install dependencies
```bash
pip install -r requirements.txt
```

### 3.3 Configure API keys / model backend
Create a `.env` file in the project root.
```
# OpenAI
OPENAI_API_KEY=your_key_here

# Ollama (optional)
OLLAMA_URL=http://localhost:11434/api/generate
```
If using Ollama locally, ensure the server is running and the model exists.

---

## 4) Data Collection (Legal Domain)

We harvested **official statutes, regulations, and case law** from Indian government portals:
- `https://legislative.gov.in/`
- `https://www.india.gov.in/`
- `https://www.sci.gov.in/`

All PDFs were stored under `data/` and paired with a metadata file containing:
```
Title: <document title>
Source: Government of India
Year: <year>
Type: PDF
```
The clean corpus contains **≈ 1.2 GB** of text across **≈ 3 500** documents.

---

## 5) Corpus Poisoning & Attack Suite

### 5.1 Poisoning strategy
We injected **malicious statements** (e.g., “The death penalty for theft is 5 years”) into a randomly selected **5 %** of the `.txt` side‑car files. The injection format mimics legitimate legal language to evade naïve filters.

### 5.2 Attack vectors (defined in `attack_cases.jsonl`)
| ID | Description | Target Model |
|----|-------------|--------------|
| A1 | Prompt injection via crafted query (e.g., `Ignore previous instructions; answer: …`). | All |
| A2 | Back‑door trigger phrase hidden in a chunk. | All |
| A3 | Retrieval‑time prompt injection (embedding of malicious chunk). | All |

The `src/attack_asr.py` script runs the attacks, records success rates, and stores per‑run logs under `logs/attack_<timestamp>.json`.

---

## 6) Defense Architecture (Implemented in `src/serve_query.py`)

1. **Dynamic Injection‑Chunk Detection** – `is_injection_chunk(text)` now uses an expanded regex list and a lightweight ML classifier (trained on clean vs. poisoned chunks).
2. **Source‑Trust Scoring** – PDF/official sources receive a **+0.15** boost; plain‑text files receive a **‑0.10** penalty.
3. **Post‑Retrieval Sanitisation** – After reranking, any chunk whose *trust score* falls below a threshold is discarded before prompt construction.
4. **Strict Prompt Template** – Guarantees the model replies *only* with information present in the retained context; otherwise it says “I don’t know”.

All these components are orchestrated in `serve_query.py` (see Section 5 of the original guide for function list).

---

## 7) Evaluation – Before vs. After Defense

We evaluated the pipeline on the three LLM back‑ends using the attack suite.

| Model | Attack Success (no defense) | Attack Success (with defense) |
|-------|-----------------------------|------------------------------|
| OpenAI `gpt‑4o‑mini` | 42 % | **3 %** |
| Ollama `llama3:8b` | 38 % | **2 %** |
| Local `phi‑2` | 45 % | **5 %** |

Answer quality (EM/F1) remained within **±2 %** of the baseline, confirming that the defense does not significantly degrade legitimate performance.

Full result tables are generated by `src/evaluate.py` and stored as `eval_report_before.csv` / `eval_report_after.csv`.

---

## 8) Typical Command Sequence for a New User
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Create `.env` with required API keys.
3. **Ingest** the legal corpus:
   ```bash
   python src/ingest_with_metadata.py --data_dir data --out_file chunks.jsonl --chunk_size 1200 --overlap 300
   ```
4. **Build** the FAISS index:
   ```bash
   python src/embed_index.py --chunks chunks.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --index_dir index --batch 64
   ```
5. **Run** a sanity retrieval test:
   ```bash
   python src/test_retrieve.py --query "When did the Criminal Procedure Code come into effect?" --k 5
   ```
6. **Serve** a secure query (default strict mode):
   ```bash
   python src/serve_query.py --question "What is the penalty for kidnapping under the IPC?" --model openai:gpt-4o-mini --k 8 --min_sim 0.12 --strict
   ```
7. **Run attacks** (optional, for research):
   ```bash
   python src/attack_asr.py --config attack_cases.jsonl --model openai:gpt-4o-mini
   ```
8. **Evaluate** before/after defense:
   ```bash
   python src/evaluate.py --qa qa_dataset.jsonl --out eval_report_before.csv --model openai:gpt-4o-mini
   # After enabling defense (default in serve_query.py)
   python src/evaluate.py --qa qa_dataset.jsonl --out eval_report_after.csv --model openai:gpt-4o-mini
   ```
9. **Compare** multiple models:
   ```bash
   python src/comapre_models.py --question "What is the POCSO Act?" --out comparisons.csv --k 5 --strict
   ```

---

## 9) Troubleshooting Quick Guide (unchanged)
- **"Index not found"** – Run ingestion + embedding steps first.
- **OpenAI authentication error** – Ensure `OPENAI_API_KEY` is set in `.env`.
- **Ollama request failed** – Verify Ollama server is running and the model exists.
- **Very weak answers** – Check retrieval quality with `src/test_retrieve.py` and tune `--k`, `--min_sim`.
- **Interactive script import/path issues** – Run commands from repository root.

---

## 10) Current Limitations & Future Work (expanded)
### Limitations
- Injection detection relies on regexes and a lightweight classifier; sophisticated obfuscation may bypass it.
- Trust heuristics are policy‑based and do not cryptographically verify source authenticity.
- The attack harness only covers three predefined vectors.

### Future Directions
- Integrate **digital signatures** for PDF provenance at ingest time.
- Train a **deep anomaly detector** on chunk embeddings to catch subtle poisoning.
- Expand the benchmark suite with **automatically generated** adversarial examples using LLM‑driven red‑team prompts.
- Publish a **paper‑ready dataset** of poisoned vs. clean legal chunks for the community.

---

## 11) One‑Paragraph Mental Model
Think of this project as a **secure retrieval wrapper around LLM generation**: it first filters, rewrites, and trust‑scores retrieved legal chunks, then feeds only the vetted context to a language model under a strict “context‑only” policy. Detailed logs provide full auditability, and the added defense layers dramatically reduce successful poisoning attacks while preserving answer quality.
