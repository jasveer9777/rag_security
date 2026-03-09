# RAG Security — Detailed Implementation Guide

This document explains the complete end-to-end working of the project so a new reader can set it up, run it, and understand every major file and folder.

---

## 1) What this project does

This is a **security-aware legal RAG** system over Indian legal documents.

At a high level, it:
1. Ingests legal documents from `data/`.
2. Chunks and stores them in `chunks.jsonl`.
3. Builds embeddings and a FAISS index in `index/`.
4. At query time, retrieves + reranks chunks with security controls.
5. Builds a strict prompt and calls either OpenAI or Ollama.
6. Writes detailed logs in `logs/`.

Core security goals implemented:
- Prompt-injection chunk filtering.
- Query rewriting (legal shorthand expansion).
- Trusted/PDF source bias and untrusted TXT demotion.
- Context-only prompting with strict fallback (“I don't know”).
- Query-time audit logging.

---

## 2) Repository structure (what each folder/file is for)

## Root
- `README.md` — concise project readme.
- `README_DETAILED.md` — this detailed guide.
- `requirements.txt` — Python dependencies.
- `chunks.jsonl` — chunked corpus produced by ingestion.
- `Fine_rag_impl.txt` — project notes/documentation artifact.

## data/
Input corpus files (`.pdf`, `.txt`, `.md`).

- If a document is `SomeAct.pdf`, a sidecar metadata file `SomeAct.txt` can be parsed as key-value metadata.

## index/
Built retrieval artifacts:
- `index.faiss` — FAISS vector index.
- `metas.pkl` — metadata list aligned with vectors.

## logs/
Per-query audit logs written by serving scripts.

## models/
Model client wrappers:
- `openai_client.py` — OpenAI wrapper (supports new + old SDK behavior).
- `ollama_client.py` — Ollama HTTP generate wrapper.

## src/
Main pipeline and utilities:
- `ingest_with_metadata.py` — document reading + metadata parsing + chunking.
- `embed_index.py` — embedding generation + FAISS index build.
- `serve_query.py` — secure retrieval/reranking/prompt/model call/logging.
- `interactive_openai.py` — interactive CLI with detailed inspection output.
- `test_retrieve.py` — retrieval-only sanity test (no LLM call).
- `evaluate.py` — QA dataset evaluation (EM/F1/Precision@k).
- `comapre_models.py` — compare outputs across multiple models.

## papers/
Presentation and paper drafts related to this implementation.

---

## 3) Environment setup (step-by-step)

### 3.1 Python setup
Use Python 3.10+ (recommended).

### 3.2 Install dependencies
From project root:

`pip install -r requirements.txt`

### 3.3 Configure API keys / model backend
Create a `.env` file in project root.

For OpenAI:
- `OPENAI_API_KEY=your_key_here`

Optional for Ollama:
- `OLLAMA_URL=http://localhost:11434/api/generate`

If using Ollama locally, make sure Ollama server is running and the model exists.

---

## 4) End-to-end run flow

## Step 1 — Ingest documents
Script: `src/ingest_with_metadata.py`

Example:

`python src/ingest_with_metadata.py --data_dir data --out_file chunks.jsonl --chunk_size 1200 --overlap 300`

What it does:
- Reads files from `data/` with supported extensions (`.pdf`, `.txt`, `.md`).
- For PDFs, extracts page text with `[PAGE n]` markers.
- Parses sidecar metadata text files (`Key: Value` or `Key - Value`).
- Splits content into overlapping chunks.
- Writes one JSON object per chunk to `chunks.jsonl`.

Output created:
- `chunks.jsonl`

---

## Step 2 — Build embeddings and index
Script: `src/embed_index.py`

Example:

`python src/embed_index.py --chunks chunks.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --index_dir index --batch 64`

What it does:
- Loads chunks and metadata from `chunks.jsonl`.
- Generates embeddings using SentenceTransformers.
- Normalizes embeddings (`faiss.normalize_L2`).
- Builds FAISS `IndexFlatIP` (cosine-like similarity search).
- Saves:
  - `index/index.faiss`
  - `index/metas.pkl`

Output created:
- `index/index.faiss`
- `index/metas.pkl`

---

## Step 3 — Query (single-shot)
Script: `src/serve_query.py`

Example:

`python src/serve_query.py --question "When did CrPC come into effect?" --model openai:gpt-4o-mini --k 8 --min_sim 0.12 --strict`

What it does (internally):
1. Loads FAISS index + metadata.
2. Rewrites query using legal normalization map.
3. Encodes query and retrieves over-fetched candidates.
4. Filters likely injection chunks.
5. Computes combined rerank score using:
   - embedding similarity
   - filename overlap
   - exact phrase signal
   - metadata keyword signal
   - PDF/trusted-source boost
   - untrusted TXT penalty
6. Builds context with thresholds (`min_sim`, max chars).
7. Builds strict safety prompt.
8. Calls model backend.
9. Logs full diagnostics to `logs/query_<timestamp>.json`.

---

## Step 4 — Interactive mode (debug-friendly)
Script: `src/interactive_openai.py`

Example:

`python src/interactive_openai.py --model openai:gpt-4o-mini --k 5 --min_sim 0.15 --strict`

What it shows per question:
- Original user query.
- Top retrieved chunks with score/source/snippet.
- Full prompt sent to model.
- Extracted model response text.
- Timing info.

It also writes logs to `logs/interactive_query_<timestamp>.json`.

---

## Step 5 — Retrieval sanity test (no generation)
Script: `src/test_retrieve.py`

Example:

`python src/test_retrieve.py --query "When did CRPC come into effect?" --k 5`

Use this to verify retrieval quality independently from LLM behavior.

---

## Step 6 — Evaluation on QA dataset
Script: `src/evaluate.py`

Example:

`python src/evaluate.py --qa qa_dataset.jsonl --out eval_report.csv --model ollama:gpt-oss:20b --k 5`

Expected QA dataset format (`jsonl`, one item per line):
- `id`
- `question`
- `gold_answer`
- `gold_chunk_ids` (list)

Metrics produced:
- `precision_k` (hit in retrieved gold chunk IDs)
- `em` (exact match)
- `f1` (token overlap F1)

Output:
- `eval_report.csv`

---

## Step 7 — Compare models
Script: `src/comapre_models.py` (filename intentionally as present in repo)

Example:

`python src/comapre_models.py --question "What is the POCSO Act?" --out comparisons.csv --k 5 --strict`

What it does:
- Runs same question across configured models.
- Saves model-wise answers and retrieved chunk references.

Output:
- `comparisons.csv`

---

## 5) Security implementation details (actual logic)

Implemented in `src/serve_query.py`.

## 5.1 Query rewrite
`rewrite_query()` applies regex-based legal shorthand expansion using `QUERY_REWRITE_MAP`.

Examples:
- `crpc` → `code of criminal procedure 1973`
- `pocso` → `protection of children from sexual offences act 2012`
- `bns`, `bnss`, `bsa`, `ipc`, `iea`, etc.

## 5.2 Injection chunk filtering
`is_injection_chunk(text)` checks retrieved text against configured suspicious patterns and drops matched chunks before prompt construction.

## 5.3 Trust-aware reranking
Combined score includes:
- base embedding score
- filename score
- exact phrase score
- metadata keyword score
- PDF/trusted boost
- untrusted TXT penalty

This helps reduce poisoning influence from untrusted text sources.

## 5.4 Context and strict prompting
- `build_context()` applies similarity and length constraints.
- `build_prompt()` in strict mode enforces context-only answers and fallback: “I don't know”.
- Prompt includes explicit safety note to ignore instructions embedded in context chunks.

## 5.5 Audit logs
`serve()` writes a detailed JSON log with:
- question + rewritten question
- retrieval score components
- source/chunk references
- prompt text
- answer
- latency

---

## 6) File-by-file technical map

## `src/ingest_with_metadata.py`
Key functions:
- `text_from_pdf(path)`
- `text_from_file(path)`
- `parse_metadata_file(meta_path)`
- `chunk_text(text, chunk_size, overlap)`
- `ingest(data_dir, out_file, chunk_size, overlap)`

## `src/embed_index.py`
Key functions:
- `load_chunks(chunks_file)`
- `build_embeddings(model_name, texts, batch_size)`
- `build_faiss_index(embeddings, index_dir)`
- `save_metas(metas, index_dir)`
- `main(args)`

## `src/serve_query.py`
Key functions:
- `rewrite_query(q)`
- `load_index()`
- `is_injection_chunk(text)`
- `filename_match_score(meta, rewritten_query)`
- `exact_phrase_score(meta, rewritten_query)`
- `metadata_keyword_score(meta, rewritten_query)`
- `retrieve(query, index, metas, embed_model, k, fetch_k_mult)`
- `build_context(retrieved, min_sim, max_chars)`
- `build_prompt(context, question, cutoff, strict)`
- `call_model(prompt, model_choice, temperature, max_tokens)`
- `serve(question, model_choice, k, min_sim, cutoff, strict)`

## `src/interactive_openai.py`
Key functions:
- `pretty_snippet(text, max_len)`
- `extract_response_text(response_obj)`
- `interactive_loop(model_choice, k, min_sim, cutoff, strict)`

## `src/test_retrieve.py`
Key functions:
- `load_index()`
- `retrieve(index, metas, model, query, k)`

## `src/evaluate.py`
Key functions:
- `exact_match(a, b)`
- `f1(pred, gold)`
- `precision_at_k(retrieved, gold_chunk_ids)`
- `evaluate(qa_path, out_csv, model, k)`

## `src/comapre_models.py`
Key functions:
- `compare(question, out_csv, k, cutoff, min_sim, strict)`

## `models/openai_client.py`
Key function:
- `openai_chat(prompt, model, temperature, max_tokens)`

## `models/ollama_client.py`
Key function:
- `ollama_generate(prompt, model, max_tokens, temperature)`

---

## 7) Typical command sequence for a new user

1. `pip install -r requirements.txt`
2. Create `.env` (OpenAI key and/or Ollama URL)
3. `python src/ingest_with_metadata.py --data_dir data --out_file chunks.jsonl --chunk_size 1200 --overlap 300`
4. `python src/embed_index.py --chunks chunks.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --index_dir index --batch 64`
5. `python src/test_retrieve.py --query "What is Section 302 IPC?" --k 5`
6. `python src/serve_query.py --question "What is Section 302 IPC?" --model openai:gpt-4o-mini --k 8 --min_sim 0.12 --strict`
7. `python src/interactive_openai.py --model openai:gpt-4o-mini --k 5 --min_sim 0.15 --strict`

Optional:
- `python src/evaluate.py --qa qa_dataset.jsonl --out eval_report.csv --model ollama:gpt-oss:20b --k 5`
- `python src/comapre_models.py --question "What is the POCSO Act?" --out comparisons.csv --k 5 --strict`

---

## 8) Troubleshooting quick guide

- **"Index not found"**
  - Run ingestion + embedding/index steps first.

- **OpenAI authentication error**
  - Ensure `OPENAI_API_KEY` is set in `.env`.

- **Ollama request failed**
  - Ensure Ollama server is running and model is available.
  - Verify `OLLAMA_URL`.

- **Very weak answers**
  - Check retrieval first using `src/test_retrieve.py`.
  - Tune `--k`, `--min_sim`, and chunking parameters.

- **Interactive script exits with import/path issues**
  - Run commands from repository root.

---

## 9) Current limitations and future improvements

Current limitations:
- Injection detection is regex-based and may miss obfuscated attacks.
- Trust heuristics are policy-based, not cryptographically verified.
- No formal adversarial benchmark harness in repo by default.

Natural next steps:
- Add provenance/signature checks at ingest.
- Add learned anomaly/injection detection.
- Add benchmark suite for poisoning + prompt injection + backdoor scenarios.

---

## 10) One-paragraph mental model

Think of this project as a **secure retrieval wrapper around LLM generation**: retrieval is enriched with query rewrite + trust-aware reranking + injection filtering so only safer, relevant legal chunks become context; prompt policy then constrains generation to context-only behavior; finally detailed logs make every answer auditable.
