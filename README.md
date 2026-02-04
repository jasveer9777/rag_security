# RAG Security (Legal RAG with safety-aware retrieval)

This project implements a Retrieval-Augmented Generation (RAG) pipeline tailored for legal documents, with security-focused retrieval and prompt-safety controls. It ingests documents, builds a FAISS index, retrieves and re-ranks chunks, then calls either OpenAI or a local Ollama model to answer questions.

## Key security features

- **Prompt-injection filtering**: Detects and drops context chunks that include common instruction-injection patterns before they ever reach the model.
- **Trusted-source biasing**: Promotes PDF and trusted-domain sources while **demoting untrusted TXT** sources to reduce poisoning risk.
- **Query normalization**: Expands common legal shorthand (e.g., POCSO, CrPC) to improve retrieval accuracy and reduce ambiguity.
- **Context-only answering**: Strict prompt that instructs the model to answer **only** from provided context and say “I don’t know” otherwise.
- **Safety note in prompt**: Explicitly tells the model to ignore any instructions embedded in retrieved context.
- **Diagnostics and audit logs**: Every query writes a JSON log with retrieval scores, prompt, and answer for traceability.

These protections are implemented primarily in [src/serve_query.py](src/serve_query.py).

## Project structure

- [src/ingest_with_metadata.py](src/ingest_with_metadata.py) — Ingests documents and emits chunked JSONL.
- [src/embed_index.py](src/embed_index.py) — Builds embeddings and a FAISS index from chunks.
- [src/serve_query.py](src/serve_query.py) — Core RAG pipeline (retrieve, re-rank, build prompt, call model).
- [src/interactive_openai.py](src/interactive_openai.py) — Interactive CLI that shows chunks, prompt, and response.
- [src/test_retrieve.py](src/test_retrieve.py) — Quick retrieval sanity check (no LLM).
- [src/evaluate.py](src/evaluate.py) — RAG evaluation on a QA dataset.
- [src/comapre_models.py](src/comapre_models.py) — Compare multiple models on the same query.
- [models/openai_client.py](models/openai_client.py) — OpenAI chat wrapper (new and old SDK support).
- [models/ollama_client.py](models/ollama_client.py) — Ollama HTTP client wrapper.
- [data/](data/) — Input documents and optional metadata text files.
- [chunks.jsonl](chunks.jsonl) — Output of ingestion (one chunk per line).
- [index/](index/) — FAISS index and metadata files.
- [logs/](logs/) — Query audit logs.
- [requirements.txt](requirements.txt) — Python dependencies.

## End-to-end pipeline (from start to finish)

1. **Ingest**
   - Run [src/ingest_with_metadata.py](src/ingest_with_metadata.py).
   - Produces [chunks.jsonl](chunks.jsonl) containing text chunks plus metadata.

2. **Embed + Index**
   - Run [src/embed_index.py](src/embed_index.py).
   - Creates [index/index.faiss](index/index.faiss) and [index/metas.pkl](index/metas.pkl).

3. **Query**
   - Use [src/serve_query.py](src/serve_query.py) for single-shot queries.
   - Or [src/interactive_openai.py](src/interactive_openai.py) for interactive querying.
   - Logs are stored in [logs/](logs/).

4. **Evaluate / Compare**
   - [src/evaluate.py](src/evaluate.py) evaluates retrieval + answer quality.
   - [src/comapre_models.py](src/comapre_models.py) compares models side-by-side.

## Setup

1. Install dependencies from [requirements.txt](requirements.txt).
2. Create a .env file in the project root with your API keys if using OpenAI:
   - OPENAI_API_KEY=your_key_here
3. (Optional) If using Ollama, ensure the server is running and set:
   - OLLAMA_URL=http://localhost:11434/api/generate

## Usage

### 1) Ingest documents

Run from the project root:

python src/ingest_with_metadata.py --data_dir data --out_file chunks.jsonl --chunk_size 1200 --overlap 300

### 2) Build embeddings + FAISS index

python src/embed_index.py --chunks chunks.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --index_dir index --batch 64

### 3) Query (single shot)

python src/serve_query.py --question "When did CrPC come into effect?" --model openai:gpt-4o-mini --k 8 --min_sim 0.12 --strict

### 4) Interactive mode

python src/interactive_openai.py --model openai:gpt-4o-mini --k 5 --min_sim 0.15 --strict

### 5) Retrieval-only test

python src/test_retrieve.py --query "When did CRPC come into effect?" --k 5

### 6) Evaluate

python src/evaluate.py --qa qa_dataset.jsonl --out eval_report.csv --model ollama:ggml-mistral-7b --k 5

### 7) Compare models

python src/comapre_models.py --question "What is the POCSO Act?" --out comparisons.csv --k 5 --strict

## Detailed file and function guide

### [src/ingest_with_metadata.py](src/ingest_with_metadata.py)

**Purpose**: Load documents, parse optional metadata files, chunk the text, and save chunks to JSONL.

**Key functions**
- `text_from_pdf(path)` — Extracts PDF text with page markers.
- `text_from_file(path)` — Reads text from .txt or .md.
- `parse_metadata_file(meta_path)` — Parses key-value metadata from sidecar .txt.
- `chunk_text(text, chunk_size, overlap)` — Splits text into overlapping chunks.
- `ingest(data_dir, out_file, chunk_size, overlap)` — Main ingestion pipeline. Writes [chunks.jsonl](chunks.jsonl).

**Invocation**
- Direct CLI: python src/ingest_with_metadata.py …
- Produces input for [src/embed_index.py](src/embed_index.py).

### [src/embed_index.py](src/embed_index.py)

**Purpose**: Encode chunk text into embeddings and build a FAISS index.

**Key functions**
- `load_chunks(chunks_file)` — Reads [chunks.jsonl](chunks.jsonl) into text + metadata arrays.
- `build_embeddings(model_name, texts, batch_size)` — Creates embeddings with SentenceTransformers.
- `build_faiss_index(embeddings, index_dir)` — Normalizes vectors and builds a cosine-similarity index.
- `save_metas(metas, index_dir)` — Stores metadata in [index/metas.pkl](index/metas.pkl).
- `main(args)` — CLI entry point.

**Invocation**
- Direct CLI: python src/embed_index.py …
- Outputs used by [src/serve_query.py](src/serve_query.py) and [src/test_retrieve.py](src/test_retrieve.py).

### [src/serve_query.py](src/serve_query.py)

**Purpose**: The core RAG pipeline (retrieve + re-rank + prompt + model call + logging).

**Security and safety controls**
- Query rewrite via `rewrite_query()` using `QUERY_REWRITE_MAP`.
- Prompt-injection filtering via `is_injection_chunk()`.
- Source trust heuristics via `is_trusted_source()` and `is_pdf_source()`.
- Demotion of untrusted TXT via `DEMOTE_UNTRUSTED_TXT` and `UNTRUSTED_TXT_PENALTY`.
- Strict “context-only” answers via `build_prompt()`.
- Explicit safety note in prompt about ignoring embedded instructions.

**Key functions**
- `rewrite_query(q)` — Normalizes and expands shorthand queries.
- `load_index()` — Loads FAISS index and metadata.
- `is_injection_chunk(text)` — Detects prompt-injection patterns.
- `filename_match_score(meta, rewritten_query)` — Scores filename overlap.
- `exact_phrase_score(meta, rewritten_query)` — Scores exact phrase matches in text.
- `metadata_keyword_score(meta, rewritten_query)` — Scores keyword overlap in metadata.
- `retrieve(query, index, metas, embed_model, k, fetch_k_mult)` — Retrieves and re-ranks chunks.
- `build_context(retrieved, min_sim, max_chars)` — Builds context string from top chunks.
- `build_prompt(context, question, cutoff, strict)` — Builds the safety-aware prompt.
- `call_model(prompt, model_choice, temperature, max_tokens)` — Calls OpenAI or Ollama.
- `serve(question, model_choice, k, min_sim, cutoff, strict)` — End-to-end query handler; writes logs.

**Invocation**
- Direct CLI: python src/serve_query.py …
- Imported by [src/interactive_openai.py](src/interactive_openai.py), [src/evaluate.py](src/evaluate.py), and [src/comapre_models.py](src/comapre_models.py).

### [src/interactive_openai.py](src/interactive_openai.py)

**Purpose**: Interactive CLI that exposes the full RAG internals for inspection.

**Key functions**
- `pretty_snippet(text, max_len)` — Formats chunk snippets.
- `extract_response_text(response_obj)` — Robustly extracts LLM output text.
- `interactive_loop(model_choice, k, min_sim, cutoff, strict)` — Runs interactive query loop.

**Invocation**
- Direct CLI: python src/interactive_openai.py …
- Imports `load_index`, `retrieve`, `build_context`, `build_prompt`, and `call_model` from [src/serve_query.py](src/serve_query.py).

### [src/test_retrieve.py](src/test_retrieve.py)

**Purpose**: Quick retrieval sanity check against FAISS, without calling an LLM.

**Key functions**
- `load_index()` — Loads FAISS and metadata.
- `retrieve(index, metas, model, query, k)` — Returns top-k chunks by similarity.

**Invocation**
- Direct CLI: python src/test_retrieve.py …

### [src/evaluate.py](src/evaluate.py)

**Purpose**: Measure QA accuracy against a labeled dataset.

**Key functions**
- `exact_match(a, b)` — Exact string match.
- `f1(pred, gold)` — Token overlap F1.
- `precision_at_k(retrieved, gold_chunk_ids)` — Checks if any gold chunk retrieved.
- `evaluate(qa_path, out_csv, model, k)` — Runs evaluation and writes CSV.

**Invocation**
- Direct CLI: python src/evaluate.py …
- Uses `serve()` from [src/serve_query.py](src/serve_query.py).

### [src/comapre_models.py](src/comapre_models.py)

**Purpose**: Compare outputs across multiple models for the same query.

**Key functions**
- `compare(question, out_csv, k, cutoff, min_sim, strict)` — Runs models and writes CSV.

**Invocation**
- Direct CLI: python src/comapre_models.py …
- Uses `serve()` from [src/serve_query.py](src/serve_query.py).

### [models/openai_client.py](models/openai_client.py)

**Purpose**: OpenAI API wrapper supporting both the new and old SDK interfaces.

**Key functions**
- `openai_chat(prompt, model, temperature, max_tokens)` — Calls OpenAI and returns text, time, and usage.

**Invocation**
- Used by `call_model()` in [src/serve_query.py](src/serve_query.py).

### [models/ollama_client.py](models/ollama_client.py)

**Purpose**: Simple HTTP client for local Ollama models.

**Key functions**
- `ollama_generate(prompt, model, max_tokens)` — Calls Ollama API and returns text + raw payload.

**Invocation**
- Used by `call_model()` in [src/serve_query.py](src/serve_query.py).

## Data inputs and outputs

- Input documents in [data/](data/) can be .pdf, .txt, or .md.
- Sidecar metadata: for a file Document.pdf you can provide Document.txt containing lines like Key: Value.
- Output chunks go to [chunks.jsonl](chunks.jsonl).
- FAISS index files are stored in [index/](index/).
- Logs are stored in [logs/](logs/).

## Notes

- This project is designed to **reduce** common RAG risks (prompt injection, poisoned text, untrusted sources), but it does not guarantee complete security in adversarial settings.
- For higher assurance, consider additional controls like strict allow-lists, content signature verification, and human review of sources.
