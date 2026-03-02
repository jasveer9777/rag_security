# Secure-by-Design Legal RAG: Hardening Retrieval-Augmented Generation Against Prompt Injection and Knowledge Poisoning

## Abstract
Retrieval-Augmented Generation (RAG) improves factual grounding by coupling language models with external knowledge, but it also expands the attack surface of deployed systems. Recent work shows that RAG pipelines are vulnerable to indirect prompt injection, corpus poisoning, and retriever-level backdoors. In this paper, we present an implementation-focused security hardening of a legal-domain RAG system built over Indian statutes, using *Backdoored Retrievers for Prompt Injection Attacks on Retrieval Augmented Generation of Large Language Models* as the primary reference. We implement a defense-in-depth pipeline that combines query normalization, injection-aware chunk filtering, trusted-source weighting, untrusted text demotion, strict context-only prompting, and structured query-time audit logging. Unlike purely conceptual defenses, our work contributes a complete engineering blueprint with reproducible modules for ingestion, indexing, retrieval, reranking, model dispatch, and evaluation. We position our implementation against known attack classes from Backdoored and PoisonedRAG, and discuss how practical controls reduce real-world risk while preserving usability in legal QA settings.

## 1. Introduction
RAG systems are increasingly used in high-stakes domains such as law, healthcare, and finance. In legal QA, errors can produce misinformation, non-compliance, or unsafe guidance. While RAG reduces hallucination by grounding answers in retrieved documents, prior work demonstrates that retrieval itself becomes a security-critical component.

The base paper, *Backdoored Retrievers for Prompt Injection Attacks on RAG* [1], demonstrates that attackers can manipulate RAG outputs via corpus poisoning and retriever backdoors, with high attack success rates (ASR), including values up to approximately 0.91 in some settings. *PoisonedRAG* [2] further shows that injecting only a small number of malicious texts can drive targeted outputs, reporting up to 90% ASR (and up to 97% in some configurations). A broader survey [3] frames these as full-chain RAG risks spanning adversarial retrieval, poisoning, privacy leakage, and backdoor behavior.

Motivated by these findings, we designed and implemented a legal RAG stack with practical hardening controls. Our core objective is not to claim absolute security, but to reduce exploitability under realistic deployment constraints.

### Contributions
1. **Implementation contribution**: We provide a complete secure-leaning legal RAG pipeline with modular ingestion, indexing, retrieval, reranking, model routing, and logging.
2. **Defense-in-depth retrieval design**: We integrate multi-signal reranking with source trust priors and prompt-injection filtering before context reaches the model.
3. **Prompt-layer containment**: We enforce context-only answering and explicit refusal behavior ("I don’t know") when evidence is absent.
4. **Operational transparency**: We log retrieval diagnostics and full prompt/response traces for auditability and post-incident analysis.
5. **Research alignment**: We map our controls directly to attack modes identified in [1], [2], and [3], and discuss where gaps remain.

## 2. Background and Threat Model
### 2.1 Threats from prior literature
- **Indirect prompt injection**: Malicious instructions embedded in retrieved text override model behavior [1].
- **Knowledge-base poisoning**: Adversarial documents inserted in corpus influence retrieval and generation [1], [2].
- **Retriever-targeted manipulation**: Backdoored or poisoned retrievers elevate attacker-controlled documents [1], [2].
- **Full-chain risks**: Surveyed attack surfaces include retrieval perturbation, generation control, privacy leakage, and adaptive triggers [3].

### 2.2 Target system assumptions
Our legal RAG assumes:
- A mixed-quality corpus where trusted and untrusted documents may co-exist.
- Black-box access to the QA interface by benign and potentially malicious users.
- No guarantee that all ingested content is clean.

### 2.3 Security goals
- Reduce probability that malicious chunks enter final model context.
- Reduce over-reliance on untrusted sources.
- Force grounded/refusal behavior when context evidence is weak.
- Preserve retrieval relevance for legitimate legal queries.

## 3. System Architecture
The implemented pipeline consists of seven modules:

1. **Document ingestion with metadata** (`src/ingest_with_metadata.py`)
2. **Embedding and FAISS indexing** (`src/embed_index.py`)
3. **Security-aware retrieval and reranking** (`src/serve_query.py`)
4. **Prompt construction with safety notes** (`src/serve_query.py`)
5. **Model dispatch (OpenAI/Ollama)** (`models/openai_client.py`, `models/ollama_client.py`)
6. **Interactive QA interface** (`src/interactive_openai.py`)
7. **Evaluation and model comparison** (`src/evaluate.py`, `src/comapre_models.py`)

The corpus includes legal documents (e.g., IPC, CrPC, BNS, BNSS, BSA, POCSO, NDPS, IEA), chunked and embedded into FAISS for similarity retrieval.

## 4. Security Hardening Design
### 4.1 Query normalization for legal shorthand
The retriever rewrites shorthand and common misspellings (e.g., `POCSO`, `CrPC`, `IPC`, `BNS`) into expanded canonical forms. This reduces lexical ambiguity and helps relevant legal chunks outrank noisy matches.

### 4.2 Prompt-injection chunk filtering
Before reranking, each candidate chunk is scanned for suspicious instruction-like patterns (e.g., "ignore previous instructions", "answer exactly"). Detected chunks are dropped from the candidate set.

### 4.3 Trust-aware reranking and source priors
A combined score is used for ranking:

$$
S_{combined} = S_{embed} + \alpha S_{filename} + \beta S_{phrase} + \gamma S_{meta} + \delta S_{trusted} - \lambda S_{untrusted\_txt}
$$

Where:
- $S_{embed}$ is cosine-like similarity from FAISS,
- $S_{filename}$ captures overlap between query tokens and source filename,
- $S_{phrase}$ rewards direct phrase presence in chunk text,
- $S_{meta}$ rewards metadata keyword alignment,
- $S_{trusted}$ boosts trusted/PDF sources,
- $S_{untrusted\_txt}$ penalizes untrusted plain-text sources.

This directly targets poisoning patterns where attacker-controlled text files are injected into corpora.

### 4.4 Context-only answer policy
The prompt builder supports strict mode that forces:
- "Answer only from provided context"
- "If answer absent, respond exactly: I don’t know"

This limits generation freedom under sparse or adversarial retrieval.

### 4.5 Embedded-instruction nullification
The system prompt explicitly warns the model to ignore instructions found inside context chunks and use only factual legal content.

### 4.6 Audit logging for forensic analysis
Each query logs:
- rewritten query,
- retrieval scores and component diagnostics,
- selected chunks,
- generated prompt,
- final answer,
- latency.

These logs support reproducibility, debugging, and incident investigation.

## 5. Implementation Details
### 5.1 Ingestion and indexing
- PDF/text ingestion with optional sidecar metadata parsing.
- Overlapping chunking to preserve continuity.
- SentenceTransformer embeddings (`all-MiniLM-L6-v2`).
- FAISS `IndexFlatIP` over normalized vectors.

### 5.2 Retrieval-time defenses (core module)
In `serve_query.py`, retrieval performs:
1. query rewrite,
2. over-fetch from FAISS,
3. injection filtering,
4. multi-signal reranking,
5. trust/penalty adjustment,
6. top-$k$ selection.

### 5.3 Multi-model inference support
The system supports both hosted and local models through a unified `model_choice` format (`openai:*` or `ollama:*`). Security controls remain model-agnostic because they act before generation.

## 6. Empirical Observations from the Implemented System
We report implementation-level observations from existing project logs and retrieval traces.

### 6.1 Qualitative retrieval behavior
For legal statute questions (e.g., attempt to murder, affray/public order), retrieved top chunks frequently include relevant sections from IPC/BNS documents, indicating useful domain grounding. The logs show ranked chunk outputs with diagnostic scores that can be inspected per query.

### 6.2 Prompt-safety behavior
Generated prompts include explicit anti-injection instructions and context-only constraints, reducing susceptibility to direct instruction override from retrieved text.

### 6.3 Operational traceability
For each interactive query, logs store complete request/response records and retrieval evidence. This creates a measurable audit trail often missing in baseline RAG demos.

### 6.4 Comparison with attack-centric papers
Unlike [1] and [2], which primarily demonstrate attacks, our implementation emphasizes practical mitigations deployable in day-to-day legal RAG systems. We do not claim to outperform their attack ASR under controlled red-team benchmarks; instead, we provide concrete controls aligned with their threat findings.

## 7. Discussion: Relation to Prior Work
### 7.1 Alignment with Backdoored [1]
Backdoored shows that small poisoning changes and retriever compromise can dramatically affect generation outcomes. Our trust-aware reranking and untrusted text demotion are direct mitigations against this pathway. Injection chunk filtering and prompt-level safeguards further reduce success of malicious instructions that survive retrieval.

### 7.2 Alignment with PoisonedRAG [2]
PoisonedRAG demonstrates high ASR with only a few injected malicious texts. Our approach counters this by reducing attacker text prominence through source priors and lexical/metadata reranking, and by requiring context-grounded answers. However, adaptive attacks that mimic trusted formatting remain a concern.

### 7.3 Alignment with the RAG security survey [3]
The survey advocates full-chain defenses. Our system contributes retrieval-layer and prompt-layer controls with observability, but future extensions should include privacy-preserving retrieval, dynamic anomaly detection, and trigger-level adversarial testing.

## 8. Limitations and Future Work
1. **No formal red-team benchmark yet**: We currently provide engineering hardening and qualitative evidence; quantitative ASR reduction vs. attack baselines remains future work.
2. **Heuristic injection detection**: Regex-based filtering can be bypassed by obfuscation.
3. **Trust priors are policy-dependent**: Incorrect trust lists can under- or over-penalize sources.
4. **Retriever model robustness**: We do not yet train robust retrievers or apply certified defenses.

Planned improvements:
- automated adversarial benchmark harness (poisoning + backdoor simulations),
- learned injection detectors,
- provenance verification and document signing,
- uncertainty-aware refusal calibration,
- adaptive defenses across multilingual legal corpora.

## 9. Conclusion
This paper presents a practical, secure-by-design implementation of legal RAG inspired by contemporary RAG security research. Using Backdoored [1] as the base reference, and incorporating insights from PoisonedRAG [2] and a full-chain security survey [3], we implemented layered defenses spanning retrieval, prompting, and operations. The result is a deployable architecture that raises the cost of prompt-injection and poisoning attacks while maintaining legal-domain utility. Our central claim is implementation realism: security gains in RAG require not only new attack papers, but also robust engineering defaults, transparent diagnostics, and defense-in-depth at every query step.

## References
[1] C. Clop and Y. Teglia, “Backdoored Retrievers for Prompt Injection Attacks on Retrieval Augmented Generation of Large Language Models,” 2025.

[2] W. Zou, R. Geng, B. Wang, and J. Jia, “PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models,” in *Proceedings of the 34th USENIX Security Symposium*, 2025.

[3] “Retrieval-Augmented Generation: A Survey of Security Threats and Defenses,” 2025.
