"""
Interactive OpenAI RAG with detailed output format.

Shows: user query → top 5 chunks → prompt sent to LLM → LLM response
"""

import sys
import pathlib
import argparse
import json
import time
from pathlib import Path

# Add project root to Python path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env automatically
from dotenv import load_dotenv
load_dotenv(Path(ROOT) / ".env")

from sentence_transformers import SentenceTransformer

# Import from the RAG pipeline (ensure these exist)
from src.serve_query import (
    load_index, retrieve, build_context,
    build_prompt, call_model, EMBED_MODEL
)

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

def pretty_snippet(text, max_len=300):
    s = text.replace("\n", " ").strip()
    if len(s) > max_len:
        return s[:max_len].rsplit(" ", 1)[0] + " ..."
    return s

def extract_response_text(response_obj):
    """
    Robustly extract assistant text from various possible response shapes:
    - dict with "text"
    - new OpenAI v1 response object (has .choices with .message["content"])
    - older OpenAI ChatCompletion (choices[0].message.content or choices[0].text)
    - ollama-style dicts (may have 'text' or 'response')
    - plain string
    """
    # 1) dict with explicit 'text'
    if isinstance(response_obj, dict):
        # common keys
        for key in ("text", "response", "output", "content"):
            if key in response_obj and isinstance(response_obj[key], str):
                return response_obj[key].strip()
        # sometimes nested like {'choices': [{'message': {'content': '...'}}]}
        if "choices" in response_obj and isinstance(response_obj["choices"], (list,tuple)) and len(response_obj["choices"])>0:
            ch = response_obj["choices"][0]
            if isinstance(ch, dict):
                # try several nested shapes
                if "message" in ch and isinstance(ch["message"], dict) and "content" in ch["message"]:
                    return ch["message"]["content"].strip()
                if "text" in ch and isinstance(ch["text"], str):
                    return ch["text"].strip()
    # 2) OpenAI new v1 client object or similar (has .choices)
    try:
        # Many OpenAI responses allow attribute access
        if hasattr(response_obj, "choices") and getattr(response_obj, "choices"):
            choice0 = response_obj.choices[0]
            # new: choice0.message["content"]
            try:
                msg = getattr(choice0, "message", None)
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"].strip()
                # sometimes message is an object with content attr
                if hasattr(msg, "get") and msg.get("content"):
                    return msg.get("content").strip()
                if hasattr(choice0, "message") and hasattr(choice0.message, "get") and choice0.message.get("content"):
                    return choice0.message.get("content").strip()
            except Exception:
                pass
            # older: choice0.message.content or choice0.text
            if hasattr(choice0, "message") and hasattr(choice0.message, "content"):
                return choice0.message.content.strip()
            if hasattr(choice0, "text"):
                return choice0.text.strip()
    except Exception:
        pass

    # 3) fallback: if it's a string
    if isinstance(response_obj, str):
        return response_obj.strip()

    # 4) last resort: stringify
    try:
        return str(response_obj).strip()
    except Exception:
        return ""

def interactive_loop(model_choice, k, min_sim, cutoff, strict):
    # Load once
    index, metas = load_index()
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("\n🚀 Interactive OpenAI RAG is ready.")
    print("Type your question. Type 'exit' to quit.\n")

    while True:
        try:
            q = input("QUESTION> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if q.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        if not q:
            continue

        t0 = time.time()

        # 1. Retrieve chunks
        retrieved = retrieve(q, index, metas, embed_model, k=k)

        # 2. Build context from filtered chunks
        context = build_context(retrieved, min_sim=min_sim)

        # 3. Build final prompt
        prompt = build_prompt(context, q, cutoff=cutoff, strict=strict)

        # ====================================================================
        # DISPLAY FORMAT AS REQUESTED
        # ====================================================================
        
        print("\n" + "="*80)
        print("USER QUERY")
        print("="*80)
        print(q)
        
        print("\n" + "="*80)
        print("TOP 5 RETRIEVED CHUNKS")
        print("="*80)
        for i, r in enumerate(retrieved[:5], 1):
            meta = r["meta"]
            print(f"\n[Chunk {i}]")
            print(f"  Score: {r['score']:.4f}")
            print(f"  Source: {meta.get('source_path', 'N/A')}")
            print(f"  Chunk Index: {meta.get('chunk_index', 'N/A')}")
            print(f"  Chunk ID: {meta.get('id', 'N/A')}")
            print(f"  Text Snippet: {pretty_snippet(meta.get('chunk_text', ''), 200)}")
        
        print("\n" + "="*80)
        print("PROMPT SENT TO LLM")
        print("="*80)
        print(prompt)

        # 4. Call model
        import io
        import contextlib
        
        try:
            # Suppress any prints from call_model function
            with contextlib.redirect_stdout(io.StringIO()):
                response_obj, latency = call_model(prompt, model_choice)
        except Exception as e:
            print(f"\n[ERROR] LLM call failed: {e}\n")
            continue

        # 5. Extract content-only text
        answer_text = extract_response_text(response_obj)

        t1 = time.time()

        print("\n" + "="*80)
        print("LLM RESPONSE (CONTENT ONLY)")
        print("="*80)
        print(answer_text)
        print("\n" + "="*80)
        print(f"[Total time: {t1-t0:.2f}s | Model latency: {latency:.2f}s]")
        print("="*80 + "\n")

        # Save audit log (silent)
        ts = int(time.time())
        log_path = LOG_DIR / f"interactive_query_{ts}.json"

        safe_retrieved = [
            {
                "score": r["score"],
                "source_path": r["meta"].get("source_path"),
                "chunk_index": r["meta"].get("chunk_index"),
                "chunk_id": r["meta"].get("id"),
                "snippet": pretty_snippet(r["meta"].get("chunk_text",""), 400)
            } for r in retrieved
        ]
        log_data = {
            "timestamp": ts,
            "query": q,
            "model": model_choice,
            "k": k,
            "min_sim": min_sim,
            "cutoff": cutoff,
            "strict": strict,
            "retrieved": safe_retrieved,
            "prompt": prompt,
            "answer": answer_text,
            "total_time": round(t1 - t0, 3),
            "model_latency": round(latency if isinstance(latency, (int,float)) else 0.0, 3)
        }
        log_path.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="anthropic:claude-sonnet-4-5-saarathi02")  # NEW: Azure Claude
    # parser.add_argument("--model", default="openai:gpt-4o-mini")  # OLD: OpenAI (commented)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min_sim", type=float, default=0.15)
    parser.add_argument("--cutoff", type=str, default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    interactive_loop(args.model, args.k, args.min_sim, args.cutoff, args.strict)