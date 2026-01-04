# models/ollama_client.py
import os
import time
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_TIMEOUT = 60

def ollama_generate(prompt, model="ggml-mistral-7b", max_tokens=512):
    """
    Call local Ollama generate endpoint. Adjust if your Ollama API differs.
    Returns {'text':..., 'time':..., 'raw':...}
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens
    }
    start = time.time()
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        j = r.json()
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")
    elapsed = time.time() - start
    # typical Ollama responses may vary; try common keys
    text = j.get("text") or j.get("response") or j.get("output") or str(j)
    return {"text": text, "time": elapsed, "raw": j}
