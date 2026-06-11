import os
import time
import requests
import json

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_TIMEOUT = 120

def ollama_generate(prompt, model="gpt-oss:20b", max_tokens=512, temperature=0.0):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature
        }
    }
    start = time.time()

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=DEFAULT_TIMEOUT,
            stream=True
        )
        response.raise_for_status()

        full_text = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                # collect content (skip thinking tokens)
                if "message" in chunk:
                    content = chunk["message"].get("content", "")
                    if content:  # only add non-empty content
                        full_text += content
                # keep going until done=True, don't break early
                if chunk.get("done", False):
                    break

        elapsed = time.time() - start
        return {"text": full_text, "time": elapsed, "raw": {"response": full_text}}

    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")