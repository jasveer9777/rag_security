# models/ollama_client.py
import os
import time
import requests
import json

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_TIMEOUT = 120

def ollama_generate(prompt, model="gpt-oss:20b", max_tokens=512, temperature=0.0):
    """
    Call local Ollama generate endpoint using streaming API.
    Returns {'text':..., 'time':..., 'raw':...}
    """
    payload = {
        "model": model,
        "prompt": prompt,
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
        
        # Collect the streaming response
        full_text = ""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "response" in chunk:
                    full_text += chunk["response"]
                if chunk.get("done", False):
                    break
        
        elapsed = time.time() - start
        return {"text": full_text, "time": elapsed, "raw": {"response": full_text}}
        
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}")
