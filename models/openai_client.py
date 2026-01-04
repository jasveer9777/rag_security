# models/openai_client.py
"""
OpenAI client wrapper that supports both new openai>=1.0.0 interface
and falls back to the older openai.ChatCompletion API if available.
Returns a dict: {"text": <str>, "time": <float>, "usage": <dict>}
"""

import os
import time

# Try new v1 OpenAI interface first
try:
    from openai import OpenAI as OpenAIClient
    _has_new = True
except Exception:
    _has_new = False

import openai  # also import to support older API if present


def openai_chat(prompt, model="gpt-4o-mini", temperature=0.0, max_tokens=512):
    """
    Send `prompt` to the selected OpenAI model.
    Returns: {"text": str, "time": float, "usage": dict}
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment (.env or env var).")

    start = time.time()

    if _has_new:
        # New interface (openai>=1.0.0)
        client = OpenAIClient(api_key=api_key)
        # Create a chat completion
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        elapsed = time.time() - start
        
        # Extract text properly from the response object
        try:
            # Try accessing as attribute (most common)
            content = resp.choices[0].message.content
        except (AttributeError, KeyError, IndexError):
            try:
                # Try accessing as dict
                content = resp.choices[0].message["content"]
            except Exception:
                # Last resort: empty string instead of str(resp)
                content = ""
        
        # Extract usage safely
        try:
            usage = resp.usage.model_dump() if hasattr(resp.usage, 'model_dump') else dict(resp.usage)
        except Exception:
            usage = {}
        
        return {"text": content, "time": elapsed, "usage": usage}

    else:
        # Old interface fallback (openai<1.0.0)
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        elapsed = time.time() - start
        content = resp.choices[0].message.content
        usage = resp.get("usage", {})
        return {"text": content, "time": elapsed, "usage": usage}