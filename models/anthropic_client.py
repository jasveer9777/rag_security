# models/anthropic_client.py
"""
Anthropic Claude client wrapper for Azure-hosted Claude models.
Supports Azure OpenAI Service with Anthropic API.
Returns a dict: {"text": <str>, "time": <float>, "usage": <dict>}
"""

import os
import time
import requests


def anthropic_chat(prompt, model="claude-sonnet-4-5-saarathi02", temperature=0.0, max_tokens=512):
    """
    Send `prompt` to Azure-hosted Claude model.
    Returns: {"text": str, "time": float, "usage": dict}
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    endpoint = os.getenv("ANTHROPIC_ENDPOINT")
    deployment = os.getenv("ANTHROPIC_DEPLOYMENT", model)
    
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in environment (.env or env var).")
    if not endpoint:
        raise ValueError("ANTHROPIC_ENDPOINT not set in environment (.env or env var).")
    
    # Construct Azure Anthropic endpoint URL
    # Azure format: https://<resource>.services.ai.azure.com/anthropic/v1/messages
    url = f"{endpoint.rstrip('/')}/v1/messages"
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,  # Azure uses x-api-key for Anthropic
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": deployment,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    start = time.time()
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Anthropic API request failed: {e}")
    
    elapsed = time.time() - start
    
    # Extract content from response
    try:
        # Azure Anthropic response format: {"content": [{"type": "text", "text": "..."}], "usage": {...}}
        content_blocks = data.get("content", [])
        if content_blocks and isinstance(content_blocks, list):
            text = content_blocks[0].get("text", "")
        else:
            text = str(data)
        
        usage = data.get("usage", {})
    except Exception as e:
        raise RuntimeError(f"Failed to parse Anthropic response: {e}")
    
    return {"text": text, "time": elapsed, "usage": usage}
