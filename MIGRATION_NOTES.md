# Migration to Azure-hosted Claude (Anthropic)

## Date: February 3, 2026

## Summary
Successfully migrated the RAG Security project from OpenAI to Azure-hosted Claude Sonnet 4.5.

## Changes Made

### 1. New Files Created
- **`models/anthropic_client.py`**: New client wrapper for Azure-hosted Claude API
  - Handles authentication with Azure's x-api-key header
  - Uses Azure Anthropic endpoint format
  - Returns consistent response format with OpenAI client

### 2. Updated Files

#### `requirements.txt`
- Added `anthropic` package for Azure Claude support
- Kept `openai` for backward compatibility (commented in code)

#### `src/serve_query.py`
- Added import for `anthropic_client`
- Updated `call_model()` to support `anthropic:` prefix
- Changed default model from `openai:gpt-4o-mini` to `anthropic:claude-sonnet-4-5-saarathi02`
- Added .env loading support
- Old OpenAI configuration commented but preserved

#### `src/interactive_openai.py`
- Changed default model to `anthropic:claude-sonnet-4-5-saarathi02`
- Old OpenAI default commented but preserved

#### `src/comapre_models.py`
- Updated model list to use Azure Claude as primary
- Old OpenAI and Ollama models commented but preserved

#### `src/evaluate.py`
- Changed default model to Azure Claude
- Old Ollama default commented but preserved

### 3. Configuration (.env)
Current environment variables:
```
ANTHROPIC_API_KEY=<your-key>
ANTHROPIC_ENDPOINT=https://bdo-internal01-resource.services.ai.azure.com/anthropic/
ANTHROPIC_DEPLOYMENT=claude-sonnet-4-5-saarathi02
```

## Usage

### Running Queries

**Single query with CLI:**
```bash
python src/serve_query.py --question "What is attempt to murder?" --k 5
```

**Interactive mode:**
```bash
python src/interactive_openai.py
```

**Using specific model:**
```bash
# Use Azure Claude (default)
python src/serve_query.py --question "Your question" --model anthropic:claude-sonnet-4-5-saarathi02

# Use OpenAI (if API key is set)
python src/serve_query.py --question "Your question" --model openai:gpt-4o-mini

# Use Ollama local model
python src/serve_query.py --question "Your question" --model ollama:ggml-mistral-7b
```

## Model Prefix Convention

- `anthropic:` - Azure-hosted Claude models (NEW)
- `openai:` - OpenAI models (OLD, still supported)
- `ollama:` - Local Ollama models (still supported)

## Testing

Tested successfully with:
- Query: "attempt to murder punishment"
- Retrieved 5 relevant chunks from IPC and BNS
- Claude provided accurate, well-formatted response citing Section 307 IPC
- Response time: ~3-4 seconds

## Backward Compatibility

All old configurations are preserved as comments. To revert to OpenAI:
1. Uncomment OpenAI defaults in source files
2. Comment out Anthropic defaults
3. Set `OPENAI_API_KEY` in .env
4. No code changes needed - just parameter updates

## Notes

- Azure Anthropic API uses `x-api-key` header (not `api-key`)
- Endpoint format: `{ANTHROPIC_ENDPOINT}/v1/messages`
- Model specified in payload, not in URL path
- Response format: `{"content": [{"type": "text", "text": "..."}], "usage": {...}}`
