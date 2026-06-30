# models/hf.py
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, GenerationConfig

_pipeline = None

def load_hf_pipeline(model_path: str):
    global _pipeline
    if _pipeline is None:
        print(f"[HF] Loading model from {model_path} ...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            clean_up_tokenization_spaces=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16,
            device_map="auto"
        )

        # Fix: set generation config on model to avoid conflicts
        model.generation_config = GenerationConfig(
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

        _pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        print("[HF] Model loaded successfully.")
    return _pipeline


def hf_generate(prompt: str, model_path: str, max_tokens: int = 512, temperature: float = 0.1):
    pipe = load_hf_pipeline(model_path)
    start = time.time()

    # Pass ALL generation params explicitly — no generation_config conflict
    generate_kwargs = {
        "max_new_tokens": max_tokens,
        "return_full_text": False,
        "pad_token_id": pipe.tokenizer.eos_token_id,
    }

    if temperature > 0:
        generate_kwargs["do_sample"]    = True
        generate_kwargs["temperature"]  = temperature
    else:
        generate_kwargs["do_sample"]    = False

    outputs = pipe(prompt, **generate_kwargs)
    elapsed = time.time() - start
    text    = outputs[0]["generated_text"].strip()
    return {"text": text, "time": elapsed}