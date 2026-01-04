# src/compare_models.py
"""
Run same question across multiple models with RAG and output comparisons.csv
"""
import argparse, csv
from serve_query import serve

MODELS = [
    "openai:gpt-4o-mini",
    "ollama:ggml-mistral-7b"
]

def compare(question, out_csv, k, cutoff, min_sim, strict):
    rows = []

    for model in MODELS:
        try:
            print(f"[RUNNING] {model}")
            ans, retrieved, _ = serve(
                question, model, k, min_sim, cutoff, strict
            )
            rows.append({
                "model": model,
                "answer": ans.replace("\n", " "),
                "retrieved_chunks": ";".join([
                    f"{r['meta']['source_path']}#c{r['meta']['chunk_index']}"
                    for r in retrieved
                ])
            })
        except Exception as e:
            rows.append({
                "model": model,
                "answer": f"ERROR: {e}",
                "retrieved_chunks": ""
            })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "answer", "retrieved_chunks"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    print(f"[DONE] Saved comparison → {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--question", required=True)
    p.add_argument("--out", default="comparisons.csv")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--cutoff", default=None)
    p.add_argument("--min_sim", type=float, default=0.0)
    p.add_argument("--strict", action="store_true")
    args = p.parse_args()

    compare(args.question, args.out, args.k, args.cutoff, args.min_sim, args.strict)
