# src/evaluate.py
"""
Evaluate RAG system on prepared QA dataset.
Each line in qa_dataset.jsonl:
{
  "id": "1",
  "question": "...",
  "gold_answer": "...",
  "gold_chunk_ids": ["uuid_3", "uuid_5"]
}
"""

import argparse, json, csv, re
from collections import Counter
from serve_query import serve


def exact_match(a, b):
    return int(a.strip().lower() == b.strip().lower())


def f1(pred, gold):
    t1 = re.findall(r"\w+", pred.lower())
    t2 = re.findall(r"\w+", gold.lower())
    common = Counter(t1) & Counter(t2)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    p = overlap / len(t1)
    r = overlap / len(t2)
    return 2 * p * r / (p + r)


def precision_at_k(retrieved, gold_chunk_ids):
    for r in retrieved:
        if r["meta"]["id"] in gold_chunk_ids:
            return 1
    return 0


def evaluate(qa_path, out_csv, model, k):
    rows = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q = item["question"]
            gold = item.get("gold_answer", "")
            gold_ids = item.get("gold_chunk_ids", [])

            ans, retrieved, _ = serve(q, model, k, 0.0, None, True)

            rows.append({
                "id": item["id"],
                "question": q,
                "precision_k": precision_at_k(retrieved, gold_ids),
                "em": exact_match(ans, gold),
                "f1": f1(ans, gold)
            })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question", "precision_k", "em", "f1"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[DONE] Saved evaluation → {out_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--qa", required=True)
    p.add_argument("--out", default="eval_report.csv")
    p.add_argument("--model", default="ollama:gpt-oss:20b")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()

    evaluate(args.qa, args.out, args.model, args.k)
