"""Sequence length benchmark for multilingual-e5-small on existing HHGOA index.

Benchmarks query embedding at max_seq_length 32, 128, and 256 on validation
queries. Reports Recall@5, Recall@20, and embedding latency per setting.

Per HHGOA Task 2 architecture document:
    "Benchmark sequence lengths 32, 128, and 256 on the existing validation
    queries; do not choose a value without reporting Recall@k and latency."

Usage
-----
    .venv\\Scripts\\python.exe bench\\benchmark_seq_lengths.py

Outputs a results table to stdout and writes bench/seq_len_results.json.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is on path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

# ---------------------------------------------------------------------------
# Load validation queries from the HHGOA validation parquet
# ---------------------------------------------------------------------------

def load_validation_queries(
    parquet_path: Path,
    n: int = 200,
) -> List[Tuple[str, List[str]]]:
    """Load (query_text, relevant_passage_ids) pairs from validation parquet.

    Returns list of (query, [relevant_passage_text, ...]) tuples.
    We treat exact passage text match as the ground-truth signal since
    the index uses passage text as the primary key.
    """
    import pandas as pd

    print(f"[bench] Loading validation parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"[bench] Loaded {len(df)} rows. Columns: {list(df.columns)}")

    pairs: List[Tuple[str, List[str]]] = []
    for _, row in df.head(n).iterrows():
        query = str(row.get("query", row.get("question", ""))).strip()
        if not query:
            continue

        # Extract relevant passages — MSMARCO-XI stores them in passages.passage_text
        passages_field = row.get("passages", {})
        relevant: List[str] = []
        if isinstance(passages_field, dict):
            texts = passages_field.get("passage_text")
            if texts is None or (isinstance(texts, (list, np.ndarray)) and len(texts) == 0):
                texts = passages_field.get("Translated_passages")
            if texts is None:
                texts = []

            is_selected = passages_field.get("is_selected")
            has_selection = False
            if is_selected is not None and len(is_selected) > 0:
                has_selection = any(bool(s) for s in is_selected)

            if has_selection and is_selected is not None:
                for t, sel in zip(texts, is_selected):
                    if sel:
                        relevant.append(str(t).strip())
            else:
                for t in texts:
                    if t:
                        relevant.append(str(t).strip())
        elif isinstance(passages_field, (list, np.ndarray)):
            for p in passages_field:
                if p:
                    relevant.append(str(p).strip())

        if query and relevant:
            pairs.append((query, relevant))

    print(f"[bench] Extracted {len(pairs)} valid query-passage pairs.\n")
    return pairs


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def benchmark_seq_length(
    seq_len: int,
    queries: List[Tuple[str, List[str]]],
    chunk_list: List,
    faiss_index,
    k_values: List[int] = [5, 20],
) -> Dict:
    """Run retrieval for all queries at a given max_seq_length.

    Returns dict with recall@k values and average embedding latency.
    """
    from sentence_transformers import SentenceTransformer
    import torch
    import faiss as faiss_lib

    print(f"\n{'='*60}")
    print(f"  Benchmarking max_seq_length = {seq_len}")
    print(f"{'='*60}")

    model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    model.max_seq_length = seq_len
    torch.set_num_threads(min(8, os.cpu_count() or 4))

    # Build text set from chunk list for passage matching
    chunk_texts = [c.text.strip() for c in chunk_list]

    embed_latencies = []
    hits: Dict[int, int] = {k: 0 for k in k_values}
    total = 0

    for query_text, relevant_passages in queries:
        # Build a set of first 80 chars for fast substring matching
        relevant_fingerprints = set()
        for rp in relevant_passages:
            relevant_fingerprints.add(rp[:80])

        # Embed query
        t0 = time.perf_counter()
        prefixed = f"query: {query_text}"
        with torch.inference_mode():
            q_vec = model.encode(
                prefixed,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).astype(np.float32).reshape(1, -1)
        embed_ms = (time.perf_counter() - t0) * 1000.0
        embed_latencies.append(embed_ms)

        # FAISS search at max k
        max_k = max(k_values)
        distances, indices = faiss_index.search(q_vec, max_k)
        retrieved_indices = [int(i) for i in indices[0] if i >= 0]

        # Check recall at each k
        for k in k_values:
            top_k_indices = retrieved_indices[:k]
            for idx in top_k_indices:
                if idx < len(chunk_texts):
                    chunk_fp = chunk_texts[idx][:80]
                    if chunk_fp in relevant_fingerprints:
                        hits[k] += 1
                        break

        total += 1
        if total % 50 == 0:
            print(f"  ... {total}/{len(queries)} queries done")

    avg_embed_ms = sum(embed_latencies) / max(1, len(embed_latencies))
    p95_embed_ms = float(np.percentile(embed_latencies, 95)) if embed_latencies else 0.0

    result = {
        "max_seq_length": seq_len,
        "n_queries": total,
        "avg_embed_ms": round(avg_embed_ms, 2),
        "p95_embed_ms": round(p95_embed_ms, 2),
    }
    for k in k_values:
        recall = hits[k] / max(1, total)
        result[f"recall@{k}"] = round(recall, 4)
        print(f"  Recall@{k}: {recall:.4f}  ({hits[k]}/{total})")

    print(f"  Avg embed latency: {avg_embed_ms:.2f}ms  |  P95: {p95_embed_ms:.2f}ms")
    return result


def main():
    import faiss

    index_dir = _root / "index"
    chunk_list_path = index_dir / "chunk_list.pkl"
    faiss_path = index_dir / "faiss_hnswflat.index"
    parquet_path = _root / "data" / "validation_hinval.parquet"

    if not chunk_list_path.exists() or not faiss_path.exists():
        print("[ERROR] Index files not found. Run build_index first.")
        sys.exit(1)

    if not parquet_path.exists():
        print("[ERROR] Validation parquet not found at data/validation_hinval.parquet")
        sys.exit(1)

    print("[bench] Loading FAISS index ...")
    faiss_index = faiss.read_index(str(faiss_path))
    if hasattr(faiss_index, "hnsw"):
        faiss_index.hnsw.efSearch = 64
    print(f"[bench] FAISS index loaded: {faiss_index.ntotal} vectors")

    print("[bench] Loading chunk list ...")
    with open(chunk_list_path, "rb") as f:
        chunk_list = pickle.load(f)
    print(f"[bench] Loaded {len(chunk_list)} chunks\n")

    queries = load_validation_queries(parquet_path, n=200)
    if not queries:
        print("[ERROR] No validation queries loaded.")
        sys.exit(1)

    results = []
    for seq_len in [32, 128, 256]:
        r = benchmark_seq_length(seq_len, queries, chunk_list, faiss_index)
        results.append(r)

    # Print comparison table
    print(f"\n\n{'='*70}")
    print("  SEQ LENGTH BENCHMARK RESULTS")
    print(f"{'='*70}")
    header = f"  {'seq_len':>10}  {'Recall@5':>10}  {'Recall@20':>10}  {'avg_ms':>8}  {'p95_ms':>8}"
    print(header)
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    for r in results:
        print(
            f"  {r['max_seq_length']:>10}  "
            f"{r.get('recall@5', 0):>10.4f}  "
            f"{r.get('recall@20', 0):>10.4f}  "
            f"{r['avg_embed_ms']:>8.2f}  "
            f"{r['p95_embed_ms']:>8.2f}"
        )
    print(f"{'='*70}\n")

    # Pick best seq_len: highest Recall@5 within latency budget (< 50ms P95)
    candidates = [r for r in results if r["p95_embed_ms"] < 50.0]
    if candidates:
        best = max(candidates, key=lambda r: r.get("recall@5", 0))
    else:
        best = max(results, key=lambda r: r.get("recall@5", 0))

    print(f"  RECOMMENDATION: max_seq_length = {best['max_seq_length']}")
    print(f"  (Recall@5={best.get('recall@5', 0):.4f}, P95 embed={best['p95_embed_ms']:.2f}ms)\n")

    out_path = _root / "bench" / "seq_len_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "recommended_seq_len": best["max_seq_length"]}, f, indent=2)
    print(f"[bench] Results written to {out_path}")


if __name__ == "__main__":
    main()
