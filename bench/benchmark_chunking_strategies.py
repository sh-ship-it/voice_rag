"""Benchmark Fixed-size, Semantic, Small-to-Big, and Combined indexes separately.

Evaluates on the exact same 200 labeled validation queries from data/validation_hinval.parquet.
Reports Recall@5, Recall@10, Recall@20, MRR, and Retrieval Latency (ms).

Per HHGOA Task 2 Blocker 3 requirements.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def load_validation_pairs(parquet_path: Path, n: int = 200) -> List[Tuple[str, List[str]]]:
    """Load query -> relevant Hindi passages from parquet."""
    df = pd.read_parquet(parquet_path)
    pairs: List[Tuple[str, List[str]]] = []
    for _, row in df.head(n).iterrows():
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        passages_field = row.get("passages", {})
        relevant: List[str] = []
        if isinstance(passages_field, dict):
            texts = passages_field.get("Translated_passages")
            if texts is None or len(texts) == 0:
                texts = passages_field.get("passage_text") or []
            is_selected = passages_field.get("is_selected")
            has_selection = is_selected is not None and len(is_selected) > 0 and any(bool(s) for s in is_selected)
            if has_selection and is_selected is not None:
                for t, sel in zip(texts, is_selected):
                    if sel and t:
                        relevant.append(str(t).strip())
            else:
                for t in texts:
                    if t:
                        relevant.append(str(t).strip())
        if query and relevant:
            pairs.append((query, relevant))
    return pairs


def eval_strategy(
    name: str,
    strategy_chunk_indices: List[int],
    chunk_list: List,
    full_faiss_index,
    model: SentenceTransformer,
    queries: List[Tuple[str, List[str]]],
    k_vals: List[int] = [5, 10, 20],
) -> Dict:
    """Evaluate retrieval metrics for a specific chunking strategy subset."""
    print(f"\nEvaluating strategy: '{name}' ({len(strategy_chunk_indices):,} chunks) ...")
    
    # Sub-index chunk texts for fingerprint matching
    idx_set = set(strategy_chunk_indices)
    
    recalls = {k: 0 for k in k_vals}
    mrr_total = 0.0
    latencies = []
    total = len(queries)
    
    for q_text, relevant_passages in queries:
        rel_fps = {rp[:80] for rp in relevant_passages}
        
        # Embed query
        t0 = time.perf_counter()
        q_vec = model.encode(f"query: {q_text}", normalize_embeddings=True, show_progress_bar=False).astype(np.float32).reshape(1, -1)
        
        # Search FAISS at large top_k, filter to strategy subset
        distances, indices = full_faiss_index.search(q_vec, 200)
        dur_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(dur_ms)
        
        # Filter retrieved indices to this strategy's chunks
        filtered_retrieved = []
        for idx_int in indices[0]:
            if idx_int in idx_set:
                filtered_retrieved.append(idx_int)
                if len(filtered_retrieved) >= max(k_vals):
                    break
                    
        # Check rank of first hit
        first_hit_rank = None
        for rank, c_idx in enumerate(filtered_retrieved, 1):
            chunk_fp = chunk_list[c_idx].text.strip()[:80]
            if chunk_fp in rel_fps:
                first_hit_rank = rank
                break
                
        # Update Recall@k
        for k in k_vals:
            if first_hit_rank is not None and first_hit_rank <= k:
                recalls[k] += 1
                
        # Update MRR
        if first_hit_rank is not None:
            mrr_total += 1.0 / first_hit_rank
            
    avg_latency = float(np.mean(latencies))
    mrr = mrr_total / max(1, total)
    
    res = {
        "strategy": name,
        "chunk_count": len(strategy_chunk_indices),
        "recall@5": round(recalls[5] / total, 4),
        "recall@10": round(recalls[10] / total, 4),
        "recall@20": round(recalls[20] / total, 4),
        "mrr": round(mrr, 4),
        "avg_latency_ms": round(avg_latency, 2),
    }
    return res


def main():
    index_dir = _root / "index"
    parquet_path = _root / "data" / "validation_hinval.parquet"
    
    print("[1/3] Loading validation dataset pairs ...")
    queries = load_validation_pairs(parquet_path, n=200)
    print(f"  Loaded {len(queries)} labeled query-passage evaluation pairs.")
    
    print("\n[2/3] Loading warm embedding model and FAISS index ...")
    model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    model.max_seq_length = 128
    
    full_faiss = faiss.read_index(str(index_dir / "faiss_hnswflat.index"))
    if hasattr(full_faiss, "hnsw"):
        full_faiss.hnsw.efSearch = 64
        
    with open(index_dir / "chunk_list.pkl", "rb") as f:
        chunk_list = pickle.load(f)
        
    # Group indices by chunk strategy
    strat_indices = {
        "Fixed-Size Chunks": [i for i, c in enumerate(chunk_list) if c.chunk_strategy == "fixed_size"],
        "Semantic Chunks": [i for i, c in enumerate(chunk_list) if c.chunk_strategy == "semantic"],
        "Small-to-Big Chunks": [i for i, c in enumerate(chunk_list) if c.chunk_strategy == "small_to_big"],
        "Combined (All Strategies)": list(range(len(chunk_list))),
    }
    
    print("\n[3/3] Running Strategy-by-Strategy Comparison ...")
    results = []
    for name, indices in strat_indices.items():
        r = eval_strategy(name, indices, chunk_list, full_faiss, model, queries)
        results.append(r)
        
    print("\n" + "="*86)
    print("  CHUNKING STRATEGIES COMPARISON BENCHMARK TABLE")
    print("="*86)
    header = f"{'Strategy':<26} {'Chunks':>8} {'Recall@5':>10} {'Recall@10':>10} {'Recall@20':>10} {'MRR':>8} {'Avg Lat (ms)':>12}"
    print(header)
    print("-"*86)
    for r in results:
        print(
            f"{r['strategy']:<26} {r['chunk_count']:>8,} "
            f"{r['recall@5']:>10.4f} {r['recall@10']:>10.4f} {r['recall@20']:>10.4f} "
            f"{r['mrr']:>8.4f} {r['avg_latency_ms']:>12.2f}"
        )
    print("="*86)
    
    out_path = _root / "bench" / "chunking_strategy_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}\n")


if __name__ == "__main__":
    main()
