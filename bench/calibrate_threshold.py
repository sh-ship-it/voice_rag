"""Calibrate confidence threshold using HHGOA validation parquet.

Per HHGOA Task 2 architecture document:
    "Create a small validation file from the HHGOA task examples and manually
    label whether the correct passage appears in the top 5, top 20, and top 50.
    Use the validation data to choose thresholds for:
    - Answer: Strong reranker score, clear top-1/top-2 margin, one supporting span
    - Answer cautiously: Moderate reranker score with two agreeing passages
    - Refuse: No strong candidate, contradictory candidates, no citation span"

Usage
-----
    .venv\\Scripts\\python.exe bench\\calibrate_threshold.py

Outputs recommended thresholds to bench/calibrated_thresholds.json and prints
a summary table.
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

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np


def load_validation_pairs(parquet_path: Path, n: int = 200) -> List[Tuple[str, List[str]]]:
    """Load (query, [relevant_passage_texts]) pairs from validation parquet."""
    import pandas as pd

    print(f"[calib] Loading validation parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    pairs: List[Tuple[str, List[str]]] = []
    for _, row in df.head(n).iterrows():
        query = str(row.get("query", row.get("question", ""))).strip()
        if not query:
            continue

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

    print(f"[calib] Extracted {len(pairs)} valid pairs.\n")
    return pairs


def run_calibration(pairs: List[Tuple[str, List[str]]], n_queries: int = 200) -> Dict:
    """Run retrieval for each query and collect RRF scores at hit positions.

    Returns a dict with:
    - hit_rrf_scores: RRF scores where the relevant passage was found in top-5
    - miss_rrf_scores: Top-1 RRF scores where relevant passage was NOT in top-5
    - hit_at_k: counts of hits at k=5, k=20
    - recommended thresholds
    """
    from pipeline.embed import embed_query
    from pipeline.retrieve import hybrid_retrieve

    print("[calib] Running retrieval on validation queries ...\n")

    hit_rrf_top1 = []   # RRF score of top-1 when relevant passage IS in top-5
    miss_rrf_top1 = []  # RRF score of top-1 when relevant passage is NOT in top-5
    margin_scores = []  # Score margin between rank-1 and rank-2

    hits_at_5 = 0
    hits_at_20 = 0
    total = 0

    for query_text, relevant_passages in pairs[:n_queries]:
        relevant_fps = {rp[:80] for rp in relevant_passages}

        # Embed query
        q_emb = embed_query(query_text)

        # Retrieve top-20 (use enable_rerank=False for calibration speed)
        result = hybrid_retrieve(
            query_embedding=q_emb,
            query_text=query_text,
            top_k=20,
            dense_candidates=50,
            sparse_candidates=50,
            enable_rerank=False,
        )

        chunk_texts = [sc.chunk.text[:80] for sc in result.chunks]
        rrf_scores = [sc.score for sc in result.chunks]

        # Check hit@5
        hit5 = any(ct in relevant_fps for ct in chunk_texts[:5])
        # Check hit@20
        hit20 = any(ct in relevant_fps for ct in chunk_texts[:20])

        if hit5:
            hits_at_5 += 1
        if hit20:
            hits_at_20 += 1

        if rrf_scores:
            top1_score = rrf_scores[0]
            margin = (rrf_scores[0] - rrf_scores[1]) if len(rrf_scores) > 1 else rrf_scores[0]

            if hit5:
                hit_rrf_top1.append(top1_score)
                margin_scores.append(margin)
            else:
                miss_rrf_top1.append(top1_score)

        total += 1
        if total % 50 == 0:
            print(f"  ... {total}/{n_queries} queries done")

    print(f"\n[calib] Total queries: {total}")
    print(f"[calib] Hit@5:  {hits_at_5}/{total} = {hits_at_5/max(1,total):.4f}")
    print(f"[calib] Hit@20: {hits_at_20}/{total} = {hits_at_20/max(1,total):.4f}")

    # Determine thresholds
    # "Answer" threshold: score above which 90%+ of retrievals are hits
    # Use 25th percentile of hit scores as the "answer" threshold
    # Use 10th percentile of hit scores as the "answer cautiously" threshold
    answer_threshold = float(np.percentile(hit_rrf_top1, 25)) if hit_rrf_top1 else 0.018
    cautious_threshold = float(np.percentile(hit_rrf_top1, 10)) if hit_rrf_top1 else 0.012
    refuse_threshold = cautious_threshold * 0.7  # below this, refuse

    print(f"\n[calib] Recommended thresholds:")
    print(f"  ANSWER threshold (high confidence):    {answer_threshold:.6f}")
    print(f"  CAUTIOUS threshold (medium confidence): {cautious_threshold:.6f}")
    print(f"  REFUSE threshold (low confidence):      {refuse_threshold:.6f}")

    # Print distribution stats
    if hit_rrf_top1:
        print(f"\n[calib] Hit RRF top-1 scores:")
        print(f"  min={min(hit_rrf_top1):.6f}  p10={np.percentile(hit_rrf_top1, 10):.6f}  "
              f"p25={np.percentile(hit_rrf_top1, 25):.6f}  p50={np.percentile(hit_rrf_top1, 50):.6f}  "
              f"max={max(hit_rrf_top1):.6f}")
    if miss_rrf_top1:
        print(f"\n[calib] Miss RRF top-1 scores:")
        print(f"  min={min(miss_rrf_top1):.6f}  p50={np.percentile(miss_rrf_top1, 50):.6f}  "
              f"max={max(miss_rrf_top1):.6f}")

    return {
        "n_queries": total,
        "hit_at_5": hits_at_5,
        "hit_at_20": hits_at_20,
        "hit_rate_5": round(hits_at_5 / max(1, total), 4),
        "hit_rate_20": round(hits_at_20 / max(1, total), 4),
        "recommended": {
            "CONFIDENCE_THRESHOLD_ANSWER": round(answer_threshold, 6),
            "CONFIDENCE_THRESHOLD_CAUTIOUS": round(cautious_threshold, 6),
            "CONFIDENCE_THRESHOLD_REFUSE": round(refuse_threshold, 6),
        }
    }


def main():
    parquet_path = _root / "data" / "validation_hinval.parquet"
    if not parquet_path.exists():
        print("[ERROR] Validation parquet not found at data/validation_hinval.parquet")
        sys.exit(1)

    pairs = load_validation_pairs(parquet_path, n=200)
    if not pairs:
        print("[ERROR] No validation pairs loaded.")
        sys.exit(1)

    results = run_calibration(pairs, n_queries=min(200, len(pairs)))

    out_path = _root / "bench" / "calibrated_thresholds.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[calib] Results written to {out_path}")
    print("\n[calib] Add to .env:")
    for k, v in results["recommended"].items():
        print(f"  {k}={v}")


if __name__ == "__main__":
    main()
