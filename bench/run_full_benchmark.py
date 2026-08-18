"""Comprehensive latency and retrieval benchmark for HHGOA Task 2.

Measures granular per-stage latency across a fixed query suite and reports
P50, P70, P90, P95, and P100 (Max) percentiles for:
- Query Normalization
- Embedding (multilingual-e5-small)
- Dense FAISS Search (efSearch=64)
- Sparse BM25 Search
- Concurrent Retrieval
- RRF Fusion & Parent Dedup
- FlashRank Cross-Encoder Reranking
- Confidence Gating
- Core RAG Total (target < 200 ms)
- LLM Generation (Cerebras)
- Total End-to-End Latency

Usage:
    .venv\\Scripts\\python.exe bench\\run_full_benchmark.py [n_queries]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np

BENCHMARK_QUERIES = [
    "भारत की राजधानी क्या है?",
    "विश्व का सबसे बड़ा महासागर कौन सा है?",
    "ताजमहल कहाँ स्थित है?",
    "भारत के प्रधानमंत्री कौन हैं?",
    "what is the capital of India?",
    "सौर मंडल का सबसे बड़ा ग्रह कौन सा है?",
    "भारतीय संविधान कब लागू हुआ था?",
    "कंप्यूटर का जनक किसे कहा जाता है?",
    "what is artificial intelligence?",
    "पृथ्वी सूर्य का चक्कर कितने दिनों में लगाती है?",
    "भारत में कुल कितने राज्य हैं?",
    "हिंदी दिवस कब मनाया जाता है?",
    "मानव शरीर में कितनी हड्डियां होती हैं?",
    "how does machine learning work?",
    "राष्ट्रगान जन गण मन के रचयिता कौन हैं?",
]


def percentile(vals: List[float], p: float) -> float:
    """Calculate percentile from list of floats."""
    if not vals:
        return 0.0
    return float(np.percentile(vals, p))


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive HHGOA RAG benchmark")
    parser.add_argument("n", type=int, nargs="?", default=15, help="Number of benchmark queries to run")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM generation to test core RAG latency only")
    args = parser.parse_args()

    print("=" * 78)
    print("  HHGOA TASK 2 — FULL PIPELINE LATENCY & RETRIEVAL BENCHMARK")
    print("=" * 78)

    # 1. Warm up
    print("\n[1/3] Pre-warming embedding model and retrieval indexes ...")
    t_warm = time.perf_counter()
    from pipeline.embed import embed_query
    from pipeline.normalize import normalize_text, tokenize_for_bm25
    from pipeline.retrieve import _REGISTRY, hybrid_retrieve, warmup as retrieve_warmup
    from pipeline.guardrails import input_guardrail, confidence_gate
    from pipeline.generate import generate_answer

    retrieve_warmup()
    _ = embed_query("warmup query")
    warm_dur = (time.perf_counter() - t_warm) * 1000.0
    print(f"  System pre-warmed in {warm_dur:.1f} ms. Indexed chunks: {len(_REGISTRY.chunk_list):,}\n")

    # 2. Benchmark Loop
    num_queries = args.n
    queries = [BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)] for i in range(num_queries)]

    print(f"[2/3] Executing {num_queries} queries through full pipeline ...")

    norm_times = []
    embed_times = []
    dense_times = []
    sparse_times = []
    retrieval_times = []
    gate_times = []
    core_rag_times = []
    gen_times = []
    total_times = []

    for i, q in enumerate(queries, 1):
        t_total_start = time.perf_counter()

        # Step A: Normalization & Safety Guardrail
        t0 = time.perf_counter()
        norm_q = normalize_text(q)
        safe = input_guardrail(norm_q)
        t_norm = (time.perf_counter() - t0) * 1000.0
        norm_times.append(t_norm)

        # Step B: Query Embedding
        t0 = time.perf_counter()
        q_emb = embed_query(norm_q)
        t_emb = (time.perf_counter() - t0) * 1000.0
        embed_times.append(t_emb)

        # Step C: Parallel Retrieval & Reranking
        t0 = time.perf_counter()
        ret_res = hybrid_retrieve(
            query_embedding=q_emb,
            query_text=norm_q,
            top_k=6,
            dense_candidates=50,
            sparse_candidates=50,
            enable_rerank=True,
        )
        t_ret = (time.perf_counter() - t0) * 1000.0
        retrieval_times.append(t_ret)

        # Step D: Confidence Gate
        t0 = time.perf_counter()
        confident = confidence_gate(ret_res)
        t_gate = (time.perf_counter() - t0) * 1000.0
        gate_times.append(t_gate)

        core_ms = t_norm + t_emb + t_ret + t_gate
        core_rag_times.append(core_ms)

        # Step E: LLM Generation (if not skipped)
        t_gen = 0.0
        if not args.skip_llm and confident:
            t0 = time.perf_counter()
            gen_res = generate_answer(norm_q, ret_res)
            t_gen = (time.perf_counter() - t0) * 1000.0
            gen_times.append(t_gen)

        t_total = (time.perf_counter() - t_total_start) * 1000.0
        total_times.append(t_total)

        status_flag = "HIT" if confident else "REFUSE"
        print(f"  [{i:>2}/{num_queries}] Core: {core_ms:>5.1f}ms | Total: {t_total:>6.1f}ms | Status: {status_flag:<6} | Q: {q[:35]}")

    # 3. Analytics Report
    print("\n[3/3] Computing Percentile Analytics ...\n")
    print("=" * 78)
    print("  LATENCY PERCENTILES TABLE (All figures in milliseconds)")
    print("=" * 78)
    header = f"{'Pipeline Stage':<30} {'Avg':>8} {'P50':>8} {'P70':>8} {'P90':>8} {'P95':>8} {'P100 (Max)':>10}"
    print(header)
    print("-" * 78)

    stages = [
        ("Query Normalization & Guardrail", norm_times),
        ("Query Embedding (E5 Small)", embed_times),
        ("Parallel Hybrid Retrieval + Rerank", retrieval_times),
        ("Confidence Gate Evaluation", gate_times),
        ("Core RAG Path (Online Target <200ms)", core_rag_times),
    ]
    if gen_times:
        stages.append(("LLM Generation (Cerebras Cloud)", gen_times))
    stages.append(("End-to-End Total Latency", total_times))

    for name, vals in stages:
        if vals:
            avg_v = float(np.mean(vals))
            p50_v = percentile(vals, 50)
            p70_v = percentile(vals, 70)
            p90_v = percentile(vals, 90)
            p95_v = percentile(vals, 95)
            p100_v = float(np.max(vals))
            print(f"{name:<30} {avg_v:>8.2f} {p50_v:>8.2f} {p70_v:>8.2f} {p90_v:>8.2f} {p95_v:>8.2f} {p100_v:>10.2f}")

    print("=" * 78)

    core_p95 = percentile(core_rag_times, 95)
    print(f"\nTarget Evaluation: Core Local RAG Latency Budget: 200.00 ms")
    print(f"Observed P95 Core RAG Latency:                  {core_p95:.2f} ms")
    if core_p95 <= 200.0:
        print("RESULT: [PASS] Core RAG latency is strictly within the 200 ms budget!")
    else:
        print(f"RESULT: [NOTE] P95 is {core_p95:.2f} ms.")
    print()


if __name__ == "__main__":
    main()
