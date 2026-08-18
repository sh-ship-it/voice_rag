"""bench_generation_modes.py — Latency benchmark for generation paths.

Measures the following paths in isolation using already-built index files:

  Mode A: embed + retrieve + rerank + extractive fallback (no LLM)
  Mode B: embed + retrieve + rerank + Groq generation

Reported metrics: P50, P70, P100 (max) per mode.

IMPORTANT — Latency labels
--------------------------
- "local_core_retrieval_extractive_ms"
    Embed + FAISS/BM25 hybrid retrieve + RRF + FlashRank rerank + extractive snippet.
    This is a LOCAL in-process measurement with the index in RAM.
    Typical range: 30–60 ms.  Does NOT include STT, network round-trip, or response
    transmission.  NOT representative of end-to-end voice latency.

- "groq_generation_ms"
    Time from Groq API call start to first complete JSON response.
    This is a NETWORK measurement and varies with Groq datacenter load, your
    network latency, and model queue depth.  Do NOT assume this equals the
    end-to-end answer latency seen by the user.  Includes:
    • Network round-trip to Groq US datacenter
    • Model inference on Groq hardware
    • JSON response serialization
    Does NOT include: STT, embedding, retrieval, reranking, or frontend rendering.

- "total_rag_core_ms"  (Groq path)
    embed + retrieve + rerank + Groq call.  Excludes STT.

Run
---
    .venv\\Scripts\\python.exe -m bench.bench_generation_modes

Optional env vars
-----------------
    BENCH_QUERIES=10        Number of test queries (default: 10)
    BENCH_WARMUP=2          Warmup iterations (default: 2)
    BENCH_QUERY_FILE=...    Path to JSONL file with {"query": "..."} rows
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure project root is on sys.path
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from pipeline.config import get_settings
from pipeline.embed import embed_query
from pipeline.generate import _extractive_fallback, _call_groq_sync
from pipeline.retrieve import hybrid_retrieve
from pipeline.schemas import RetrievalResult

# ---------------------------------------------------------------------------
# Sample queries (Hindi + English mix)
# ---------------------------------------------------------------------------

DEFAULT_QUERIES = [
    "भारत की राजधानी क्या है?",
    "ताजमहल कहाँ स्थित है?",
    "गंगा नदी कहाँ से निकलती है?",
    "भारत में कितने राज्य हैं?",
    "मुंबई किस राज्य में है?",
    "What is the capital of India?",
    "Where is the Taj Mahal located?",
    "How many states are there in India?",
    "What is the longest river in India?",
    "Who was the first Prime Minister of India?",
]


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def _percentiles(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"count": 0, "p50": 0.0, "p70": 0.0, "p100": 0.0, "mean": 0.0}
    s = sorted(samples)
    n = len(s)

    def pct(p: float) -> float:
        idx = max(0, min(int(p * n), n - 1))
        return round(s[idx], 2)

    return {
        "count": n,
        "mean": round(statistics.mean(samples), 2),
        "p50": pct(0.50),
        "p70": pct(0.70),
        "p100": round(max(samples), 2),  # P100 = worst case
    }


# ---------------------------------------------------------------------------
# Mode A: Retrieveal + Extractive (no LLM)
# ---------------------------------------------------------------------------

def bench_retrieveal_and_extractive(
    queries: List[str],
    warmup: int = 2,
) -> Dict:
    """Benchmark: embed + retrieve + extractive fallback.

    All measurements are in-process with index loaded in RAM.
    """
    settings = get_settings()
    top_k = settings.top_k_final

    # Warmup
    for q in queries[:warmup]:
        emb = embed_query(q)
        ret = hybrid_retrieve(emb, q, top_k=top_k)
        _extractive_fallback(ret, 0.0, "bench-warmup")

    embed_times: List[float] = []
    retrieve_times: List[float] = []
    extractive_times: List[float] = []
    total_times: List[float] = []

    for query in queries:
        t0 = time.perf_counter()

        t_e = time.perf_counter()
        emb = embed_query(query)
        embed_ms = (time.perf_counter() - t_e) * 1000.0

        t_r = time.perf_counter()
        ret = hybrid_retrieve(emb, query, top_k=top_k)
        retrieve_ms = (time.perf_counter() - t_r) * 1000.0

        t_x = time.perf_counter()
        _extractive_fallback(ret, 0.0, "bench")
        extractive_ms = (time.perf_counter() - t_x) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0

        embed_times.append(embed_ms)
        retrieve_times.append(retrieve_ms)
        extractive_times.append(extractive_ms)
        total_times.append(total_ms)

    return {
        "mode": "local_core_retrieval_extractive",
        "label": (
            "Embed + FAISS/BM25 hybrid retrieve + RRF + FlashRank + extractive snippet. "
            "IN-PROCESS only. Does NOT include STT, network, or response transmission."
        ),
        "embed_ms": _percentiles(embed_times),
        "retrieve_ms": _percentiles(retrieve_times),
        "extractive_ms": _percentiles(extractive_times),
        "total_ms": _percentiles(total_times),
    }


# ---------------------------------------------------------------------------
# Mode B: Retrieval + Groq
# ---------------------------------------------------------------------------

def bench_retrieval_and_groq(
    queries: List[str],
    warmup: int = 2,
) -> Dict:
    """Benchmark: embed + retrieve + Groq generation call.

    groq_generation_ms is a NETWORK measurement. Values depend on your
    network latency to Groq US, Groq queue depth, and model load.
    Results MUST be measured from the actual deployed Render environment
    to be representative of production latency.
    """
    settings = get_settings()
    top_k = settings.top_k_final

    if not settings.groq_api_key:
        return {
            "mode": "groq_generation",
            "error": "GROQ_API_KEY not set — skipped. Set it in .env to benchmark Groq.",
        }

    # Warmup
    for q in queries[:warmup]:
        emb = embed_query(q)
        ret = hybrid_retrieve(emb, q, top_k=top_k)
        _call_groq_sync(q, ret, settings)

    embed_times: List[float] = []
    retrieve_times: List[float] = []
    groq_times: List[float] = []
    total_times: List[float] = []
    modes: List[str] = []

    for query in queries:
        t0 = time.perf_counter()

        t_e = time.perf_counter()
        emb = embed_query(query)
        embed_ms = (time.perf_counter() - t_e) * 1000.0

        t_r = time.perf_counter()
        ret = hybrid_retrieve(emb, query, top_k=top_k)
        retrieve_ms = (time.perf_counter() - t_r) * 1000.0

        t_g = time.perf_counter()
        gen = _call_groq_sync(query, ret, settings)
        groq_ms = (time.perf_counter() - t_g) * 1000.0

        total_ms = (time.perf_counter() - t0) * 1000.0

        embed_times.append(embed_ms)
        retrieve_times.append(retrieve_ms)
        groq_times.append(groq_ms)
        total_times.append(total_ms)
        modes.append(gen.response_mode or "unknown")

    mode_counts = {}
    for m in modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1

    return {
        "mode": "groq_generation",
        "label": (
            "Embed + FAISS/BM25 hybrid retrieve + RRF + FlashRank + Groq llama-3.1-8b-instant. "
            "groq_generation_ms is a NETWORK measurement to Groq US. "
            "Measure from deployed Render environment for production-representative values."
        ),
        "embed_ms": _percentiles(embed_times),
        "retrieve_ms": _percentiles(retrieve_times),
        "groq_generation_ms": _percentiles(groq_times),
        "total_rag_core_ms": _percentiles(total_times),
        "response_mode_counts": mode_counts,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    n_queries = int(os.environ.get("BENCH_QUERIES", "10"))
    warmup = int(os.environ.get("BENCH_WARMUP", "2"))
    query_file = os.environ.get("BENCH_QUERY_FILE", "")

    if query_file and Path(query_file).exists():
        with open(query_file, encoding="utf-8") as f:
            queries = [json.loads(line)["query"] for line in f if line.strip()][:n_queries]
    else:
        queries = DEFAULT_QUERIES[:n_queries]

    print(f"\n{'='*70}")
    print("  HHGOA Generation Mode Latency Benchmark")
    print(f"  Queries: {len(queries)}  |  Warmup: {warmup}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # Mode A: Extractive (no LLM)
    # ------------------------------------------------------------------
    print("▶  Mode A: local_core_retrieval_extractive  (no LLM)")
    print("   This is the ~38ms path. In-process only.\n")
    a = bench_retrieveal_and_extractive(queries, warmup=warmup)
    print(f"   Embed      P50={a['embed_ms']['p50']}ms  P70={a['embed_ms']['p70']}ms  P100={a['embed_ms']['p100']}ms")
    print(f"   Retrieve   P50={a['retrieve_ms']['p50']}ms  P70={a['retrieve_ms']['p70']}ms  P100={a['retrieve_ms']['p100']}ms")
    print(f"   Extractive P50={a['extractive_ms']['p50']}ms  P70={a['extractive_ms']['p70']}ms  P100={a['extractive_ms']['p100']}ms")
    print(f"   ─────────────────────────────────────────────────────────")
    print(f"   TOTAL      P50={a['total_ms']['p50']}ms  P70={a['total_ms']['p70']}ms  P100={a['total_ms']['p100']}ms")
    print(f"\n   ⚠  NOTE: ~{a['total_ms']['p50']}ms is local in-process (index in RAM).")
    print(f"       Does NOT include STT, network, or response transmission.\n")

    # ------------------------------------------------------------------
    # Mode B: Groq
    # ------------------------------------------------------------------
    print("▶  Mode B: groq_generation  (Groq llama-3.1-8b-instant)\n")
    b = bench_retrieval_and_groq(queries, warmup=warmup)

    if "error" in b:
        print(f"   ⚠  {b['error']}\n")
    else:
        print(f"   Embed      P50={b['embed_ms']['p50']}ms  P70={b['embed_ms']['p70']}ms  P100={b['embed_ms']['p100']}ms")
        print(f"   Retrieve   P50={b['retrieve_ms']['p50']}ms  P70={b['retrieve_ms']['p70']}ms  P100={b['retrieve_ms']['p100']}ms")
        print(f"   Groq call  P50={b['groq_generation_ms']['p50']}ms  P70={b['groq_generation_ms']['p70']}ms  P100={b['groq_generation_ms']['p100']}ms")
        print(f"   ─────────────────────────────────────────────────────────")
        print(f"   TOTAL      P50={b['total_rag_core_ms']['p50']}ms  P70={b['total_rag_core_ms']['p70']}ms  P100={b['total_rag_core_ms']['p100']}ms")
        print(f"\n   Response modes: {b['response_mode_counts']}")
        print(f"\n   ⚠  NOTE: Groq generation time ({b['groq_generation_ms']['p50']}ms P50) is a NETWORK measurement.")
        print(f"       Measured from THIS machine. Render deployment latency will differ.")
        print(f"       Re-run from Render for production-representative numbers.\n")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_path = Path("bench/bench_generation_modes_results.json")
    out_path.write_text(json.dumps({"mode_a": a, "mode_b": b}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved → {out_path}\n")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
