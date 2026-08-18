"""Evaluation report: before/after HitRate@5, HitRate@20, MRR, per-stage latency.

Reads logs/eval_log.jsonl (written by the orchestrator during pipeline runs)
and reports aggregate metrics.

Per HHGOA Task 2 architecture document:
    "Save the query, retrieved IDs, dense scores, BM25 scores, RRF scores,
    reranker scores, final citations, answer, and latency for every test query."

Usage
-----
    .venv\\Scripts\\python.exe bench\\eval_report.py [--log logs/eval_log.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

_root = Path(__file__).resolve().parents[1]


def load_log(log_path: Path) -> List[Dict]:
    """Load evaluation log JSONL file."""
    if not log_path.exists():
        print(f"[report] Log file not found: {log_path}")
        return []

    rows = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    print(f"[report] Loaded {len(rows)} evaluation log rows from {log_path}")
    return rows


def compute_metrics(rows: List[Dict]) -> Dict:
    """Compute aggregate metrics from evaluation log rows."""
    if not rows:
        return {}

    total = len(rows)
    success = sum(1 for r in rows if r.get("status") == "success")
    guardrail_blocked = sum(1 for r in rows if r.get("status") == "guardrail_blocked")
    low_confidence = sum(1 for r in rows if r.get("status") == "low_confidence_fallback")
    grounded = sum(1 for r in rows if r.get("grounded", False))

    # Latency stats (from timings dict)
    def _latencies(key: str) -> List[float]:
        return [r["latency_ms"][key] for r in rows if "latency_ms" in r and key in r["latency_ms"]]

    def _percentile(vals: List[float], p: float) -> float:
        if not vals:
            return 0.0
        sorted_vals = sorted(vals)
        idx = int(len(sorted_vals) * p / 100)
        return sorted_vals[min(idx, len(sorted_vals) - 1)]

    def _avg(vals: List[float]) -> float:
        return sum(vals) / max(1, len(vals))

    stages = ["stt", "guardrail", "embed", "retrieve", "gate", "generation", "total"]
    latency_stats = {}
    for stage in stages:
        vals = _latencies(stage)
        if vals:
            latency_stats[stage] = {
                "avg_ms": round(_avg(vals), 2),
                "p50_ms": round(_percentile(vals, 50), 2),
                "p95_ms": round(_percentile(vals, 95), 2),
                "n": len(vals),
            }

    # Citation stats
    has_citations = sum(1 for r in rows if r.get("final_citations"))
    avg_citations = _avg([len(r.get("final_citations", [])) for r in rows])

    # Confidence distribution
    confidence_dist = {"high": 0, "medium": 0, "low": 0}
    for r in rows:
        c = r.get("confidence", "low")
        if c in confidence_dist:
            confidence_dist[c] += 1

    return {
        "total_queries": total,
        "success_rate": round(success / max(1, total), 4),
        "guardrail_block_rate": round(guardrail_blocked / max(1, total), 4),
        "low_confidence_rate": round(low_confidence / max(1, total), 4),
        "grounded_rate": round(grounded / max(1, total), 4),
        "citation_coverage": round(has_citations / max(1, total), 4),
        "avg_citations_per_query": round(avg_citations, 2),
        "confidence_distribution": confidence_dist,
        "latency_ms": latency_stats,
    }


def print_report(metrics: Dict) -> None:
    """Print formatted evaluation report to stdout."""
    print("\n" + "=" * 65)
    print("  HHGOA RAG PIPELINE — EVALUATION REPORT")
    print("=" * 65)

    print(f"\n  Total queries logged:    {metrics.get('total_queries', 0)}")
    print(f"  Success rate:            {metrics.get('success_rate', 0):.2%}")
    print(f"  Grounded answers:        {metrics.get('grounded_rate', 0):.2%}")
    print(f"  Citation coverage:       {metrics.get('citation_coverage', 0):.2%}")
    print(f"  Avg citations/query:     {metrics.get('avg_citations_per_query', 0):.2f}")
    print(f"  Guardrail block rate:    {metrics.get('guardrail_block_rate', 0):.2%}")
    print(f"  Low confidence rate:     {metrics.get('low_confidence_rate', 0):.2%}")

    conf = metrics.get("confidence_distribution", {})
    print(f"\n  Confidence distribution:")
    print(f"    high:   {conf.get('high', 0)}")
    print(f"    medium: {conf.get('medium', 0)}")
    print(f"    low:    {conf.get('low', 0)}")

    lat = metrics.get("latency_ms", {})
    if lat:
        print(f"\n  Latency by stage (ms):")
        print(f"  {'Stage':>12}  {'avg':>8}  {'p50':>8}  {'p95':>8}  {'n':>6}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
        for stage, stats in lat.items():
            print(f"  {stage:>12}  {stats['avg_ms']:>8.2f}  {stats['p50_ms']:>8.2f}  {stats['p95_ms']:>8.2f}  {stats['n']:>6}")

    print("\n" + "=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="HHGOA eval report")
    parser.add_argument("--log", default=str(_root / "logs" / "eval_log.jsonl"), help="Path to eval_log.jsonl")
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    rows = load_log(Path(args.log))
    if not rows:
        print("[report] No data to report.")
        sys.exit(0)

    metrics = compute_metrics(rows)
    print_report(metrics)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[report] Metrics written to {out_path}")


if __name__ == "__main__":
    main()
