"""Latency benchmarking harness for Voice RAG pipeline components."""

import statistics
import time
from typing import Any, Callable, Dict, List, Optional
from pipeline.config import get_settings
from pipeline.orchestrator import VoiceRAGOrchestrator
from pipeline.schemas import AudioInput


class LatencyBenchmark:
    """Measures and reports latency metrics (p50, p90, p95, p99) for pipeline components."""

    def __init__(self, orchestrator: Optional[VoiceRAGOrchestrator] = None) -> None:
        self.settings = get_settings()
        self.orchestrator = orchestrator or VoiceRAGOrchestrator()

    @staticmethod
    def calculate_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
        """Calculate summary statistics and percentiles for latency samples."""
        if not latencies_ms:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}

        sorted_lat = sorted(latencies_ms)
        n = len(sorted_lat)

        def percentile(p: float) -> float:
            idx = int(p * n)
            return sorted_lat[min(idx, n - 1)]

        return {
            "count": float(n),
            "mean": round(statistics.mean(latencies_ms), 2),
            "std": round(statistics.stdev(latencies_ms), 2) if n > 1 else 0.0,
            "min": round(min(latencies_ms), 2),
            "p50": round(percentile(0.50), 2),
            "p90": round(percentile(0.90), 2),
            "p95": round(percentile(0.95), 2),
            "p99": round(percentile(0.99), 2),
            "max": round(max(latencies_ms), 2),
        }

    def benchmark_stage(
        self,
        name: str,
        func: Callable[[], Any],
        iterations: int = 10,
        warmup: int = 2
    ) -> Dict[str, Any]:
        """Benchmark a callable pipeline stage."""
        for _ in range(warmup):
            func()

        latencies: List[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            func()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

        stats = self.calculate_percentiles(latencies)
        stats["stage_name"] = name
        return stats

    def benchmark_text_pipeline(
        self,
        sample_query: str = "What are the latest updates in LLM inference?",
        iterations: int = 5
    ) -> Dict[str, Any]:
        """Run latency benchmark on text RAG."""
        return self.benchmark_stage(
            name="text_rag_pipeline",
            func=lambda: self.orchestrator.run_text_rag(sample_query),
            iterations=iterations
        )

    def benchmark_voice_pipeline(
        self,
        sample_audio: Optional[AudioInput] = None,
        iterations: int = 5
    ) -> Dict[str, Any]:
        """Run latency benchmark on voice RAG."""
        audio = sample_audio or AudioInput(
            audio_bytes=b"\x00" * 32000,
            content_type="audio/wav",
            sample_rate=16000
        )
        return self.benchmark_stage(
            name="voice_rag_pipeline",
            func=lambda: self.orchestrator.run_voice_rag(audio),
            iterations=iterations
        )


def run_benchmark() -> None:
    """CLI entry point for running latency benchmarks."""
    print("=== Starting Voice RAG Latency Benchmark ===")
    bench = LatencyBenchmark()
    text_results = bench.benchmark_text_pipeline(iterations=5)
    print(f"Text RAG Results: {text_results}")
    voice_results = bench.benchmark_voice_pipeline(iterations=5)
    print(f"Voice RAG Results: {voice_results}")
    print("=== Benchmark Finished ===")


if __name__ == "__main__":
    run_benchmark()
