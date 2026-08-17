"""App configuration bridge for benchmarking and retrieval."""

from pipeline.config import get_settings

settings = get_settings()

# 50ms retrieval latency budget (embed + search)
LATENCY_BUDGET_MS = 50.0
