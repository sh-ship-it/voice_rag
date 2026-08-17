"""Retriever adapter module bridging benchmark.py to the pipeline's hybrid retrieval engine."""

import time
from dataclasses import dataclass
from typing import Any, List

from pipeline.embed import embed_query, get_embedding_model
from pipeline.retrieve import get_index_registry, hybrid_retrieve
from pipeline.schemas import ScoredChunk


@dataclass
class SearchResponse:
    """Benchmark search response container."""
    query: str
    chunks: List[ScoredChunk]
    embed_ms: float
    search_ms: float
    total_ms: float


def warmup():
    """Warm up embedding model and index search structures."""
    # Warm up embedding model
    get_embedding_model()
    dummy_emb = embed_query("warmup query")

    # Warm up retrieval
    reg = get_index_registry()
    hybrid_retrieve(dummy_emb, "warmup query", top_k=5, registry=reg)


def search(query: str, top_k: int = 5) -> SearchResponse:
    """Execute timed retrieval: embedding + FAISS search."""
    t_start = time.perf_counter()

    # Stage 1: Dense embedding
    t_embed_start = time.perf_counter()
    query_emb = embed_query(query)
    embed_ms = (time.perf_counter() - t_embed_start) * 1000.0

    # Stage 2: Hybrid FAISS + BM25 search
    t_search_start = time.perf_counter()
    reg = get_index_registry()
    ret_res = hybrid_retrieve(query_emb, query, top_k=top_k, registry=reg)
    search_ms = (time.perf_counter() - t_search_start) * 1000.0

    total_ms = (time.perf_counter() - t_start) * 1000.0

    return SearchResponse(
        query=query,
        chunks=ret_res.chunks,
        embed_ms=embed_ms,
        search_ms=search_ms,
        total_ms=total_ms,
    )
