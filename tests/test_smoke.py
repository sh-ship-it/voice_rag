"""Smoke tests verifying module imports, config loading, Pydantic schemas, and FastAPI routes."""

import asyncio
import importlib
import pytest
from pydantic import ValidationError

# 1. Test Module Imports
REQUIRED_MODULES = [
    "pipeline",
    "pipeline.config",
    "pipeline.schemas",
    "pipeline.chunking",
    "pipeline.embed",
    "pipeline.retrieve",
    "pipeline.guardrails",
    "pipeline.generate",
    "pipeline.stt",
    "pipeline.orchestrator",
    "bench",
    "bench.bench_latency",
    "ui",
    "ui.app",
]


@pytest.mark.parametrize("module_name", REQUIRED_MODULES)
def test_module_imports(module_name: str):
    """Verify all project modules import cleanly without syntax or dependency errors."""
    mod = importlib.import_module(module_name)
    assert mod is not None


# 2. Test Pydantic Schemas
def test_pydantic_schemas():
    """Verify all core Pydantic data schemas instantiate and serialize correctly."""
    from pipeline.schemas import (
        AudioInput,
        Chunk,
        ChunkMetadata,
        GenerationResult,
        GuardrailResult,
        LatencyBreakdown,
        PipelineResponse,
        RetrievalResult,
        RetrievalStrategy,
        ScoredChunk,
    )

    # Chunk & Metadata
    meta = ChunkMetadata(source="doc1.pdf", section="Intro", token_count=100)
    chunk = Chunk(
        chunk_id="chunk_1",
        doc_id="doc_1",
        text="Sample context chunk.",
        metadata=meta,
        embedding=[0.1, 0.2, 0.3]
    )
    assert chunk.chunk_id == "chunk_1"
    assert chunk.metadata.source == "doc1.pdf"

    # ScoredChunk & RetrievalResult
    scored_chunk = ScoredChunk(
        chunk=chunk,
        score=0.89,
        rank=1,
        retrieval_strategy=RetrievalStrategy.HYBRID
    )
    retrieval_res = RetrievalResult(
        query="What is RAG?",
        chunks=[scored_chunk],
        strategy_used=RetrievalStrategy.HYBRID,
        total_candidates_evaluated=10,
        latency_ms=12.5
    )
    assert len(retrieval_res.chunks) == 1
    assert retrieval_res.chunks[0].score == 0.89

    # GenerationResult
    gen_res = GenerationResult(
        generated_text="RAG is Retrieval-Augmented Generation.",
        model_name="llama3.1-8b",
        prompt_tokens=50,
        completion_tokens=10,
        total_tokens=60,
        latency_ms=45.2
    )
    assert gen_res.total_tokens == 60

    # LatencyBreakdown
    latencies = LatencyBreakdown(
        stt_ms=120.0,
        retrieval_ms=12.5,
        generation_ms=45.2,
        total_pipeline_ms=177.7
    )
    assert latencies.stt_ms == 120.0

    # GuardrailResult
    guard = GuardrailResult(passed=True, sanitized_text="Clean input")
    assert guard.passed is True

    # AudioInput
    audio = AudioInput(
        audio_bytes=b"\x00\x01\x02",
        content_type="audio/wav",
        sample_rate=16000
    )
    assert audio.sample_rate == 16000

    # PipelineResponse
    pipeline_res = PipelineResponse(
        query="What is RAG?",
        transcription="What is RAG?",
        answer="RAG is Retrieval-Augmented Generation.",
        retrieval_result=retrieval_res,
        generation_result=gen_res,
        guardrails=guard,
        latency=latencies
    )
    assert pipeline_res.answer == "RAG is Retrieval-Augmented Generation."
    
    # Test JSON serialization / deserialization roundtrip
    json_data = pipeline_res.model_dump_json()
    reloaded = PipelineResponse.model_validate_json(json_data)
    assert reloaded.query == pipeline_res.query
    assert reloaded.retrieval_result.chunks[0].chunk.chunk_id == "chunk_1"


# 3. Test Config Loading
def test_config_loading(monkeypatch):
    """Verify Settings loads default configs and handles environment variable overrides."""
    from pipeline.config import Settings, get_settings

    # Verify default loading
    settings = get_settings()
    assert settings.embedding_model_name is not None
    assert settings.cerebras_model is not None

    # Verify override via environment variables
    monkeypatch.setenv("SARVAM_API_KEY", "test-sarvam-key")
    monkeypatch.setenv("CEREBRAS_API_KEY", "test-cerebras-key")
    monkeypatch.setenv("TOP_K_FINAL", "7")

    custom_settings = Settings()
    assert custom_settings.sarvam_api_key == "test-sarvam-key"
    assert custom_settings.cerebras_api_key == "test-cerebras-key"
    assert custom_settings.top_k_final == 7


# 4. Test Orchestrator Execution
def test_orchestrator_execution():
    """Verify end-to-end skeleton orchestrator execution in sync and async modes."""
    from pipeline.orchestrator import VoiceRAGOrchestrator
    from pipeline.schemas import AudioInput

    orchestrator = VoiceRAGOrchestrator()

    # Sync Text RAG
    text_res = orchestrator.run_text_rag("Test query")
    assert text_res.query == "Test query"
    assert text_res.answer is not None
    assert text_res.latency.total_pipeline_ms >= 0

    # Sync Voice RAG
    audio = AudioInput(audio_bytes=b"\x00" * 100)
    voice_res = orchestrator.run_voice_rag(audio)
    assert voice_res.transcription is not None
    assert voice_res.latency.stt_ms >= 0

    # Async Text RAG
    async def _test_async_text():
        res = await orchestrator.arun_text_rag("Async query")
        assert res.query == "Async query"
        assert res.answer is not None
        return res

    asyncio.run(_test_async_text())

    # Async Voice RAG
    async def _test_async_voice():
        res = await orchestrator.arun_voice_rag(audio)
        assert res.transcription is not None
        return res

    asyncio.run(_test_async_voice())


# 5. Test Latency Benchmark Module
def test_benchmark_module():
    """Verify latency benchmark calculations and stage execution."""
    from bench.bench_latency import LatencyBenchmark

    bench = LatencyBenchmark()
    stats = bench.calculate_percentiles([10.0, 20.0, 30.0, 40.0, 50.0])
    assert stats["count"] == 5.0
    assert stats["p50"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0

    res = bench.benchmark_text_pipeline(sample_query="Test benchmark", iterations=2)
    assert res["count"] == 2.0
    assert res["stage_name"] == "text_rag_pipeline"


# 6. Test FastAPI App Routes
def test_fastapi_routes():
    """Verify FastAPI routes are registered and respond."""
    from fastapi.testclient import TestClient
    from ui.app import app

    client = TestClient(app)

    # Root endpoint serving index.html
    root_res = client.get("/")
    assert root_res.status_code == 200

    # Health check
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "models" in data

    # Text query endpoint
    post_res = client.post("/api/query/text", json={"query": "Hello RAG"})
    assert post_res.status_code == 200
    res_data = post_res.json()
    assert "answer" in res_data
    assert "latency" in res_data
