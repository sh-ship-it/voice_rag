"""FastAPI backend application for Voice-enabled RAG."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline.config import get_settings
from pipeline.orchestrator import VoiceRAGOrchestrator, arun_pipeline
from pipeline.schemas import AudioInput, PipelineResponse

app = FastAPI(
    title="Voice RAG API",
    description="Ultra low-latency (<200ms) voice-enabled Retrieval-Augmented Generation API",
    version="1.0.0"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()
orchestrator = VoiceRAGOrchestrator()

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class TextQueryRequest(BaseModel):
    """Payload for text query requests."""
    query: str = Field(..., min_length=1, description="User question or search query")
    language_code: Optional[str] = Field(default="hi-IN", description="Language code")


@app.get("/")
async def serve_index():
    """Serve the single-page voice RAG web interface."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Voice RAG Backend running. Static UI available at /static/index.html"}


@app.get("/api/health")
async def health_check():
    """System health check and configuration status."""
    from pipeline.retrieve import get_index_registry
    registry = get_index_registry()

    return {
        "status": "healthy",
        "sarvam_configured": bool(settings.sarvam_api_key),
        "cerebras_configured": bool(settings.cerebras_api_key),
        "hf_configured": bool(settings.hf_token),
        "index": {
            "is_loaded": registry.is_loaded,
            "total_chunks": len(registry.chunk_list),
        },
        "models": {
            "embedding": settings.embedding_model_name,
            "generator": settings.cerebras_model,
            "stt_model": "saaras:v3",
            "stt_language": settings.sarvam_language_code,
        },
    }


@app.post("/api/query/text", response_model=PipelineResponse)
async def query_text(request: TextQueryRequest) -> PipelineResponse:
    """Process a text-based RAG query."""
    try:
        response = await arun_pipeline(
            query_text=request.query,
            language_code=request.language_code or "hi-IN",
        )
        return response
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}") from exc


@app.post("/api/query/voice", response_model=PipelineResponse)
@app.post("/api/query/audio", response_model=PipelineResponse)
async def query_voice(
    file: UploadFile = File(...),
    language_code: Optional[str] = Form("hi-IN")
) -> PipelineResponse:
    """Process a voice/audio RAG query."""
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio payload received.")

        response = await arun_pipeline(
            audio_bytes=audio_bytes,
            language_code=language_code or "hi-IN",
        )
        return response
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Voice pipeline error: {str(exc)}") from exc


@app.get("/api/benchmark/summary")
async def get_benchmark_summary() -> Dict[str, Any]:
    """Retrieve latest benchmark summary stats from bench/results.jsonl."""
    results_path = Path("bench/results.jsonl")
    if not results_path.exists():
        return {"status": "no_results", "message": "No benchmark results found. Run python bench/run_benchmark.py first."}

    records = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    if not records:
        return {"status": "empty", "message": "Results file is empty."}

    from bench.run_benchmark import calculate_percentiles

    # Filter warm records
    warm_records = [r for r in records if not r.get("is_cold_start", False)]
    core_latencies = [r.get("total_rag_core_ms", 0.0) for r in warm_records]
    stt_latencies = [r.get("stt_ms", 0.0) for r in warm_records]
    total_latencies = [r.get("timings", {}).get("total", 0.0) for r in warm_records]

    return {
        "status": "ready",
        "total_queries": len(records),
        "warm_queries": len(warm_records),
        "core_rag_stats": calculate_percentiles(core_latencies),
        "stt_stats": calculate_percentiles(stt_latencies),
        "total_e2e_stats": calculate_percentiles(total_latencies),
        "guardrail_accuracy": sum(1 for r in warm_records if r.get("query_type") == "off_topic" and r.get("refusal")) / max(1, sum(1 for r in warm_records if r.get("query_type") == "off_topic")),
    }


def start_server():
    """Entry point for running uvicorn server directly."""
    import uvicorn
    uvicorn.run(
        "ui.app:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True
    )


if __name__ == "__main__":
    start_server()
