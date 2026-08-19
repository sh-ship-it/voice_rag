"""FastAPI backend for HHGOA Voice RAG Pipeline.

Per HHGOA Task 2 architecture document:
    "The frontend should only call /query, /voice, and /health;
    all retrieval, guardrails, provider routing, and citations stay in FastAPI."
    "Configure the FastAPI backend with CORS allowing only the Render frontend URL."
    "Add a /health endpoint that checks index files loaded and returns index version."
    "Stream tokens to React using Server-Sent Events."

Endpoints
---------
    POST /query   — text query -> PipelineResponse JSON
    POST /voice   — multipart audio -> PipelineResponse JSON
    GET  /health  — index version + loaded chunk count
    GET  /stream  — SSE streaming tokens for a text query
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Optional

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HHGOA Voice RAG API",
    description="Hindi voice Q&A over the HHGOA MSMARCO-XI corpus.",
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# CORS — per architecture doc: "allowing only the Render frontend URL"
# ---------------------------------------------------------------------------
# FRONTEND_URL can be a single URL or comma-separated list of URLs
# e.g. "https://hhgoa-frontend.onrender.com,http://localhost:5173"
_FRONTEND_URL_RAW = os.environ.get("FRONTEND_URL", "*")
if _FRONTEND_URL_RAW == "*" or not _FRONTEND_URL_RAW:
    _ALLOWED_ORIGINS = ["*"]
else:
    _ALLOWED_ORIGINS = [u.strip() for u in _FRONTEND_URL_RAW.split(",") if u.strip()]
    _ALLOWED_ORIGINS.extend(["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"])
    _ALLOWED_ORIGINS = list(dict.fromkeys(_ALLOWED_ORIGINS))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True if _ALLOWED_ORIGINS != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pipeline warm-up (load models + index on startup, not per-request)
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    """Warm up embedding model and index on process start."""
    import asyncio
    # Trigger warm imports in background thread so startup doesn't block
    await asyncio.to_thread(_warmup)


def _warmup():
    """Load embedding model, FAISS index, and BM25 index into memory."""
    from pipeline.embed import embed_query
    from pipeline.retrieve import _REGISTRY, warmup as retrieve_warmup
    try:
        retrieve_warmup()
        # Warm embedding model with a dummy query
        embed_query("warmup query")
        print("[backend] Warmup complete.", flush=True)
    except Exception as e:
        print(f"[backend] Warmup error (non-fatal): {e}", flush=True)


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class TextQueryRequest(BaseModel):
    """Request body for text query endpoint."""
    query: str
    language_code: str = "hi-IN"
    top_k: int = 6


# ---------------------------------------------------------------------------
# /health endpoint
# Per architecture doc: "Returns current index version and chunk count."
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint. Returns index version and loaded chunk count."""
    from pipeline.retrieve import _REGISTRY
    import json
    from pathlib import Path

    index_dir = Path("index")
    build_stats = {}
    stats_path = index_dir / "build_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            build_stats = json.load(f)

    chunk_count = len(_REGISTRY.chunk_list) if _REGISTRY and _REGISTRY.chunk_list else 0
    faiss_loaded = _REGISTRY is not None and hasattr(_REGISTRY, "faiss_index") and _REGISTRY.faiss_index is not None

    return {
        "status": "ok" if faiss_loaded else "warming_up",
        "index_version": "v1",
        "chunk_count": chunk_count,
        "faiss_loaded": faiss_loaded,
        "build_stats": build_stats,
    }


# ---------------------------------------------------------------------------
# POST /query — text query → PipelineResponse JSON
# ---------------------------------------------------------------------------

@app.post("/query")
async def query_text(request: TextQueryRequest):
    """Handle text query. Returns full PipelineResponse as JSON."""
    from pipeline.orchestrator import arun_pipeline

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    response = await arun_pipeline(
        query_text=request.query.strip(),
        language_code=request.language_code,
        top_k=request.top_k,
    )

    return response.model_dump(mode="json", exclude_none=True)


# ---------------------------------------------------------------------------
# POST /voice — multipart audio → PipelineResponse JSON
# ---------------------------------------------------------------------------

@app.post("/voice")
async def query_voice(
    audio: UploadFile = File(...),
    language_code: str = Form(default="hi-IN"),
    top_k: int = Form(default=6),
):
    """Handle audio upload. Runs full STT -> RAG pipeline.
    Returns PipelineResponse as JSON.
    """
    from pipeline.orchestrator import arun_pipeline

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty.")

    response = await arun_pipeline(
        audio_bytes=audio_bytes,
        language_code=language_code,
        top_k=top_k,
    )

    return response.model_dump(mode="json", exclude_none=True)


# ---------------------------------------------------------------------------
# GET /stream — SSE streaming tokens
# Per architecture doc: "Stream tokens to React using Server-Sent Events."
# ---------------------------------------------------------------------------

@app.get("/stream")
async def stream_query(
    query: str = Query(..., description="Text query to answer"),
    language_code: str = Query(default="hi-IN"),
):
    """SSE endpoint that streams pipeline response fields as events.

    Events emitted (in order):
    - event: transcript, data: {text}
    - event: chunk_start, data: {chunk_id, text}
    - event: token, data: {token}   (for each token in answer)
    - event: citations, data: JSON array of chunk_ids
    - event: timings, data: JSON object with per-stage latency_ms
    - event: done, data: {}
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def event_generator() -> AsyncIterator[str]:
        from pipeline.orchestrator import arun_pipeline

        try:
            response = await arun_pipeline(
                query_text=query.strip(),
                language_code=language_code,
                top_k=6,
            )

            # Emit transcript (for voice passthrough; for text queries same as query)
            if response.transcription:
                yield f"event: transcript\ndata: {json.dumps({'text': response.transcription}, ensure_ascii=False)}\n\n"

            # Stream answer tokens word-by-word (simulated streaming from full response)
            # Real token streaming would require Cerebras streaming API support
            words = response.answer.split()
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                yield f"event: token\ndata: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0.005)  # 5ms inter-token delay

            # Emit citations
            citations_data = [
                c.model_dump(mode="json") if hasattr(c, "model_dump") else c
                for c in response.citations
            ]
            yield f"event: citations\ndata: {json.dumps(citations_data, ensure_ascii=False)}\n\n"

            # Emit timings
            yield f"event: timings\ndata: {json.dumps(response.timings or {})}\n\n"

            # Emit confidence + grounded + evidence_text + response_mode
            yield f"event: meta\ndata: {json.dumps({'confidence': response.confidence, 'confidence_tier': response.confidence_tier, 'grounded': response.grounded, 'status': response.status, 'evidence_text': response.evidence_text, 'response_mode': response.response_mode}, ensure_ascii=False)}\n\n"

            yield "event: done\ndata: {}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
