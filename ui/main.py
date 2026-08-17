"""Minimal FastAPI backend for Voice RAG demo aid (POST /query and GET /)."""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from pipeline.config import get_settings
from pipeline.orchestrator import arun_pipeline, run_pipeline
from pipeline.schemas import PipelineResponse

app = FastAPI(
    title="Voice RAG Demo",
    description="Minimal FastAPI demo backend serving index.html and POST /query",
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
HTML_FILE = Path(__file__).parent / "index.html"


@app.get("/")
async def serve_ui():
    """Serve single static index.html at root."""
    if HTML_FILE.exists():
        return FileResponse(str(HTML_FILE))
    return JSONResponse(
        {"error": "ui/index.html not found"},
        status_code=404
    )


@app.post("/query", response_model=PipelineResponse)
async def process_query(
    file: Optional[UploadFile] = File(None),
    query_text: Optional[str] = Form(None),
    language_code: Optional[str] = Form("hi-IN"),
) -> PipelineResponse:
    """Accept multipart audio file upload (or text fallback) and return PipelineResponse."""
    try:
        audio_bytes: Optional[bytes] = None
        if file is not None:
            audio_bytes = await file.read()

        # Run pipeline
        response = await arun_pipeline(
            audio_bytes=audio_bytes,
            language_code=language_code or "hi-IN",
            query_text=query_text,
        )
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution error: {str(exc)}"
        ) from exc


def main():
    """Run uvicorn server directly."""
    uvicorn.run(
        "ui.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True
    )


if __name__ == "__main__":
    main()
