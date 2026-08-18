"""Hugging Face Spaces Entry Point for FastAPI Voice RAG Backend."""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr
from backend.main import app as fastapi_app

# Lightweight UI for root path on Hugging Face Spaces
with gr.Blocks(title="Voice RAG FastAPI Backend", theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🎙️ Voice-Enabled Indic Pure Extractive RAG API")
    gr.Markdown("✅ **Backend is Active & Running with 16 GB RAM!**")
    gr.Markdown("### Available REST Endpoints:")
    gr.Markdown("""
    - `POST /query` — Fast text retrieval & extraction
    - `POST /voice` — Audio speech-to-text + retrieval
    - `GET /stream` — Server-Sent Events (SSE) streaming
    - `GET /health` — System status & index metrics
    - `GET /docs` — Interactive Swagger API documentation
    """)

# Mount FastAPI app so both Gradio and all FastAPI endpoints work seamlessly
app = gr.mount_gradio_app(fastapi_app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
