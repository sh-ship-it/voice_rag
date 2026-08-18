---
title: Voice RAG FastAPI Backend
emoji: 🎙️
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Voice-Enabled RAG System

A modular, low-latency, voice-enabled Retrieval-Augmented Generation (RAG) system built with Python 3.11.

The architecture integrates:
- **Speech-to-Text (STT)**: Sarvam AI for multilingual and Indian English speech recognition.
- **Retrieval Engine**: Hybrid retrieval combining **FAISS** (dense vector search with sentence-transformers) and **BM25** (sparse lexical search) fused via Reciprocal Rank Fusion (RRF).
- **Fast Generation**: Cerebras Inference API (Llama 3.1) for ultra-low time-to-first-token generation.
- **Safety Guardrails**: Input/output moderation and prompt injection detection.
- **FastAPI UI**: Backend server paired with a responsive, single-page vanilla HTML/JS interface featuring voice recording and live latency breakdown cards.
- **Telemetry & Benchmarking**: Per-component latency measurements (p50, p90, p95, p99).

---

## Directory Structure

```
.
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore rules for data, indexes, and caches
├── requirements.txt          # Pinned project dependencies
├── pyproject.toml            # Python packaging and pytest configuration
├── README.md                 # Project documentation
│
├── data/                     # Downloaded & cached dataset artifacts (gitignored)
│   └── .gitkeep
├── index/                    # Persisted FAISS index + BM25 index + chunk metadata (gitignored)
│   └── .gitkeep
│
├── pipeline/                 # Core pipeline modules
│   ├── __init__.py           # Package exports
│   ├── config.py             # Settings loading via pydantic-settings & python-dotenv
│   ├── schemas.py            # Pydantic models (Chunk, RetrievalResult, GenerationResult, PipelineResponse)
│   ├── chunking.py           # Document chunking strategies & skeletons
│   ├── embed.py              # Sentence-transformers embedding generator skeleton
│   ├── retrieve.py           # FAISS dense + BM25 sparse hybrid retriever skeleton
│   ├── guardrails.py         # Input/output safety & prompt injection guardrails
│   ├── generate.py           # Cerebras LLM generation skeleton
│   ├── stt.py                # Sarvam STT speech transcription skeleton
│   └── orchestrator.py       # End-to-end voice and text RAG orchestrator
│
├── bench/                    # Latency benchmarking
│   ├── __init__.py
│   └── bench_latency.py      # Latency percentile benchmarking suite
│
├── ui/                       # FastAPI backend & single static page UI
│   ├── __init__.py
│   ├── app.py                # FastAPI application with audio & text endpoints
│   └── static/
│       ├── index.html        # Single-page voice & text RAG interface
│       ├── style.css         # Dark glassmorphic styling
│       └── app.js            # Audio recorder, API client, & telemetry visualizer
│
└── tests/                    # Test suite
    ├── __init__.py
    └── test_smoke.py         # Smoke tests for imports, schemas, config, and routes
```

---

## Installation & Setup

### 1. Prerequisites
- Python **3.11+**
- Virtual environment (`venv` or `uv`)

### 2. Create and Activate Virtual Environment

```bash
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# On Linux / macOS:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file and populate your API keys:

```bash
cp .env.example .env
```

Edit `.env` with your API keys:
- `SARVAM_API_KEY`: API key for Sarvam STT.
- `CEREBRAS_API_KEY`: API key for Cerebras inference.
- `HF_TOKEN`: Hugging Face access token for models/datasets.

---

## Pydantic Data Models

All data passed between pipeline stages is strictly validated via Pydantic models (`pipeline/schemas.py`):

| Model | Purpose |
| :--- | :--- |
| `Chunk` | Discrete text chunk with doc ID, metadata (`ChunkMetadata`), and optional vector embedding. |
| `ScoredChunk` | Retrieved chunk with relevance score, rank, and strategy (`DENSE`, `SPARSE`, `HYBRID`). |
| `RetrievalResult` | Output from retrieval containing ranked chunks, candidate counts, and latency. |
| `GenerationResult` | Output from LLM containing generated text, token usage, and latency. |
| `GuardrailResult` | Validation status, flagged categories, and sanitized query/response. |
| `AudioInput` | Container for audio bytes/base64, sample rate, and content type. |
| `LatencyBreakdown` | Millisecond-level latencies across STT, guardrails, retrieval, and generation. |
| `PipelineResponse` | Unified top-level response payload with query, transcription, answer, context, and latency metrics. |

---

## Running the Application

### 1. Start the FastAPI Server

```bash
uvicorn ui.app:app --host 0.0.0.0 --port 8000 --reload
```

Once running, navigate to:
- **Web UI**: `http://localhost:8000/`
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/api/health`

### 2. Run Latency Benchmarks

```bash
python -m bench.bench_latency
```

### 3. Run Smoke Tests

```bash
pytest tests/test_smoke.py -v
```
