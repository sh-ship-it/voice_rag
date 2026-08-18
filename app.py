"""Hugging Face Spaces Entry Point for FastAPI Voice RAG Backend."""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
