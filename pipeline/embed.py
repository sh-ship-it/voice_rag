"""Embedding generation module using sentence-transformers (multilingual-e5-small).

Loads the model once at module import so it remains warm in memory for low-latency
voice query encoding.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from pipeline.config import get_settings
from pipeline.schemas import Chunk

# ---------------------------------------------------------------------------
# Global warm model initialization
# ---------------------------------------------------------------------------

import os
import torch

# Optimize PyTorch CPU inference threads
torch.set_num_threads(min(8, os.cpu_count() or 4))

_SETTINGS = get_settings()
_MODEL_NAME = _SETTINGS.embedding_model_name  # intfloat/multilingual-e5-small

# E5 instruction prefixes
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[pipeline.embed] Loading warm embedding model: {_MODEL_NAME!r} on device '{_DEVICE}' ...", flush=True)
_WARM_MODEL: SentenceTransformer = SentenceTransformer(_MODEL_NAME, device=_DEVICE)
_WARM_MODEL.max_seq_length = 128  # Per Task 2 arch doc: benchmark 32/128/256; 128 chosen pending bench/benchmark_seq_lengths.py results
if _DEVICE == "cuda":
    try:
        _WARM_MODEL = _WARM_MODEL.half()
    except Exception:
        pass
_EMBEDDING_DIM: int = _WARM_MODEL.get_sentence_embedding_dimension()
print(f"[pipeline.embed] Model {_MODEL_NAME!r} loaded on {_DEVICE} successfully (dim={_EMBEDDING_DIM}).", flush=True)


def get_embedding_model() -> SentenceTransformer:
    """Return the global pre-warmed SentenceTransformer model instance."""
    return _WARM_MODEL


def embed_query(
    text: str,
    language: str = "hi",
    model: Optional[SentenceTransformer] = None,
) -> np.ndarray:
    """Encode a single query string into an L2-normalized 1D float32 vector."""
    m = model or _WARM_MODEL
    clean_text = text.strip()
    prefixed = f"{QUERY_PREFIX}{clean_text}"
    with torch.inference_mode():
        vec = m.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    return np.ascontiguousarray(vec, dtype=np.float32).flatten()


def embed_passages(
    texts: List[str],
    batch_size: int = 256,
    model: Optional[SentenceTransformer] = None,
) -> np.ndarray:
    """Encode a list of passage texts with the ``"passage: "`` prefix."""
    m = model or _WARM_MODEL
    prefixed = [f"{PASSAGE_PREFIX}{t.strip()}" for t in texts]
    vecs = m.encode(
        prefixed,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.array(vecs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Class-based Embedder interface
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Abstract base class for vector embedding generators."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate vector embedding for a single text string."""
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of text strings."""
        raise NotImplementedError

    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Attach vector embeddings to a list of Chunk models."""
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence-transformer embedding generator wrapper."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.settings = get_settings()
        self.model_name = model_name or self.settings.embedding_model_name
        self._model = _WARM_MODEL if self.model_name == _MODEL_NAME else SentenceTransformer(self.model_name)

    def embed_text(self, text: str) -> List[float]:
        """Generate normalized query embedding as list of floats."""
        vec = embed_query(text, model=self._model)
        return vec.tolist()

    def embed_batch(self, texts: List[str], prefix: str = PASSAGE_PREFIX) -> List[List[float]]:
        """Generate batch embeddings as list of float lists."""
        prefixed = [f"{prefix}{t.strip()}" for t in texts]
        vecs = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=128,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Populate the embedding field on a list of chunks."""
        texts = [c.text for c in chunks]
        embeddings = self.embed_batch(texts, prefix=PASSAGE_PREFIX)
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        return chunks
