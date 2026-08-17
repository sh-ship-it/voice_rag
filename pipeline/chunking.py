"""Three chunking strategies for the voice-enabled RAG pipeline.

Strategies
----------
1. fixed_size_chunker  -- sliding-window token chunker (baseline / control).
2. semantic_chunker    -- sentence-level embedding similarity boundary detection.
3. small_to_big_chunker -- sentence-level retrieval units carrying the full passage
                          as ``parent_text`` for context expansion at generation time.

All strategies accept a list of deduplicated passage strings and return
``List[Chunk]`` with ``source_passage_id``, ``language``, ``chunk_strategy``,
``token_count``, and ``parent_text`` populated.

None of the strategies call any external API or require a GPU at import time;
``semantic_chunker`` is the only one that loads a sentence-transformers model
(multilingual-e5-small), and only when it is called.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Sequence, Tuple

import numpy as np

from pipeline.schemas import Chunk, ChunkMetadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WHITESPACE_RE = re.compile(r"\s+")
# A sentence boundary: period / exclamation / question followed by whitespace
# or end-of-string, works reasonably for Hindi (Devanagari uses ।) and English.
_SENTENCE_RE = re.compile(r"(?<=[.!?।])\s+|(?<=[.!?।])$")


def _token_count(text: str) -> int:
    """Approximate token count using whitespace splitting."""
    return len(_WHITESPACE_RE.split(text.strip()))


def _make_chunk_id(passage_id: str, strategy: str, index: int) -> str:
    """Deterministic, short chunk identifier."""
    raw = f"{passage_id}|{strategy}|{index}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:10]
    return f"{strategy[:4]}_{digest}"


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences, stripping blank results."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Strategy 1 -- Fixed-size chunker (sliding window)
# ---------------------------------------------------------------------------

def fixed_size_chunker(
    passages: List[str],
    size: int = 256,
    overlap: float = 0.2,
    language: str = "hi",
) -> List[Chunk]:
    """Split each passage into fixed-token-count windows with proportional overlap.

    Parameters
    ----------
    passages:
        Deduplicated passage strings.
    size:
        Target token count per chunk (whitespace tokens).
    overlap:
        Fraction of ``size`` to repeat across consecutive chunks.
    language:
        BCP-47 language tag written into every Chunk.

    Returns
    -------
    List[Chunk]
        All chunks from all passages, tagged with ``chunk_strategy="fixed_size"``.
    """
    stride = max(1, int(size * (1.0 - overlap)))
    chunks: List[Chunk] = []

    for pid, passage in enumerate(passages):
        passage_id = f"passage_{pid}"
        words = _WHITESPACE_RE.split(passage.strip())
        if not words:
            continue

        window_idx = 0
        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            window_text = " ".join(words[start:end])
            tok_count = end - start

            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(passage_id, "fixed_size", window_idx),
                    doc_id=passage_id,
                    text=window_text,
                    metadata=ChunkMetadata(
                        source=passage_id,
                        token_count=tok_count,
                    ),
                    source_passage_id=passage_id,
                    language=language,
                    chunk_strategy="fixed_size",
                    token_count=tok_count,
                    parent_text=passage,
                )
            )

            if end == len(words):
                break
            start += stride
            window_idx += 1

    return chunks


# ---------------------------------------------------------------------------
# Strategy 2 -- Semantic chunker (embedding similarity boundary detection)
# ---------------------------------------------------------------------------

def _load_embedding_model(model_name: str = "intfloat/multilingual-e5-small"):
    """Lazy-load a SentenceTransformer model (imported here to keep top-level fast)."""
    from sentence_transformers import SentenceTransformer  # type: ignore
    return SentenceTransformer(model_name)


def _embed_sentences(
    sentences: List[str],
    model,
    prefix: str = "passage: ",
) -> np.ndarray:
    """Encode sentences with the given model, returning shape (N, D)."""
    prefixed = [f"{prefix}{s}" for s in sentences]
    embeddings = model.encode(prefixed, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)


def _detect_boundaries(
    embeddings: np.ndarray,
    k: float = 1.5,
    min_window: int = 3,
) -> List[int]:
    """Return indices *after* which a semantic boundary should be placed.

    Method
    ------
    Compute cosine similarity between every consecutive pair of sentence
    embeddings.  A rolling mean and std over the similarity sequence are
    maintained; a boundary is inserted at position ``i`` whenever::

        sim[i] < rolling_mean - k * rolling_std

    Parameters
    ----------
    embeddings:
        Shape (N, D), one row per sentence.
    k:
        Sensitivity: lower -> more cuts; higher -> fewer, larger chunks.
    min_window:
        Minimum number of sentences seen before the rolling stats are
        considered stable enough to place a boundary.
    """
    n = len(embeddings)
    if n < 2:
        return []

    sims = [_cosine_sim(embeddings[i], embeddings[i + 1]) for i in range(n - 1)]
    boundaries: List[int] = []
    running_sum = 0.0
    running_sq = 0.0

    for i, sim in enumerate(sims):
        running_sum += sim
        running_sq += sim * sim
        count = i + 1
        mean = running_sum / count
        if count >= min_window:
            variance = max(0.0, running_sq / count - mean * mean)
            std = variance ** 0.5
            if sim < mean - k * std:
                boundaries.append(i)  # boundary *after* sentence i

    return boundaries


def semantic_chunker(
    passages: List[str],
    model_name: str = "intfloat/multilingual-e5-small",
    k: float = 1.5,
    language: str = "hi",
    _model=None,  # allow injection for testing / reuse
) -> List[Chunk]:
    """Split passages at semantic boundaries detected by embedding similarity drops.

    Each passage is first sentence-tokenised, then consecutive sentence
    embeddings are compared.  A chunk boundary is inserted wherever the
    similarity falls below ``rolling_mean - k * rolling_std``.  A 1-sentence
    overlap is maintained across boundaries so no context is lost.

    Parameters
    ----------
    passages:
        Deduplicated passage strings.
    model_name:
        HuggingFace model id (sentence-transformers-compatible).
    k:
        Sensitivity for boundary detection (lower = more chunks).
    language:
        BCP-47 tag written into every Chunk.
    _model:
        Pre-loaded SentenceTransformer instance (for testing / reuse).

    Returns
    -------
    List[Chunk]
        Tagged with ``chunk_strategy="semantic"``.
    """
    model = _model or _load_embedding_model(model_name)
    chunks: List[Chunk] = []

    # First pass: identify passages with >2 sentences and collect sentences to batch-encode
    passages_data = []
    all_sentences: List[str] = []
    sentence_offsets: List[Tuple[int, int]] = []  # (start_idx, end_idx) in all_sentences

    for pid, passage in enumerate(passages):
        passage_id = f"passage_{pid}"
        sentences = _split_sentences(passage)
        if not sentences:
            passages_data.append((passage_id, passage, sentences, None))
            continue

        if len(sentences) <= 2:
            # Short passage, no boundary detection needed
            passages_data.append((passage_id, passage, sentences, None))
        else:
            start = len(all_sentences)
            all_sentences.extend(sentences)
            end = len(all_sentences)
            passages_data.append((passage_id, passage, sentences, (start, end)))

    # Batch encode all sentences across all passages at once
    all_embeddings: Optional[np.ndarray] = None
    if all_sentences:
        all_embeddings = _embed_sentences(all_sentences, model)

    # Second pass: detect boundaries and construct chunks
    for passage_id, passage, sentences, offset in passages_data:
        if not sentences:
            continue

        if offset is None:
            # <=2 sentences -> single chunk
            text = " ".join(sentences)
            tok = _token_count(text)
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(passage_id, "semantic", 0),
                    doc_id=passage_id,
                    text=text,
                    metadata=ChunkMetadata(source=passage_id, token_count=tok),
                    source_passage_id=passage_id,
                    language=language,
                    chunk_strategy="semantic",
                    token_count=tok,
                    parent_text=passage,
                )
            )
            continue

        start, end = offset
        embeddings = all_embeddings[start:end] if all_embeddings is not None else np.zeros((len(sentences), 1))
        boundaries = _detect_boundaries(embeddings, k=k)
        boundary_set = set(boundaries)

        # Build segments: each segment is a list of sentence indices.
        segments: List[List[int]] = []
        current: List[int] = [0]
        for i in range(1, len(sentences)):
            if (i - 1) in boundary_set:
                segments.append(current)
                # 1-sentence overlap: start new segment with the last sentence
                current = [i - 1, i]
            else:
                current.append(i)
        segments.append(current)

        for seg_idx, seg_indices in enumerate(segments):
            seg_text = " ".join(sentences[j] for j in seg_indices)
            tok_count = _token_count(seg_text)
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(passage_id, "semantic", seg_idx),
                    doc_id=passage_id,
                    text=seg_text,
                    metadata=ChunkMetadata(source=passage_id, token_count=tok_count),
                    source_passage_id=passage_id,
                    language=language,
                    chunk_strategy="semantic",
                    token_count=tok_count,
                    parent_text=passage,
                )
            )

    return chunks


# ---------------------------------------------------------------------------
# Strategy 3 -- Small-to-big chunker
# ---------------------------------------------------------------------------

def small_to_big_chunker(
    passages: List[str],
    max_sentence_tokens: int = 128,
    language: str = "hi",
) -> List[Chunk]:
    """Index individual sentences as the retrievable unit; carry full passage as context.

    Each sentence from each passage becomes a separate Chunk whose ``text``
    is the sentence (  ``max_sentence_tokens`` whitespace tokens) and whose
    ``parent_text`` is the complete source passage.  At generation time, the
    retriever can return the parent passage for richer context while ranking
    on the focused sentence embedding.

    Adjacent sentences are merged into the current chunk if the combined
    token count stays within ``max_sentence_tokens``; otherwise a new chunk
    is started.

    Parameters
    ----------
    passages:
        Deduplicated passage strings.
    max_sentence_tokens:
        Upper token bound before a new chunk is started (target: 64 128).
    language:
        BCP-47 tag written into every Chunk.

    Returns
    -------
    List[Chunk]
        Tagged with ``chunk_strategy="small_to_big"``.
    """
    chunks: List[Chunk] = []

    for pid, passage in enumerate(passages):
        passage_id = f"passage_{pid}"
        sentences = _split_sentences(passage)
        if not sentences:
            continue

        current_sentences: List[str] = []
        current_tokens = 0
        chunk_idx = 0

        def _flush(sents: List[str], idx: int) -> Chunk:
            text = " ".join(sents)
            tok = _token_count(text)
            return Chunk(
                chunk_id=_make_chunk_id(passage_id, "small_to_big", idx),
                doc_id=passage_id,
                text=text,
                metadata=ChunkMetadata(source=passage_id, token_count=tok),
                source_passage_id=passage_id,
                language=language,
                chunk_strategy="small_to_big",
                token_count=tok,
                parent_text=passage,
            )

        for sentence in sentences:
            s_tokens = _token_count(sentence)
            if current_sentences and current_tokens + s_tokens > max_sentence_tokens:
                chunks.append(_flush(current_sentences, chunk_idx))
                chunk_idx += 1
                current_sentences = [sentence]
                current_tokens = s_tokens
            else:
                current_sentences.append(sentence)
                current_tokens += s_tokens

        if current_sentences:
            chunks.append(_flush(current_sentences, chunk_idx))

    return chunks


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_all_chunkers(
    passages: List[str],
    language: str = "hi",
    semantic_model=None,
) -> List[Chunk]:
    """Run all three strategies over ``passages`` and concatenate results.

    Parameters
    ----------
    passages:
        Deduplicated corpus passages.
    language:
        BCP-47 tag forwarded to all chunkers.
    semantic_model:
        Pre-loaded SentenceTransformer to avoid reloading for the semantic pass.

    Returns
    -------
    List[Chunk]
        Combined output from fixed_size, semantic, and small_to_big strategies.
    """
    print(f"  [chunker] fixed_size ...", flush=True)
    fixed = fixed_size_chunker(passages, language=language)

    print(f"  [chunker] semantic (loading model if needed) ...", flush=True)
    semantic = semantic_chunker(passages, language=language, _model=semantic_model)

    print(f"  [chunker] small_to_big ...", flush=True)
    s2b = small_to_big_chunker(passages, language=language)

    return fixed + semantic + s2b
