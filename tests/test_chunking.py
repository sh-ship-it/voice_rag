"""Unit tests for the three chunking strategies in pipeline/chunking.py."""

import math
from typing import List

import numpy as np
import pytest

from pipeline.chunking import (
    _cosine_sim,
    _detect_boundaries,
    _make_chunk_id,
    _split_sentences,
    _token_count,
    fixed_size_chunker,
    run_all_chunkers,
    semantic_chunker,
    small_to_big_chunker,
)
from pipeline.schemas import Chunk

# ---------------------------------------------------------------------------
# Helpers & fixtures
# ---------------------------------------------------------------------------

SAMPLE_PASSAGES_EN = [
    "The quick brown fox jumps over the lazy dog. It was a sunny day in the meadow. The birds sang cheerfully.",
    "Artificial intelligence is transforming every industry. From healthcare to finance, algorithms are everywhere.",
    "Short passage.",
]

SAMPLE_PASSAGES_HI = [
    "भारत एक विशाल देश है। यहाँ अनेक भाषाएँ बोली जाती हैं। हिंदी सबसे अधिक बोली जाने वाली भाषा है।",
    "विज्ञान और प्रौद्योगिकी ने मानव जीवन को बदल दिया है।",
]

# A small fake embedding model that returns deterministic unit vectors
class _FakeEmbedder:
    """Returns normalised random vectors seeded by the input hash."""

    def __init__(self, dim: int = 64, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)

    def encode(self, texts, **kwargs):
        n = len(texts)
        vecs = self.rng.standard_normal((n, self.dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.maximum(norms, 1e-8)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

def test_token_count_basic():
    assert _token_count("hello world") == 2
    assert _token_count("  a  b  c  ") == 3
    assert _token_count("") == 1  # split("") → [''] → len 1


def test_split_sentences_english():
    text = "Hello world. This is a test! Is it working?"
    sents = _split_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "Hello world."


def test_split_sentences_hindi():
    text = "यह एक वाक्य है। यह दूसरा वाक्य है।"
    sents = _split_sentences(text)
    assert len(sents) == 2


def test_cosine_sim_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert math.isclose(_cosine_sim(v, v), 1.0, rel_tol=1e-6)


def test_cosine_sim_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert math.isclose(_cosine_sim(a, b), 0.0, abs_tol=1e-6)


def test_cosine_sim_zero_vector():
    a = np.zeros(3)
    b = np.array([1.0, 0.0, 0.0])
    assert _cosine_sim(a, b) == 0.0


def test_make_chunk_id_deterministic():
    cid1 = _make_chunk_id("p1", "fixed_size", 0)
    cid2 = _make_chunk_id("p1", "fixed_size", 0)
    assert cid1 == cid2


def test_make_chunk_id_unique_across_params():
    ids = {
        _make_chunk_id("p1", "fixed_size", 0),
        _make_chunk_id("p1", "fixed_size", 1),
        _make_chunk_id("p1", "semantic", 0),
        _make_chunk_id("p2", "fixed_size", 0),
    }
    assert len(ids) == 4


# ---------------------------------------------------------------------------
# Strategy 1 — fixed_size_chunker
# ---------------------------------------------------------------------------

class TestFixedSizeChunker:
    def test_returns_chunks(self):
        chunks = fixed_size_chunker(SAMPLE_PASSAGES_EN, size=10, overlap=0.0)
        assert len(chunks) > 0

    def test_chunk_strategy_tag(self):
        chunks = fixed_size_chunker(SAMPLE_PASSAGES_EN, size=10)
        assert all(c.chunk_strategy == "fixed_size" for c in chunks)

    def test_chunk_type(self):
        chunks = fixed_size_chunker(SAMPLE_PASSAGES_EN, size=10)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_token_count_within_bound(self):
        size = 8
        chunks = fixed_size_chunker(SAMPLE_PASSAGES_EN, size=size, overlap=0.0)
        for c in chunks:
            # Each chunk may be up to `size` tokens
            assert c.token_count is not None
            assert c.token_count <= size

    def test_overlap_produces_more_chunks(self):
        no_overlap = fixed_size_chunker(SAMPLE_PASSAGES_EN, size=5, overlap=0.0)
        with_overlap = fixed_size_chunker(SAMPLE_PASSAGES_EN, size=5, overlap=0.5)
        assert len(with_overlap) >= len(no_overlap)

    def test_source_passage_id_set(self):
        chunks = fixed_size_chunker(SAMPLE_PASSAGES_EN)
        assert all(c.source_passage_id is not None for c in chunks)
        assert all(c.source_passage_id.startswith("passage_") for c in chunks)

    def test_language_field(self):
        chunks = fixed_size_chunker(SAMPLE_PASSAGES_EN, language="en")
        assert all(c.language == "en" for c in chunks)

    def test_parent_text_preserved(self):
        chunks = fixed_size_chunker(SAMPLE_PASSAGES_EN[:1], size=5, overlap=0.0)
        assert all(c.parent_text is not None for c in chunks)
        assert all(SAMPLE_PASSAGES_EN[0] in c.parent_text for c in chunks)

    def test_empty_passage_list(self):
        assert fixed_size_chunker([]) == []

    def test_single_short_passage_one_chunk(self):
        chunks = fixed_size_chunker(["hello world"], size=256)
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"

    def test_chunk_ids_unique(self):
        chunks = fixed_size_chunker(SAMPLE_PASSAGES_EN * 2, size=5)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_hindi_passages(self):
        chunks = fixed_size_chunker(SAMPLE_PASSAGES_HI, size=10, language="hi")
        assert len(chunks) > 0
        assert all(c.language == "hi" for c in chunks)


# ---------------------------------------------------------------------------
# Strategy 2 — semantic_chunker
# ---------------------------------------------------------------------------

class TestSemanticChunker:
    def test_returns_chunks(self):
        model = _FakeEmbedder()
        chunks = semantic_chunker(SAMPLE_PASSAGES_EN, _model=model)
        assert len(chunks) > 0

    def test_chunk_strategy_tag(self):
        model = _FakeEmbedder()
        chunks = semantic_chunker(SAMPLE_PASSAGES_EN, _model=model)
        assert all(c.chunk_strategy == "semantic" for c in chunks)

    def test_parent_text_preserved(self):
        model = _FakeEmbedder()
        chunks = semantic_chunker(SAMPLE_PASSAGES_EN, _model=model)
        assert all(c.parent_text is not None for c in chunks)

    def test_short_passage_single_chunk(self):
        """Passages with ≤2 sentences should produce exactly 1 chunk."""
        model = _FakeEmbedder()
        short = ["Only one sentence here."]
        chunks = semantic_chunker(short, _model=model)
        assert len(chunks) == 1

    def test_empty_passage_skipped(self):
        model = _FakeEmbedder()
        chunks = semantic_chunker(["", "   "], _model=model)
        assert chunks == []

    def test_chunk_type(self):
        model = _FakeEmbedder()
        chunks = semantic_chunker(SAMPLE_PASSAGES_EN[:1], _model=model)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_ids_unique(self):
        model = _FakeEmbedder()
        chunks = semantic_chunker(SAMPLE_PASSAGES_EN, _model=model)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_detect_boundaries_empty(self):
        assert _detect_boundaries(np.zeros((1, 4))) == []


def test_detect_boundaries_low_similarity():
    """High-k should produce fewer boundaries than low-k."""
    rng = np.random.default_rng(1)
    embeddings = rng.standard_normal((10, 16)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    few = _detect_boundaries(embeddings, k=5.0)
    many = _detect_boundaries(embeddings, k=0.1)
    assert len(many) >= len(few)


def test_detect_boundaries_empty_array():
    assert _detect_boundaries(np.zeros((1, 4))) == []


# ---------------------------------------------------------------------------
# Strategy 3 — small_to_big_chunker
# ---------------------------------------------------------------------------

class TestSmallToBigChunker:
    def test_returns_chunks(self):
        chunks = small_to_big_chunker(SAMPLE_PASSAGES_EN)
        assert len(chunks) > 0

    def test_chunk_strategy_tag(self):
        chunks = small_to_big_chunker(SAMPLE_PASSAGES_EN)
        assert all(c.chunk_strategy == "small_to_big" for c in chunks)

    def test_parent_text_equals_passage(self):
        """parent_text should be the full original passage."""
        chunks = small_to_big_chunker(SAMPLE_PASSAGES_EN[:1])
        assert all(c.parent_text == SAMPLE_PASSAGES_EN[0] for c in chunks)

    def test_token_limit_respected(self):
        limit = 15
        chunks = small_to_big_chunker(SAMPLE_PASSAGES_EN, max_sentence_tokens=limit)
        for c in chunks:
            assert c.token_count is not None
            # Token count should not wildly exceed the limit
            assert c.token_count <= limit + 20  # one extra sentence grace

    def test_chunk_ids_unique(self):
        chunks = small_to_big_chunker(SAMPLE_PASSAGES_EN * 2)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_empty_list(self):
        assert small_to_big_chunker([]) == []

    def test_single_sentence_passage(self):
        chunks = small_to_big_chunker(["A single sentence."])
        assert len(chunks) == 1
        assert chunks[0].text.strip() == "A single sentence."

    def test_source_passage_id_set(self):
        chunks = small_to_big_chunker(SAMPLE_PASSAGES_EN)
        assert all(c.source_passage_id is not None for c in chunks)

    def test_hindi_passages(self):
        chunks = small_to_big_chunker(SAMPLE_PASSAGES_HI, language="hi")
        assert all(c.language == "hi" for c in chunks)


# ---------------------------------------------------------------------------
# run_all_chunkers — integration
# ---------------------------------------------------------------------------

class TestRunAllChunkers:
    def test_all_strategies_present(self):
        model = _FakeEmbedder()
        chunks = run_all_chunkers(SAMPLE_PASSAGES_EN, language="en", semantic_model=model)
        strategies = {c.chunk_strategy for c in chunks}
        assert strategies == {"fixed_size", "semantic", "small_to_big"}

    def test_total_chunk_count_reasonable(self):
        model = _FakeEmbedder()
        chunks = run_all_chunkers(SAMPLE_PASSAGES_EN, semantic_model=model)
        # Should have at least 3 strategies * 1 chunk per passage
        assert len(chunks) >= 3 * len(SAMPLE_PASSAGES_EN)

    def test_all_chunks_have_provenance_fields(self):
        model = _FakeEmbedder()
        chunks = run_all_chunkers(SAMPLE_PASSAGES_EN[:1], semantic_model=model)
        for c in chunks:
            assert c.source_passage_id is not None
            assert c.chunk_strategy is not None
            assert c.token_count is not None
            assert c.parent_text is not None
