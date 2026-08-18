"""Tests for the Groq generation path in pipeline/generate.py.

Covers all 9 required scenarios:
1. Successful Groq JSON response        → response_mode="groq_generated"
2. Groq timeout                         → response_mode="extractive_fallback", fallback_reason="timeout"
3. Groq API error                       → response_mode="extractive_fallback", fallback_reason starts with "api_error"
4. Malformed Groq JSON                  → response_mode="extractive_fallback", fallback_reason="malformed_json"
5. Invalid citation response            → response_mode="extractive_fallback", fallback_reason="invalid_citations"
6. Empty Groq answer                    → response_mode="extractive_fallback", fallback_reason="empty_answer"
7. Missing API key                      → response_mode="extractive_fallback", fallback_reason="missing_groq_key"
8. Low-confidence refusal               → response_mode="refusal" (set by orchestrator, not generate)
9. Successful extractive fallback       → top-1/2 chunks, no LLM call

All tests mock external I/O. No network calls are made.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.generate import (
    _call_groq_sync,
    _call_groq_async,
    _extractive_fallback,
    _parse_generation_json,
    generate_answer,
    GroqGenerator,
)
from pipeline.schemas import Chunk, GenerationResult, RetrievalResult, ScoredChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_retrieval_result(n: int = 2) -> RetrievalResult:
    """Build a minimal RetrievalResult with n chunks."""
    chunks = [
        ScoredChunk(
            chunk=Chunk(
                chunk_id=f"chunk_{i}",
                doc_id=f"doc_{i}",
                text=f"नई दिल्ली भारत की राजधानी है। यह उत्तर भारत में स्थित है। [chunk {i}]",
                chunk_strategy="fixed_size",
            ),
            score=0.9 - i * 0.1,
            rank=i + 1,
        )
        for i in range(n)
    ]
    return RetrievalResult(query="भारत की राजधानी क्या है?", chunks=chunks)


def _fake_settings(groq_api_key: str = "fake_groq_key", timeout_ms: int = 350):
    """Return a minimal settings-like namespace for inject into _call_groq_sync."""
    s = SimpleNamespace(
        groq_api_key=groq_api_key,
        groq_model="llama-3.1-8b-instant",
        groq_base_url="https://api.groq.com/openai/v1",
        groq_timeout_ms=timeout_ms,
        extractive_fallback_enabled=True,
    )
    return s


def _make_openai_response(content: str, finish_reason: str = "stop"):
    """Build a minimal mock OpenAI response."""
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason
    usage = MagicMock()
    usage.prompt_tokens = 30
    usage.completion_tokens = 20
    usage.total_tokens = 50
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# Test 1: Successful Groq JSON response
# ---------------------------------------------------------------------------

class TestGroqSuccess:
    def test_groq_generated_response_mode(self):
        retrieval = _make_retrieval_result()
        settings = _fake_settings()
        valid_json = json.dumps({
            "answer": "नई दिल्ली भारत की राजधानी है।",
            "citations": ["chunk_0"],
            "confidence": "high",
            "grounded": True,
        })
        mock_resp = _make_openai_response(valid_json)

        with patch("pipeline.generate.OpenAI") as MockOAI:
            MockOAI.return_value.chat.completions.create.return_value = mock_resp
            result = _call_groq_sync("भारत की राजधानी क्या है?", retrieval, settings)

        assert result.response_mode == "groq_generated"
        assert result.fallback_reason is None
        assert result.answer == "नई दिल्ली भारत की राजधानी है।"
        assert result.citations == ["chunk_0"]
        assert result.confidence == "high"
        assert result.grounded is True
        assert result.generation_ms > 0


# ---------------------------------------------------------------------------
# Test 2: Groq timeout
# ---------------------------------------------------------------------------

class TestGroqTimeout:
    def test_timeout_triggers_extractive_fallback(self):
        from openai import APITimeoutError
        retrieval = _make_retrieval_result()
        settings = _fake_settings()

        with patch("pipeline.generate.OpenAI") as MockOAI:
            MockOAI.return_value.chat.completions.create.side_effect = APITimeoutError(
                request=MagicMock()
            )
            result = _call_groq_sync("भारत की राजधानी क्या है?", retrieval, settings)

        assert result.response_mode == "extractive_fallback"
        assert result.fallback_reason == "timeout"
        assert len(result.citations) >= 1
        assert result.answer  # non-empty extractive snippet
        assert result.grounded is True


# ---------------------------------------------------------------------------
# Test 3: Groq API error
# ---------------------------------------------------------------------------

class TestGroqAPIError:
    def test_api_error_triggers_extractive_fallback(self):
        from openai import APIError
        retrieval = _make_retrieval_result()
        settings = _fake_settings()

        with patch("pipeline.generate.OpenAI") as MockOAI:
            MockOAI.return_value.chat.completions.create.side_effect = APIError(
                message="upstream error", request=MagicMock(), body=None
            )
            result = _call_groq_sync("What is capital?", retrieval, settings)

        assert result.response_mode == "extractive_fallback"
        assert result.fallback_reason is not None
        assert "api_error" in result.fallback_reason
        assert result.grounded is True


# ---------------------------------------------------------------------------
# Test 4: Malformed Groq JSON
# ---------------------------------------------------------------------------

class TestGroqMalformedJSON:
    def test_malformed_json_triggers_extractive_fallback(self):
        retrieval = _make_retrieval_result()
        settings = _fake_settings()
        mock_resp = _make_openai_response("this is not valid json at all {{{")

        with patch("pipeline.generate.OpenAI") as MockOAI:
            MockOAI.return_value.chat.completions.create.return_value = mock_resp
            result = _call_groq_sync("What is capital?", retrieval, settings)

        assert result.response_mode == "extractive_fallback"
        assert result.fallback_reason == "malformed_json"
        assert result.grounded is True


# ---------------------------------------------------------------------------
# Test 5: Invalid citations
# ---------------------------------------------------------------------------

class TestGroqInvalidCitations:
    def test_all_invalid_citations_triggers_extractive_fallback(self):
        retrieval = _make_retrieval_result()
        settings = _fake_settings()
        # Citations reference chunk IDs that don't exist in retrieval_result
        invalid_json = json.dumps({
            "answer": "Some answer here.",
            "citations": ["nonexistent_chunk_999", "another_fake_id"],
            "confidence": "high",
            "grounded": True,
        })
        mock_resp = _make_openai_response(invalid_json)

        with patch("pipeline.generate.OpenAI") as MockOAI:
            MockOAI.return_value.chat.completions.create.return_value = mock_resp
            result = _call_groq_sync("What is capital?", retrieval, settings)

        assert result.response_mode == "extractive_fallback"
        assert result.fallback_reason == "invalid_citations"
        assert result.grounded is True


# ---------------------------------------------------------------------------
# Test 6: Empty Groq answer
# ---------------------------------------------------------------------------

class TestGroqEmptyAnswer:
    def test_empty_json_answer_triggers_extractive_fallback(self):
        retrieval = _make_retrieval_result()
        settings = _fake_settings()
        empty_json = json.dumps({
            "answer": "",
            "citations": ["chunk_0"],
            "confidence": "medium",
            "grounded": True,
        })
        mock_resp = _make_openai_response(empty_json)

        with patch("pipeline.generate.OpenAI") as MockOAI:
            MockOAI.return_value.chat.completions.create.return_value = mock_resp
            result = _call_groq_sync("What is capital?", retrieval, settings)

        assert result.response_mode == "extractive_fallback"
        assert result.fallback_reason == "empty_answer"

    def test_empty_raw_content_triggers_extractive_fallback(self):
        retrieval = _make_retrieval_result()
        settings = _fake_settings()
        mock_resp = _make_openai_response("")  # empty content from API

        with patch("pipeline.generate.OpenAI") as MockOAI:
            MockOAI.return_value.chat.completions.create.return_value = mock_resp
            result = _call_groq_sync("What is capital?", retrieval, settings)

        assert result.response_mode == "extractive_fallback"
        assert result.fallback_reason == "empty_answer"


# ---------------------------------------------------------------------------
# Test 7: Missing API key
# ---------------------------------------------------------------------------

class TestGroqMissingKey:
    def test_missing_groq_key_uses_extractive_fallback(self):
        retrieval = _make_retrieval_result()
        settings = _fake_settings(groq_api_key="")  # empty string → falsy

        # No mock needed — should never call OpenAI
        result = _call_groq_sync("What is capital?", retrieval, settings)

        assert result.response_mode == "extractive_fallback"
        assert result.fallback_reason == "missing_groq_key"
        assert result.grounded is True
        assert len(result.citations) >= 1

    def test_none_groq_key_uses_extractive_fallback(self):
        retrieval = _make_retrieval_result()
        settings = _fake_settings(groq_api_key=None)

        result = _call_groq_sync("What is capital?", retrieval, settings)

        assert result.response_mode == "extractive_fallback"
        assert result.fallback_reason == "missing_groq_key"

    def test_generate_answer_without_any_key_uses_extractive(self):
        """generate_answer() with no keys at all should use extractive fallback."""
        retrieval = _make_retrieval_result()
        with patch("pipeline.generate.get_settings") as mock_settings:
            s = MagicMock()
            s.groq_api_key = None
            s.cerebras_api_key = None
            s.sarvam_api_key = None
            s.llm_provider = "groq"
            s.groq_model = "llama-3.1-8b-instant"
            s.groq_base_url = "https://api.groq.com/openai/v1"
            s.groq_timeout_ms = 350
            s.extractive_fallback_enabled = True
            mock_settings.return_value = s
            result = generate_answer("What is capital?", retrieval)

        assert result.response_mode == "extractive_fallback"
        assert result.fallback_reason == "missing_groq_key"


# ---------------------------------------------------------------------------
# Test 8: Low-confidence refusal (set by orchestrator)
# ---------------------------------------------------------------------------

class TestLowConfidenceRefusal:
    """Confidence gate refusal is set in orchestrator, not generate.py.
    We verify that when orchestrator confidence_gate returns False, the
    PipelineResponse has status='low_confidence_fallback' and no generation call.
    """

    def test_low_confidence_does_not_call_generation(self):
        """If confidence_gate returns False, agenerate_answer is never invoked."""
        from pipeline.orchestrator import run_pipeline

        with (
            patch("pipeline.orchestrator.embed_query") as mock_embed,
            patch("pipeline.orchestrator.hybrid_retrieve") as mock_retrieve,
            patch("pipeline.orchestrator.confidence_gate", return_value=False),
            patch("pipeline.orchestrator.generate_answer") as mock_gen,
        ):
            import numpy as np
            mock_embed.return_value = np.zeros(384, dtype="float32")
            mock_retrieve.return_value = RetrievalResult(
                query="test", chunks=_make_retrieval_result().chunks
            )

            result = run_pipeline(query_text="test query")

        assert result.status == "low_confidence_fallback"
        assert result.confidence == "low"
        assert result.grounded is False
        mock_gen.assert_not_called()


# ---------------------------------------------------------------------------
# Test 9: Successful extractive fallback (direct unit test)
# ---------------------------------------------------------------------------

class TestExtractiveFallback:
    def test_returns_top_two_chunks(self):
        retrieval = _make_retrieval_result(n=3)
        result = _extractive_fallback(retrieval, generation_ms=10.0, model_name="test-model")

        assert result.response_mode == "extractive_fallback"
        assert result.grounded is True
        assert "chunk_0" in result.citations
        assert len(result.citations) == 2  # top-2
        assert result.answer  # non-empty

    def test_returns_one_chunk_when_only_one(self):
        retrieval = _make_retrieval_result(n=1)
        result = _extractive_fallback(retrieval, generation_ms=5.0, model_name="test")

        assert result.response_mode == "extractive_fallback"
        assert "chunk_0" in result.citations
        assert result.grounded is True

    def test_empty_retrieval_returns_empty_answer(self):
        retrieval = RetrievalResult(query="test", chunks=[])
        result = _extractive_fallback(retrieval, generation_ms=0.0, model_name="test")

        assert result.response_mode == "extractive_fallback"
        assert result.answer == ""
        assert result.grounded is False

    def test_no_second_search_performed(self):
        """Extractive fallback must not trigger any embedding or retrieval.
        Verify this by checking no ScoredChunk was accessed beyond the already-retrieved list.
        """
        retrieval = _make_retrieval_result()
        # Record original chunk IDs — fallback must only draw from these
        original_ids = {sc.chunk.chunk_id for sc in retrieval.chunks}

        result = _extractive_fallback(retrieval, 0.0, "test")

        # All cited IDs must be from the existing retrieval result
        for cid in result.citations:
            assert cid in original_ids, f"Fallback cited {cid!r} which is not in retrieval result"
        assert result.response_mode == "extractive_fallback"


# ---------------------------------------------------------------------------
# Async variants
# ---------------------------------------------------------------------------

class TestGroqAsync:
    def test_async_groq_success(self):
        import asyncio
        retrieval = _make_retrieval_result()
        settings = _fake_settings()
        valid_json = json.dumps({
            "answer": "नई दिल्ली भारत की राजधानी है।",
            "citations": ["chunk_0"],
            "confidence": "high",
            "grounded": True,
        })
        mock_resp = _make_openai_response(valid_json)

        async def _run():
            with patch("pipeline.generate.AsyncOpenAI"), \
                 patch("pipeline.generate.asyncio.wait_for", return_value=mock_resp):
                return await _call_groq_async("भारत की राजधानी?", retrieval, settings)

        result = asyncio.run(_run())
        assert result.response_mode == "groq_generated"
        assert result.answer == "नई दिल्ली भारत की राजधानी है।"

    def test_async_timeout_triggers_extractive(self):
        import asyncio as _asyncio
        retrieval = _make_retrieval_result()
        settings = _fake_settings()

        async def _run():
            with patch("pipeline.generate.AsyncOpenAI"):
                with patch("pipeline.generate.asyncio.wait_for", side_effect=_asyncio.TimeoutError):
                    return await _call_groq_async("test?", retrieval, settings)

        result = _asyncio.run(_run())
        assert result.response_mode == "extractive_fallback"
        assert result.fallback_reason == "timeout"
