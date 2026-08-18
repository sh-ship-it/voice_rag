"""Unit tests for pipeline/generate.py (Cerebras OpenAI-compatible generation)."""

import json
from unittest.mock import MagicMock, patch
import pytest

from pipeline.generate import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT_SECONDS,
    _parse_generation_json,
    format_context_prompt,
    generate_answer,
    CerebrasGenerator,
)
from pipeline.schemas import Chunk, GenerationResult, RetrievalResult, ScoredChunk


class TestGeneratePrompt:
    def test_format_context_prompt_with_chunks(self):
        chunks = [
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="chunk_delhi_1",
                    doc_id="doc_1",
                    text="नई दिल्ली भारत की राजधानी है।",
                    chunk_strategy="fixed_size",
                ),
                score=0.032,
                rank=1,
            ),
            ScoredChunk(
                chunk=Chunk(
                    chunk_id="chunk_agra_2",
                    doc_id="doc_2",
                    text="ताजमहल आगरा में स्थित है।",
                    chunk_strategy="small_to_big",
                ),
                score=0.025,
                rank=2,
            ),
        ]
        prompt = format_context_prompt("भारत की राजधानी क्या है?", chunks)
        assert "chunk_delhi_1" in prompt
        assert "नई दिल्ली भारत की राजधानी है।" in prompt
        assert "chunk_agra_2" in prompt
        assert "User Question: भारत की राजधानी क्या है?" in prompt

    def test_format_context_prompt_empty_chunks(self):
        prompt = format_context_prompt("Test question", [])
        assert "No relevant evidence retrieved" in prompt
        assert "User Question: Test question" in prompt


class TestGenerateJsonParsing:
    def test_parse_valid_json(self):
        raw = json.dumps({
            "answer": "नई दिल्ली भारत की राजधानी है।",
            "citations": ["chunk_delhi_1"],
            "confidence": "high",
            "grounded": True,
        })
        res = _parse_generation_json(raw, "llama-3.3-70b", 45.2)
        assert isinstance(res, GenerationResult)
        assert res.answer == "नई दिल्ली भारत की राजधानी है।"
        assert res.citations == ["chunk_delhi_1"]
        assert res.confidence == "high"
        assert res.grounded is True
        assert res.generation_ms == 45.2

    def test_parse_json_in_markdown_codeblock(self):
        raw = """```json
{
  "answer": "ताजमहल शाहजहाँ द्वारा बनवाया गया था।",
  "citations": ["chunk_taj_1"],
  "confidence": "high",
  "grounded": true
}
```"""
        res = _parse_generation_json(raw, "llama-3.3-70b", 60.1)
        assert res is not None
        assert res.answer == "ताजमहल शाहजहाँ द्वारा बनवाया गया था।"
        assert res.citations == ["chunk_taj_1"]
        assert res.grounded is True

    def test_parse_malformed_json_returns_none(self):
        raw = "This is not json content"
        res = _parse_generation_json(raw, "llama-3.3-70b", 10.0)
        assert res is None


class TestGenerateAnswerExecution:
    def test_fallback_without_api_key(self):
        """With no keys and no retrieved chunks, the result is an empty extractive fallback."""
        res = generate_answer(
            query="भारत की राजधानी क्या है?",
            retrieval_result=RetrievalResult(query="test", chunks=[]),
            api_key=None,
        )
        assert isinstance(res, GenerationResult)
        # With empty retrieval, extractive fallback returns empty answer
        assert res.grounded is False
        assert res.response_mode in ("extractive_fallback", "refusal")

    def test_mock_successful_generation(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({
            "answer": "नई दिल्ली भारत की राजधानी है।",
            "citations": ["chunk_delhi_1"],
            "confidence": "high",
            "grounded": True,
        })
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 45
        mock_response.usage.completion_tokens = 18
        mock_response.usage.total_tokens = 63

        mock_client.chat.completions.create.return_value = mock_response

        retrieval_res = RetrievalResult(
            query="भारत की राजधानी क्या है?",
            chunks=[
                ScoredChunk(
                    chunk=Chunk(chunk_id="chunk_delhi_1", doc_id="d1", text="नई दिल्ली राजधानी है।"),
                    score=0.032,
                    rank=1,
                )
            ]
        )

        res = generate_answer(
            query="भारत की राजधानी क्या है?",
            retrieval_result=retrieval_res,
            api_key="test_mock_key",
            client=mock_client,
            max_tokens=DEFAULT_MAX_TOKENS,
            timeout_s=DEFAULT_TIMEOUT_SECONDS,
        )

        assert res.answer == "नई दिल्ली भारत की राजधानी है।"
        assert res.citations == ["chunk_delhi_1"]
        assert res.confidence == "high"
        assert res.grounded is True
        assert res.prompt_tokens == 45
        assert res.completion_tokens == 18
        assert res.generation_ms > 0.0

    def test_mock_timeout_fallback(self):
        """When Cerebras client (injected mock) times out with no chunks, returns empty extractive fallback."""
        from openai import APITimeoutError
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APITimeoutError(request=MagicMock())

        res = generate_answer(
            query="What is the capital?",
            retrieval_result=RetrievalResult(query="test", chunks=[]),
            api_key="test_mock_key",
            client=mock_client,
        )

        assert isinstance(res, GenerationResult)
        assert res.grounded is False
        assert res.confidence == "low"
        # Empty retrieval + timeout → empty extractive fallback (no chunks to extract from)
        assert res.response_mode == "extractive_fallback"
        assert res.fallback_reason == "cerebras_failure"

    def test_cerebras_generator_class(self):
        gen = CerebrasGenerator(api_key=None)
        res = gen.generate(
            query="Hello",
            retrieval_result=RetrievalResult(query="Hello", chunks=[]),
        )
        assert isinstance(res, GenerationResult)
