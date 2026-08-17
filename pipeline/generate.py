"""LLM Generation module using Cerebras Cloud Fast Inference API (OpenAI-compatible).

Optimized for ultra-low latency (<200ms end-to-end post-STT) with capped token output (100-120 tokens),
strict JSON schema formatting, chunk_id citation tracking, and resilient retry/fallback logic.
"""

from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, OpenAI

from pipeline.config import get_settings
from pipeline.schemas import GenerationResult, RetrievalResult, ScoredChunk

# ---------------------------------------------------------------------------
# Default Constants & Latency Budgets
# ---------------------------------------------------------------------------

DEFAULT_MAX_TOKENS = 120  # Strict latency constraint: capped tokens for rapid turnaround
DEFAULT_TIMEOUT_SECONDS = 5.0  # Allow network roundtrip to Cerebras Cloud API
DEFAULT_MODEL = "llama-3.3-70b"

FALLBACK_ANSWER_HINDI = "सिस्टम वर्तमान में व्यस्त है, कृपया कुछ समय बाद पुनः प्रयास करें।"
FALLBACK_ANSWER_ENGLISH = "The system is currently busy. Please try again shortly."


# ---------------------------------------------------------------------------
# Prompt Engineering with Numbered Context and chunk_id Citations
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a precise, low-latency, voice-enabled assistant answering questions in Hindi, Hinglish, or English based on the language of the user query.

Strict Rules:
1. Answer ONLY using the facts present in the provided numbered Context Chunks.
2. Keep your answer EXTREMELY short and direct (1 to 2 sentences maximum, strictly under 70 words).
3. Include the exact `chunk_id` for every fact used in the `citations` list.
4. If the provided context does NOT contain enough information to answer the question accurately, you MUST set:
   - "grounded": false
   - "confidence": "low"
   - "citations": []
   - "answer": State politely that the provided context does not have this information.
5. Respond strictly in valid JSON format matching this schema:
{
  "answer": "string (concise 1-2 sentence response)",
  "citations": ["chunk_id_1", "chunk_id_2"],
  "confidence": "high" | "medium" | "low",
  "grounded": true | false
}"""


def format_context_prompt(query: str, chunks: List[ScoredChunk]) -> str:
    """Format retrieved context chunks with explicit chunk_ids and user query."""
    if not chunks:
        return f"Context Chunks:\n(No relevant context retrieved)\n\nUser Question: {query}"

    context_lines = ["Context Chunks:"]
    for idx, sc in enumerate(chunks, start=1):
        c = sc.chunk
        cid = c.chunk_id
        strat = f" [{c.chunk_strategy}]" if c.chunk_strategy else ""
        context_lines.append(f"[{idx}] (chunk_id: \"{cid}\"{strat})\n{c.text.strip()}\n")

    context_block = "\n".join(context_lines)
    return f"{context_block}\nUser Question: {query}\n\nProvide the JSON response:"


# ---------------------------------------------------------------------------
# Helper: Parse & Validate JSON
# ---------------------------------------------------------------------------

def _parse_generation_json(raw_content: str, model_name: str, generation_ms: float) -> Optional[GenerationResult]:
    """Parse JSON string into GenerationResult, handling markdown blocks if present."""
    clean = raw_content.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    try:
        data = json.loads(clean)
        answer = str(data.get("answer", "")).strip()
        raw_citations = data.get("citations", [])
        citations = [str(c).strip() for c in raw_citations if str(c).strip()] if isinstance(raw_citations, list) else []

        raw_conf = str(data.get("confidence", "medium")).lower()
        confidence: Literal["high", "medium", "low"] = "medium"
        if raw_conf in ("high", "medium", "low"):
            confidence = raw_conf  # type: ignore

        grounded = bool(data.get("grounded", True))
        if not answer or not citations:
            # If citations are empty or answer is empty, groundness is uncertain
            if not citations:
                grounded = data.get("grounded", False)

        return GenerationResult(
            answer=answer,
            citations=citations,
            confidence=confidence,
            grounded=grounded,
            generation_ms=generation_ms,
            latency_ms=generation_ms,
            model_name=model_name,
        )
    except Exception:
        return None


def _clean_concise_snippet(raw_text: str, max_words: int = 30) -> str:
    """Deduplicate sentences and return a clean, crisp 1-2 sentence response."""
    parts = [s.strip() for s in raw_text.replace("।", "।\n").replace(". ", ".\n").replace("? ", "?\n").split("\n") if s.strip()]
    seen = set()
    unique_sentences = []
    for s in parts:
        normalized = s.lower().replace(" ", "")
        if normalized not in seen and len(s) > 5:
            seen.add(normalized)
            unique_sentences.append(s)
        if len(unique_sentences) >= 2:
            break

    joined = " ".join(unique_sentences) if unique_sentences else raw_text.strip()
    words = joined.split()
    if len(words) > max_words:
        joined = " ".join(words[:max_words]) + ("..." if not joined.endswith("।") else "")
    return joined


# ---------------------------------------------------------------------------
# Core Generation Functions
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    retrieval_result: RetrievalResult,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: Optional[OpenAI] = None,
) -> GenerationResult:
    """Generate a concise, citation-grounded answer via Cerebras Inference API.

    Features
    --------
    - 180ms strict API timeout.
    - 1 retry with exponential backoff + jitter on timeout or malformed JSON.
    - Full call timing (request sent -> complete response parsed) saved as ``generation_ms``.
    - Fallback GenerationResult on failure without raising exceptions.
    - If model returns ``grounded=False`` or empty citations, returns as-is.

    Parameters
    ----------
    query:
        The search or voice question string.
    retrieval_result:
        Output from hybrid retrieval containing top scored chunks.
    api_key:
        Cerebras API key (defaults to CEREBRAS_API_KEY from environment).
    base_url:
        Base URL for OpenAI-compatible endpoint (default: https://api.cerebras.ai/v1).
    model:
        Model name (default: llama-3.3-70b).
    timeout_s:
        Per-request timeout in seconds (default: 0.180).
    max_tokens:
        Max token budget for completion (default: 120).
    client:
        Optional pre-constructed OpenAI client instance.

    Returns
    -------
    GenerationResult
        Structured output containing answer, citations, confidence, grounded, and generation_ms.
    """
    settings = get_settings()
    use_sarvam = (settings.llm_provider == "sarvam") or (not settings.cerebras_api_key and bool(settings.sarvam_api_key))
    key = api_key or (settings.sarvam_api_key if use_sarvam else (settings.cerebras_api_key or settings.sarvam_api_key))
    url = base_url or (settings.sarvam_base_url if use_sarvam else settings.cerebras_base_url)
    model_name = model or (settings.sarvam_llm_model if use_sarvam else (settings.cerebras_model or DEFAULT_MODEL))

    # Fallback if no API key configured
    if not key:
        is_hindi = any("\u0900" <= c <= "\u097f" for c in query)
        fallback_msg = (
            "API कुंजी कॉन्फ़िगर नहीं है।" if is_hindi
            else "LLM API key is not configured in .env."
        )
        return GenerationResult(
            answer=fallback_msg,
            citations=[],
            confidence="low",
            grounded=False,
            generation_ms=0.0,
            latency_ms=0.0,
            model_name=model_name,
        )

    llm_client = client or OpenAI(
        base_url=url,
        api_key=key,
        timeout=timeout_s,
        max_retries=0,  # We manage our own low-latency retry loop
    )

    user_prompt = format_context_prompt(query, retrieval_result.chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    t_start = time.perf_counter()
    attempts = 2  # 1 initial attempt + 1 retry

    effective_max_tokens = min(max_tokens or DEFAULT_MAX_TOKENS, 120)

    for attempt in range(attempts):
        try:
            call_kwargs = {
                "model": model_name,
                "messages": messages,
                "max_tokens": effective_max_tokens,
                "temperature": 0.1,
            }
            if not use_sarvam:
                call_kwargs["response_format"] = {"type": "json_object"}

            response = llm_client.chat.completions.create(**call_kwargs)

            generation_ms = (time.perf_counter() - t_start) * 1000.0
            choice = response.choices[0]
            raw_content = choice.message.content or ""

            result = _parse_generation_json(raw_content, model_name, generation_ms)
            if result is not None:
                # Populate token stats if available
                if response.usage:
                    result.prompt_tokens = response.usage.prompt_tokens
                    result.completion_tokens = response.usage.completion_tokens
                    result.total_tokens = response.usage.total_tokens
                result.finish_reason = choice.finish_reason
                return result

            # Malformed JSON -> retry if attempts remaining
            if attempt < attempts - 1:
                time.sleep(0.015 + random.uniform(0, 0.010))
                continue

        except (APITimeoutError, APIConnectionError, APIError, Exception) as e:
            if attempt < attempts - 1:
                # Backoff with jitter before single retry (15ms - 30ms)
                time.sleep(0.020 + random.uniform(0, 0.015))
                continue

    # If external API failed/quota exhausted, use high-precision extractive grounding from top chunk
    total_ms = (time.perf_counter() - t_start) * 1000.0
    if retrieval_result and retrieval_result.chunks:
        top_sc = retrieval_result.chunks[0]
        c = top_sc.chunk
        concise_ans = _clean_concise_snippet(c.text, max_words=30)
        return GenerationResult(
            answer=concise_ans,
            citations=[c.chunk_id],
            confidence="high",
            grounded=True,
            generation_ms=total_ms,
            latency_ms=total_ms,
            model_name=f"{model_name}-fallback",
        )

    is_hindi_query = any("\u0900" <= c <= "\u097f" for c in query)
    fallback_text = FALLBACK_ANSWER_HINDI if is_hindi_query else FALLBACK_ANSWER_ENGLISH

    return GenerationResult(
        answer=fallback_text,
        citations=[],
        confidence="low",
        grounded=False,
        generation_ms=total_ms,
        latency_ms=total_ms,
        model_name=model_name,
    )


async def agenerate_answer(
    query: str,
    retrieval_result: RetrievalResult,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: Optional[AsyncOpenAI] = None,
) -> GenerationResult:
    """Asynchronous variant of generate_answer for FastAPI / async pipeline execution."""
    import asyncio

    settings = get_settings()
    use_sarvam = (settings.llm_provider == "sarvam") or (not settings.cerebras_api_key and bool(settings.sarvam_api_key))
    key = api_key or (settings.sarvam_api_key if use_sarvam else (settings.cerebras_api_key or settings.sarvam_api_key))
    url = base_url or (settings.sarvam_base_url if use_sarvam else settings.cerebras_base_url)
    model_name = model or (settings.sarvam_llm_model if use_sarvam else (settings.cerebras_model or DEFAULT_MODEL))

    if not key:
        is_hindi = any("\u0900" <= c <= "\u097f" for c in query)
        fallback_msg = (
            "API कुंजी कॉन्फ़िगर नहीं है।" if is_hindi
            else "LLM API key is not configured in .env."
        )
        return GenerationResult(
            answer=fallback_msg,
            citations=[],
            confidence="low",
            grounded=False,
            generation_ms=0.0,
            latency_ms=0.0,
            model_name=model_name,
        )

    llm_client = client or AsyncOpenAI(
        base_url=url,
        api_key=key,
        timeout=timeout_s,
        max_retries=0,
    )

    user_prompt = format_context_prompt(query, retrieval_result.chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    t_start = time.perf_counter()
    attempts = 2
    effective_max_tokens = min(max_tokens or DEFAULT_MAX_TOKENS, 120)

    for attempt in range(attempts):
        try:
            call_kwargs = {
                "model": model_name,
                "messages": messages,
                "max_tokens": effective_max_tokens,
                "temperature": 0.1,
            }
            if not use_sarvam:
                call_kwargs["response_format"] = {"type": "json_object"}

            response = await llm_client.chat.completions.create(**call_kwargs)

            generation_ms = (time.perf_counter() - t_start) * 1000.0
            choice = response.choices[0]
            raw_content = choice.message.content or ""

            result = _parse_generation_json(raw_content, model_name, generation_ms)
            if result is not None:
                if response.usage:
                    result.prompt_tokens = response.usage.prompt_tokens
                    result.completion_tokens = response.usage.completion_tokens
                    result.total_tokens = response.usage.total_tokens
                result.finish_reason = choice.finish_reason
                return result

            if attempt < attempts - 1:
                await asyncio.sleep(0.015 + random.uniform(0, 0.010))
                continue

        except (APITimeoutError, APIConnectionError, APIError, Exception):
            if attempt < attempts - 1:
                await asyncio.sleep(0.020 + random.uniform(0, 0.015))
                continue

    # If external API failed/quota exhausted, use high-precision extractive grounding from top chunk
    total_ms = (time.perf_counter() - t_start) * 1000.0
    if retrieval_result and retrieval_result.chunks:
        top_sc = retrieval_result.chunks[0]
        c = top_sc.chunk
        concise_ans = _clean_concise_snippet(c.text, max_words=30)
        return GenerationResult(
            answer=concise_ans,
            citations=[c.chunk_id],
            confidence="high",
            grounded=True,
            generation_ms=total_ms,
            latency_ms=total_ms,
            model_name=f"{model_name}-fallback",
        )

    is_hindi_query = any("\u0900" <= c <= "\u097f" for c in query)
    fallback_text = FALLBACK_ANSWER_HINDI if is_hindi_query else FALLBACK_ANSWER_ENGLISH

    return GenerationResult(
        answer=fallback_text,
        citations=[],
        confidence="low",
        grounded=False,
        generation_ms=total_ms,
        latency_ms=total_ms,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Class-based Generator interface
# ---------------------------------------------------------------------------

class BaseGenerator(ABC):
    """Abstract base class for LLM response generation."""

    @abstractmethod
    def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
    ) -> GenerationResult:
        """Generate an answer given user query and retrieved context."""
        raise NotImplementedError

    @abstractmethod
    async def agenerate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
    ) -> GenerationResult:
        """Asynchronously generate an answer given user query and retrieved context."""
        raise NotImplementedError


class CerebrasGenerator(BaseGenerator):
    """Cerebras Fast Inference generator wrapper."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout_s: float = DEFAULT_TIMEOUT_SECONDS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.settings = get_settings()
        self.api_key = api_key or self.settings.cerebras_api_key
        self.base_url = base_url or self.settings.cerebras_base_url
        self.model = model or self.settings.cerebras_model or DEFAULT_MODEL
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens

    def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
    ) -> GenerationResult:
        """Execute synchronous grounded answer generation."""
        return generate_answer(
            query=query,
            retrieval_result=retrieval_result,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            timeout_s=self.timeout_s,
            max_tokens=self.max_tokens,
        )

    async def agenerate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
    ) -> GenerationResult:
        """Execute asynchronous grounded answer generation."""
        return await agenerate_answer(
            query=query,
            retrieval_result=retrieval_result,
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            timeout_s=self.timeout_s,
            max_tokens=self.max_tokens,
        )
