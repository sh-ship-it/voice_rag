"""Safety, prompt-injection, off-topic, and retrieval confidence guardrails.

Optimized for ultra-low latency (<2ms) with zero neural network calls at evaluation time.
Uses compiled regex patterns and vector dot products for safety and off-topic checks.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from pipeline.schemas import GuardrailResult, RetrievalResult, ScoredChunk

# ---------------------------------------------------------------------------
# Fast Compiled Regex & Keyword Blocklists
# ---------------------------------------------------------------------------

_PROMPT_INJECTION_PATTERNS = [
    r"\bignore\s+(?:all\s+|previous\s+|prior\s+)*(?:instructions|prompts|rules)\b",
    r"\b(system\s+prompt|developer\s+mode|jailbreak|dan\s+mode)\b",
    r"\b(disregard\s+(?:all\s+|previous\s+|prior\s+)*(?:instructions|prompts|rules)|override\s+system|reveal\s+instructions)\b",
    r"\b(you\s+are\s+now|act\s+as\s+an\s+unrestricted)\b",
    r"\b(pretend\s+you\s+have\s+no\s+rules|unfiltered\s+mode)\b",
]

_HARMFUL_PATTERNS = [
    r"\b(make\s+a\s+bomb|synthesize\s+explosives|hack\s+into)\b",
    r"\b(steal\s+passwords|credit\s+card\s+fraud)\b",
]

_INJECTION_REGEX = re.compile("|".join(_PROMPT_INJECTION_PATTERNS), flags=re.IGNORECASE)
_HARMFUL_REGEX = re.compile("|".join(_HARMFUL_PATTERNS), flags=re.IGNORECASE)

# Theoretical max RRF score for top-1 in both Dense and Sparse (1/61 + 1/61 = 0.0327868)
MAX_THEORETICAL_RRF_SCORE = 2.0 / 61.0


# ---------------------------------------------------------------------------
# Functional Guardrails APIs
# ---------------------------------------------------------------------------

def input_guardrail(
    query_text: str,
    query_embedding: Optional[np.ndarray] = None,
    corpus_centroid: Optional[np.ndarray] = None,
    min_topic_similarity: float = 0.15,
) -> bool:
    """Evaluate input query for prompt injections, unsafe content, and off-topic domain.

    Executes in single-digit milliseconds with zero model calls:
    1. Rejects empty or whitespace-only queries.
    2. Runs fast compiled regex checks for prompt injection and harmful intents.
    3. If query_embedding & corpus_centroid are provided, computes vector dot-product
       cosine similarity; rejects if similarity falls below ``min_topic_similarity``.

    Parameters
    ----------
    query_text:
        Raw input string from speech-to-text or user input.
    query_embedding:
        Precomputed 1D float32 query vector (optional).
    corpus_centroid:
        Precomputed L2-normalized 1D float32 corpus centroid (optional).
    min_topic_similarity:
        Minimum required cosine similarity against corpus centroid (default: 0.15).

    Returns
    -------
    bool
        True if the query passes all safety and domain checks; False if rejected.
    """
    if not query_text or not query_text.strip():
        return False

    clean_text = query_text.strip()

    # 1. Prompt injection pattern matching (<0.5ms)
    if _INJECTION_REGEX.search(clean_text):
        return False

    # 2. Harmful intent pattern matching (<0.5ms)
    if _HARMFUL_REGEX.search(clean_text):
        return False

    # 3. Off-topic centroid similarity check (<0.1ms)
    if query_embedding is not None and corpus_centroid is not None:
        q_vec = np.asarray(query_embedding, dtype=np.float32).flatten()
        c_vec = np.asarray(corpus_centroid, dtype=np.float32).flatten()

        denom = np.linalg.norm(q_vec) * np.linalg.norm(c_vec)
        if denom > 0:
            sim = float(np.dot(q_vec, c_vec) / denom)
            if sim < min_topic_similarity:
                return False

    return True


def confidence_gate(
    retrieval_result: RetrievalResult,
    threshold: float = 0.30,
    min_gap_ratio: float = 0.005,
) -> bool:
    """Verify whether retrieval confidence is sufficient to proceed to LLM generation."""
    chunks = retrieval_result.chunks
    if not chunks:
        return False

    top1_score = chunks[0].score

    # Normalize RRF score to [0.0, 1.0] scale
    if top1_score <= MAX_THEORETICAL_RRF_SCORE:
        norm_score = top1_score / MAX_THEORETICAL_RRF_SCORE
    else:
        norm_score = min(1.0, top1_score)

    # 1. Top-1 score check
    if norm_score < threshold:
        return False

    # 2. Flatness check only if top1 score is marginal
    if len(chunks) >= 5 and norm_score < 0.40:
        top5_score = chunks[4].score
        gap = top1_score - top5_score
        if top1_score > 0 and (gap / top1_score) < min_gap_ratio:
            return False

    return True


# ---------------------------------------------------------------------------
# Structured Guardrail Classes
# ---------------------------------------------------------------------------

class BaseGuardrail(ABC):
    """Abstract base class for safety and moderation guardrails."""

    @abstractmethod
    def validate_input(self, text: str, embedding: Optional[np.ndarray] = None) -> GuardrailResult:
        """Evaluate input query for safety, injection, and domain relevance."""
        raise NotImplementedError

    @abstractmethod
    def validate_output(
        self,
        generated_text: str,
        context_chunks: Optional[List[ScoredChunk]] = None
    ) -> GuardrailResult:
        """Evaluate generated text for safety or empty generation."""
        raise NotImplementedError


class SafetyGuardrails(BaseGuardrail):
    """Structured rule-based guardrail engine."""

    def __init__(self, corpus_centroid: Optional[np.ndarray] = None) -> None:
        self.corpus_centroid = corpus_centroid

    def validate_input(
        self,
        text: str,
        embedding: Optional[np.ndarray] = None
    ) -> GuardrailResult:
        """Evaluate input query returning structured GuardrailResult."""
        if not text or not text.strip():
            return GuardrailResult(
                passed=False,
                flagged_categories=["empty_input"],
                rejection_reason="Query text cannot be empty.",
            )

        if _INJECTION_REGEX.search(text):
            return GuardrailResult(
                passed=False,
                flagged_categories=["prompt_injection"],
                rejection_reason="Prompt injection pattern detected.",
            )

        if _HARMFUL_REGEX.search(text):
            return GuardrailResult(
                passed=False,
                flagged_categories=["harmful_content"],
                rejection_reason="Harmful or policy-violating intent detected.",
            )

        if embedding is not None and self.corpus_centroid is not None:
            if not input_guardrail(text, embedding, self.corpus_centroid):
                return GuardrailResult(
                    passed=False,
                    flagged_categories=["off_topic"],
                    rejection_reason="Query is off-topic relative to the corpus domain.",
                )

        return GuardrailResult(
            passed=True,
            flagged_categories=[],
            sanitized_text=text.strip(),
        )

    def validate_output(
        self,
        generated_text: str,
        context_chunks: Optional[List[ScoredChunk]] = None
    ) -> GuardrailResult:
        """Evaluate generated text for emptiness and policy violations."""
        if not generated_text or not generated_text.strip():
            return GuardrailResult(
                passed=False,
                flagged_categories=["empty_output"],
                rejection_reason="Generated response is empty.",
            )

        if _HARMFUL_REGEX.search(generated_text):
            return GuardrailResult(
                passed=False,
                flagged_categories=["harmful_output"],
                rejection_reason="Generated response contained restricted patterns.",
            )

        return GuardrailResult(
            passed=True,
            flagged_categories=[],
            sanitized_text=generated_text.strip(),
        )
