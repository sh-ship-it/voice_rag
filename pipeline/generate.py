"""Pure Extractive RAG Generation Module.

Zero LLM calls. The answer is extracted directly and verbatim from the top retrieved
and reranked dataset evidence. Preserves verified citations and calibrated confidence tiers.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

from pipeline.confidence import compute_evidence_score, ConfidenceResult
from pipeline.schemas import GenerationResult, RetrievalResult, ScoredChunk

REFUSAL_TEXT_HINDI = "क्षमा करें, इस विषय पर उपलब्ध ज्ञानकोष में पर्याप्त जानकारी नहीं मिली।"
REFUSAL_TEXT_ENGLISH = "I'm sorry, there is not enough relevant information in the knowledge base to answer this question."

_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?।])\s+|(?<=[.!?।])$")


def split_into_sentences(text: str) -> List[str]:
    """Split Hindi and English text into individual sentences cleanly."""
    if not text:
        return []
    raw_sentences = _SENTENCE_SPLIT_REGEX.split(text.strip())
    cleaned = []
    for s in raw_sentences:
        s_clean = s.strip()
        if len(s_clean) >= 6:
            cleaned.append(s_clean)
    return cleaned if cleaned else [text.strip()]


def select_best_evidence_sentence(
    query: str,
    ranked_chunks: List[ScoredChunk],
    max_sentences: int = 2,
) -> Optional[Tuple[str, str, str]]:
    """Extract the best supporting sentence(s) and full passage from top ranked chunks.

    Returns
    -------
    Optional[Tuple[answer_sentence, full_evidence_text, primary_chunk_id]]
    """
    if not ranked_chunks:
        return None

    top_sc = ranked_chunks[0]
    top_chunk = top_sc.chunk
    full_text = top_chunk.text.strip()
    chunk_id = top_chunk.chunk_id

    # If small_to_big has parent_text, use parent for full evidence
    full_evidence = top_chunk.parent_text.strip() if top_chunk.parent_text else full_text

    # Extract sentences from the top chunk
    sentences = split_into_sentences(full_text)
    if not sentences:
        return full_text, full_evidence, chunk_id

    # Compute keyword overlap score with query tokens
    query_tokens = set(re.findall(r"\w+", query.lower()))
    scored_sentences = []

    for idx, sent in enumerate(sentences):
        sent_tokens = set(re.findall(r"\w+", sent.lower()))
        overlap = len(query_tokens.intersection(sent_tokens))
        # Bias slightly towards the first sentence if ties occur
        pos_bias = 0.5 / (idx + 1.0)
        scored_sentences.append((overlap + pos_bias, idx, sent))

    scored_sentences.sort(key=lambda x: x[0], reverse=True)

    # Pick top 1-2 most informative sentences
    selected = [scored_sentences[0][2]]
    if max_sentences > 1 and len(scored_sentences) > 1 and scored_sentences[1][0] > 0.5:
        idx1, idx2 = scored_sentences[0][1], scored_sentences[1][1]
        if abs(idx1 - idx2) == 1:
            if idx1 < idx2:
                selected = [scored_sentences[0][2], scored_sentences[1][2]]
            else:
                selected = [scored_sentences[1][2], scored_sentences[0][2]]
        else:
            selected.append(scored_sentences[1][2])

    answer_text = " ".join(selected).strip()
    return answer_text, full_evidence, chunk_id


def generate_extractive_response(
    query: str,
    retrieval_result: RetrievalResult,
) -> GenerationResult:
    """Generate pure extractive response directly from retrieved evidence.

    1. Computes multi-feature calibrated evidence score from retrieval signals.
    2. If decision is 'refuse', returns grounded refusal response.
    3. Selects the most relevant exact sentence from top reranked chunks.
    4. Sets confidence_tier ('high' | 'cautious' | 'low') and response_mode='extractive'.
    """
    t_start = time.perf_counter()
    ranked_chunks = retrieval_result.chunks if retrieval_result else []

    is_hindi = any("\u0900" <= c <= "\u097f" for c in query)
    refusal_msg = REFUSAL_TEXT_HINDI if is_hindi else REFUSAL_TEXT_ENGLISH

    if not ranked_chunks:
        dur_ms = (time.perf_counter() - t_start) * 1000.0
        return GenerationResult(
            answer=refusal_msg,
            citations=[],
            confidence="low",
            grounded=False,
            generation_ms=dur_ms,
            latency_ms=dur_ms,
            model_name="extractive-verbatim",
            response_mode="refusal",
            fallback_reason="empty_retrieval",
        )

    # Phase 3 Multi-Feature Confidence Estimation
    conf_res: ConfidenceResult = compute_evidence_score(query, retrieval_result)

    if conf_res.decision == "refuse":
        dur_ms = (time.perf_counter() - t_start) * 1000.0
        return GenerationResult(
            answer=refusal_msg,
            citations=[],
            confidence="low",
            grounded=False,
            generation_ms=dur_ms,
            latency_ms=dur_ms,
            model_name="extractive-verbatim",
            response_mode="refusal",
            fallback_reason=f"low_evidence_score:{conf_res.evidence_score:.3f}",
        )

    extracted = select_best_evidence_sentence(query, ranked_chunks, max_sentences=2)
    if not extracted:
        dur_ms = (time.perf_counter() - t_start) * 1000.0
        return GenerationResult(
            answer=refusal_msg,
            citations=[],
            confidence="low",
            grounded=False,
            generation_ms=dur_ms,
            latency_ms=dur_ms,
            model_name="extractive-verbatim",
            response_mode="refusal",
            fallback_reason="no_sentence_extracted",
        )

    answer_sentence, full_evidence, primary_cid = extracted

    # Collect top 1-3 verified citation chunk_ids
    citations = [sc.chunk.chunk_id for sc in ranked_chunks[:3]]
    if primary_cid not in citations:
        citations.insert(0, primary_cid)
    citations = citations[:3]

    dur_ms = (time.perf_counter() - t_start) * 1000.0
    mapped_conf: Literal["high", "medium", "low"] = "high" if conf_res.confidence_tier == "high" else "medium"

    return GenerationResult(
        answer=answer_sentence,
        citations=citations,
        confidence=mapped_conf,
        grounded=True,
        generation_ms=dur_ms,
        latency_ms=dur_ms,
        model_name="extractive-verbatim",
        response_mode="extractive",
    )


# Backward compatibility aliases
def generate_answer(query: str, retrieval_result: RetrievalResult) -> GenerationResult:
    """Synchronous extractive generation entry point."""
    return generate_extractive_response(query, retrieval_result)


async def agenerate_answer(query: str, retrieval_result: RetrievalResult) -> GenerationResult:
    """Asynchronous extractive generation entry point."""
    return generate_extractive_response(query, retrieval_result)
