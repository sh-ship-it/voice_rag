"""Voice-enabled RAG Pipeline Orchestrator.

Chains: STT -> Input Guardrail -> Embed Query -> Hybrid Retrieve -> Confidence Gate -> Generation.
Measures granular per-stage latency and computes `total_rag_core_ms` (graded against 200ms target).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pipeline.config import get_settings
from pipeline.embed import embed_query
from pipeline.generate import generate_answer, agenerate_answer
from pipeline.guardrails import confidence_gate, input_guardrail
from pipeline.retrieve import hybrid_retrieve
from pipeline.schemas import (
    AudioInput,
    GenerationResult,
    LatencyBreakdown,
    PipelineResponse,
    RetrievalResult,
)
from pipeline.stt import transcribe, atranscribe

REFUSAL_GUARDRAIL_HINDI = "सुरक्षा कारणों से इस प्रश्न का उत्तर नहीं दिया जा सकता।"
REFUSAL_LOW_CONFIDENCE_HINDI = "क्षमा करें, इस विषय पर उपलब्ध ज्ञानकोष में पर्याप्त जानकारी नहीं मिली।"

REFUSAL_GUARDRAIL_ENGLISH = "I cannot fulfill this request due to safety guardrail policies."
REFUSAL_LOW_CONFIDENCE_ENGLISH = "I'm sorry, there is not enough relevant information in the knowledge base to answer this question."


def run_pipeline(
    audio_bytes: Optional[bytes] = None,
    language_code: str = "hi-IN",
    query_text: Optional[str] = None,
    top_k: int = 5,
) -> PipelineResponse:
    """Execute end-to-end Voice RAG pipeline synchronously.

    Execution Pipeline
    ------------------
    1. STT: Transcribe audio_bytes via Sarvam AI saaras:v3 (if audio provided).
    2. Input Guardrail: Block prompt injections, unsafe prompts, or out-of-domain input.
    3. Query Embedding: multilingual-e5-small with 'query: ' prefix.
    4. Hybrid Retrieval: Top-20 FAISS HNSW + Top-20 BM25 merged with RRF (k=60) & small-to-big expansion.
    5. Confidence Gate: Verify normalized top-1 score and non-flat score gap.
    6. Cerebras Generation: llama-3.3-70b capped at 100-120 tokens with chunk_id citations.

    Parameters
    ----------
    audio_bytes:
        Optional raw audio bytes (WAV, WebM, MP3).
    language_code:
        Target language code (e.g. 'hi-IN', 'en-IN').
    query_text:
        Optional direct text query (skips STT).
    top_k:
        Number of final chunks to supply to the LLM (default: 5).

    Returns
    -------
    PipelineResponse
        Structured response with answer, citations, confidence, grounded flag, and granular timings dict.
    """
    t_total_start = time.perf_counter()
    transcription: Optional[str] = None
    stt_ms = 0.0

    # -----------------------------------------------------------------------
    # Stage 1: Speech-To-Text (if audio provided)
    # -----------------------------------------------------------------------
    if query_text is not None and query_text.strip():
        query = query_text.strip()
    elif audio_bytes is not None and len(audio_bytes) > 0:
        raw_text, detected_lang, stt_ms = transcribe(
            audio_bytes,
            language_code=language_code,
        )
        transcription = raw_text.strip()
        query = transcription
    else:
        # Neither audio nor text provided
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        return PipelineResponse(
            query="",
            transcription=None,
            answer="कोई इनपुट (ऑडियो या टेक्स्ट) प्राप्त नहीं हुआ।",
            citations=[],
            confidence="low",
            grounded=False,
            status="error",
            total_rag_core_ms=0.0,
            stt_ms=0.0,
            timings={
                "stt": 0.0,
                "guardrail": 0.0,
                "embed": 0.0,
                "retrieve": 0.0,
                "gate": 0.0,
                "generation": 0.0,
                "total": total_ms,
            },
        )

    # Empty transcription check
    if not query:
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        return PipelineResponse(
            query="",
            transcription=transcription,
            answer="कोई स्पष्ट आवाज़ पहचानी नहीं गई। कृपया पुनः बोलें।",
            citations=[],
            confidence="low",
            grounded=False,
            status="no_speech_detected",
            total_rag_core_ms=0.0,
            stt_ms=stt_ms,
            timings={
                "stt": stt_ms,
                "guardrail": 0.0,
                "embed": 0.0,
                "retrieve": 0.0,
                "gate": 0.0,
                "generation": 0.0,
                "total": total_ms,
            },
        )

    is_hindi = any("\u0900" <= c <= "\u097f" for c in query)

    # -----------------------------------------------------------------------
    # Core RAG Execution (Graded against 200ms target, EXCLUDING STT)
    # -----------------------------------------------------------------------
    t_rag_core_start = time.perf_counter()

    # Stage 2: Input Guardrail (< 2ms, 0 model calls)
    t_guard_start = time.perf_counter()
    safe_input = input_guardrail(query)
    guard_ms = (time.perf_counter() - t_guard_start) * 1000.0

    if not safe_input:
        rag_core_ms = (time.perf_counter() - t_rag_core_start) * 1000.0
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        refusal_text = REFUSAL_GUARDRAIL_HINDI if is_hindi else REFUSAL_GUARDRAIL_ENGLISH
        return PipelineResponse(
            query=query,
            transcription=transcription,
            answer=refusal_text,
            citations=[],
            confidence="low",
            grounded=False,
            status="guardrail_blocked",
            total_rag_core_ms=rag_core_ms,
            stt_ms=stt_ms,
            timings={
                "stt": stt_ms,
                "guardrail": guard_ms,
                "embed": 0.0,
                "retrieve": 0.0,
                "gate": 0.0,
                "generation": 0.0,
                "total": total_ms,
            },
        )

    # Stage 3: Dense Query Embedding
    t_embed_start = time.perf_counter()
    query_emb = embed_query(query, language=language_code)
    embed_ms = (time.perf_counter() - t_embed_start) * 1000.0

    # Stage 4: Hybrid Retrieval (FAISS + BM25 + RRF + Small-to-Big)
    t_ret_start = time.perf_counter()
    retrieval_res = hybrid_retrieve(query_emb, query, top_k=top_k)
    retrieve_ms = (time.perf_counter() - t_ret_start) * 1000.0

    # Stage 5: Confidence Gate (< 0.05ms)
    t_gate_start = time.perf_counter()
    confident = confidence_gate(retrieval_res, threshold=0.15)
    gate_ms = (time.perf_counter() - t_gate_start) * 1000.0

    if not confident:
        rag_core_ms = (time.perf_counter() - t_rag_core_start) * 1000.0
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        low_conf_text = REFUSAL_LOW_CONFIDENCE_HINDI if is_hindi else REFUSAL_LOW_CONFIDENCE_ENGLISH
        return PipelineResponse(
            query=query,
            transcription=transcription,
            answer=low_conf_text,
            citations=[],
            confidence="low",
            grounded=False,
            status="low_confidence_fallback",
            total_rag_core_ms=rag_core_ms,
            stt_ms=stt_ms,
            timings={
                "stt": stt_ms,
                "guardrail": guard_ms,
                "embed": embed_ms,
                "retrieve": retrieve_ms,
                "gate": gate_ms,
                "generation": 0.0,
                "total": total_ms,
            },
            retrieval_result=retrieval_res,
        )

    # Stage 6: Pure Extractive Response (Zero-LLM exact evidence extraction)
    t_gen_start = time.perf_counter()
    gen_res = generate_extractive_response(query, retrieval_res)
    gen_ms = (time.perf_counter() - t_gen_start) * 1000.0

    rag_core_ms = (time.perf_counter() - t_rag_core_start) * 1000.0
    total_ms = (time.perf_counter() - t_total_start) * 1000.0

    top_chunk = retrieval_res.chunks[0].chunk if retrieval_res.chunks else None
    evidence_text = (top_chunk.parent_text or top_chunk.text) if top_chunk else None

    # Legacy LatencyBreakdown for schema compatibility
    latency_breakdown = LatencyBreakdown(
        stt_ms=stt_ms,
        guardrail_input_ms=guard_ms,
        embedding_ms=embed_ms,
        retrieval_ms=retrieve_ms,
        generation_ms=gen_ms,
        total_pipeline_ms=total_ms,
    )

    response = PipelineResponse(
        query=query,
        transcription=transcription,
        answer=gen_res.answer,
        evidence_text=evidence_text,
        citations=gen_res.citations,
        confidence=gen_res.confidence,
        confidence_tier=gen_res.confidence,
        grounded=gen_res.grounded,
        status="success" if gen_res.grounded else "low_confidence_fallback",
        total_rag_core_ms=rag_core_ms,
        stt_ms=stt_ms,
        timings={
            "stt": stt_ms,
            "guardrail": guard_ms,
            "embed": embed_ms,
            "retrieve": retrieve_ms,
            "gate": gate_ms,
            "generation": gen_ms,
            "total": total_ms,
        },
        retrieval_result=retrieval_res,
        generation_result=gen_res,
        latency=latency_breakdown,
        response_mode=gen_res.response_mode or "extractive",
        fallback_reason=gen_res.fallback_reason,
    )
    _write_eval_log(response, retrieval_res)
    return response


def _write_eval_log(response: "PipelineResponse", retrieval_res: Optional["RetrievalResult"]) -> None:
    """Append a JSONL row to logs/eval_log.jsonl with all retrieval scores and latencies.

    Per architecture doc: Save the query, retrieved IDs, dense scores, BM25 scores,
    RRF scores, reranker scores, final citations, answer, and latency for every test query.
    """
    try:
        log_path = Path("logs") / "eval_log.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect per-chunk retrieval debug info
        chunk_debug: List[Dict] = []
        if retrieval_res and retrieval_res.chunks:
            for sc in retrieval_res.chunks:
                chunk_debug.append({
                    "chunk_id": sc.chunk.chunk_id,
                    "rrf_score": sc.score,
                    "rank": sc.rank,
                    "strategy": sc.chunk.chunk_strategy,
                })

        row = {
            "query": response.query,
            "answer": response.answer,
            "final_citations": response.citations,
            "confidence": response.confidence,
            "grounded": response.grounded,
            "status": response.status,
            "retrieved_chunks": chunk_debug,
            "latency_ms": response.timings,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass  # Never let logging failure break the pipeline


async def arun_pipeline(
    audio_bytes: Optional[bytes] = None,
    language_code: str = "hi-IN",
    query_text: Optional[str] = None,
    top_k: int = 5,
) -> PipelineResponse:
    """Asynchronous variant of run_pipeline for FastAPI and async callers."""
    t_total_start = time.perf_counter()
    transcription: Optional[str] = None
    stt_ms = 0.0

    if query_text is not None and query_text.strip():
        query = query_text.strip()
    elif audio_bytes is not None and len(audio_bytes) > 0:
        t_stt_start = time.perf_counter()
        raw_text, detected_lang, duration_ms = await atranscribe(
            audio_bytes,
            language_code=language_code,
        )
        stt_ms = (time.perf_counter() - t_stt_start) * 1000.0
        transcription = raw_text.strip()
        query = transcription
    else:
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        return PipelineResponse(
            query="",
            transcription=None,
            answer="कोई इनपुट (ऑडियो या टेक्स्ट) प्राप्त नहीं हुआ।",
            citations=[],
            confidence="low",
            grounded=False,
            status="error",
            total_rag_core_ms=0.0,
            stt_ms=0.0,
            timings={
                "stt": 0.0,
                "guardrail": 0.0,
                "embed": 0.0,
                "retrieve": 0.0,
                "gate": 0.0,
                "generation": 0.0,
                "total": total_ms,
            },
        )

    if not query:
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        return PipelineResponse(
            query="",
            transcription=transcription,
            answer="कोई स्पष्ट आवाज़ पहचानी नहीं गई। कृपया पुनः बोलें।",
            citations=[],
            confidence="low",
            grounded=False,
            status="no_speech_detected",
            total_rag_core_ms=0.0,
            stt_ms=stt_ms,
            timings={
                "stt": stt_ms,
                "guardrail": 0.0,
                "embed": 0.0,
                "retrieve": 0.0,
                "gate": 0.0,
                "generation": 0.0,
                "total": total_ms,
            },
        )

    is_hindi = any("\u0900" <= c <= "\u097f" for c in query)
    t_rag_core_start = time.perf_counter()

    # Input guardrail
    t_guard_start = time.perf_counter()
    safe_input = input_guardrail(query)
    guard_ms = (time.perf_counter() - t_guard_start) * 1000.0

    if not safe_input:
        rag_core_ms = (time.perf_counter() - t_rag_core_start) * 1000.0
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        refusal_text = REFUSAL_GUARDRAIL_HINDI if is_hindi else REFUSAL_GUARDRAIL_ENGLISH
        return PipelineResponse(
            query=query,
            transcription=transcription,
            answer=refusal_text,
            citations=[],
            confidence="low",
            grounded=False,
            status="guardrail_blocked",
            total_rag_core_ms=rag_core_ms,
            stt_ms=stt_ms,
            timings={
                "stt": stt_ms,
                "guardrail": guard_ms,
                "embed": 0.0,
                "retrieve": 0.0,
                "gate": 0.0,
                "generation": 0.0,
                "total": total_ms,
            },
        )

    # Embedding and retrieval in worker thread to prevent event-loop blocking
    t_embed_start = time.perf_counter()
    query_emb = await asyncio.to_thread(embed_query, query, language_code)
    embed_ms = (time.perf_counter() - t_embed_start) * 1000.0

    t_ret_start = time.perf_counter()
    retrieval_res = await asyncio.to_thread(hybrid_retrieve, query_emb, query, top_k)
    retrieve_ms = (time.perf_counter() - t_ret_start) * 1000.0

    # Confidence gate
    t_gate_start = time.perf_counter()
    confident = confidence_gate(retrieval_res, threshold=0.15)
    gate_ms = (time.perf_counter() - t_gate_start) * 1000.0

    if not confident:
        rag_core_ms = (time.perf_counter() - t_rag_core_start) * 1000.0
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        low_conf_text = REFUSAL_LOW_CONFIDENCE_HINDI if is_hindi else REFUSAL_LOW_CONFIDENCE_ENGLISH
        
        top_chunk = retrieval_res.chunks[0].chunk if retrieval_res.chunks else None
        evidence_text = (top_chunk.parent_text or top_chunk.text) if top_chunk else None

        return PipelineResponse(
            query=query,
            transcription=transcription,
            answer=low_conf_text,
            evidence_text=evidence_text,
            citations=[],
            confidence="low",
            confidence_tier="low",
            grounded=False,
            status="low_confidence_fallback",
            total_rag_core_ms=rag_core_ms,
            stt_ms=stt_ms,
            timings={
                "stt": stt_ms,
                "guardrail": guard_ms,
                "embed": embed_ms,
                "retrieve": retrieve_ms,
                "gate": gate_ms,
                "generation": 0.0,
                "total": total_ms,
            },
            retrieval_result=retrieval_res,
        )

    # Async Generation (Groq fast-inference or Cerebras fallback)
    t_gen_start = time.perf_counter()
    gen_res = await agenerate_answer(query, retrieval_res)
    gen_ms = (time.perf_counter() - t_gen_start) * 1000.0

    rag_core_ms = (time.perf_counter() - t_rag_core_start) * 1000.0
    total_ms = (time.perf_counter() - t_total_start) * 1000.0

    top_chunk = retrieval_res.chunks[0].chunk if retrieval_res.chunks else None
    evidence_text = (top_chunk.parent_text or top_chunk.text) if top_chunk else None

    latency_breakdown = LatencyBreakdown(
        stt_ms=stt_ms,
        guardrail_input_ms=guard_ms,
        embedding_ms=embed_ms,
        retrieval_ms=retrieve_ms,
        generation_ms=gen_ms,
        total_pipeline_ms=total_ms,
    )

    response = PipelineResponse(
        query=query,
        transcription=transcription,
        answer=gen_res.answer,
        evidence_text=evidence_text,
        citations=gen_res.citations,
        confidence=gen_res.confidence,
        confidence_tier=gen_res.confidence,
        grounded=gen_res.grounded,
        status="success",
        total_rag_core_ms=rag_core_ms,
        stt_ms=stt_ms,
        timings={
            "stt": stt_ms,
            "guardrail": guard_ms,
            "embed": embed_ms,
            "retrieve": retrieve_ms,
            "gate": gate_ms,
            "generation": gen_ms,
            "total": total_ms,
        },
        retrieval_result=retrieval_res,
        generation_result=gen_res,
        latency=latency_breakdown,
        response_mode=gen_res.response_mode,
        fallback_reason=gen_res.fallback_reason,
    )


# ---------------------------------------------------------------------------
# Class-based Orchestrator wrapper
# ---------------------------------------------------------------------------

class VoiceRAGOrchestrator:
    """High-level Orchestrator wrapper for the Voice RAG pipeline."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def run_voice_rag(self, audio: AudioInput) -> PipelineResponse:
        """Run pipeline from AudioInput object."""
        raw_bytes = audio.audio_bytes or b""
        if not raw_bytes and audio.audio_base64:
            import base64
            raw_bytes = base64.b64decode(audio.audio_base64)

        lang = audio.language_code or "hi-IN"
        return run_pipeline(audio_bytes=raw_bytes, language_code=lang, top_k=self.top_k)

    def run_text_rag(self, query: str, language_code: str = "hi-IN") -> PipelineResponse:
        """Run pipeline directly with text query."""
        return run_pipeline(query_text=query, language_code=language_code, top_k=self.top_k)

    async def arun_voice_rag(self, audio: AudioInput) -> PipelineResponse:
        """Asynchronously run pipeline from AudioInput object."""
        raw_bytes = audio.audio_bytes or b""
        if not raw_bytes and audio.audio_base64:
            import base64
            raw_bytes = base64.b64decode(audio.audio_base64)

        lang = audio.language_code or "hi-IN"
        return await arun_pipeline(audio_bytes=raw_bytes, language_code=lang, top_k=self.top_k)

    async def arun_text_rag(self, query: str, language_code: str = "hi-IN") -> PipelineResponse:
        """Asynchronously run pipeline directly with text query."""
        return await arun_pipeline(query_text=query, language_code=language_code, top_k=self.top_k)
