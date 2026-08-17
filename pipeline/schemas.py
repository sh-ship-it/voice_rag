"""Pydantic data schemas passed between pipeline stages."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalStrategy(str, Enum):
    """Retrieval method used to fetch chunks."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


class ChunkMetadata(BaseModel):
    """Metadata attributes associated with a document chunk."""
    source: Optional[str] = Field(default=None, description="Origin document identifier or filename")
    section: Optional[str] = Field(default=None, description="Document section or heading")
    page_number: Optional[int] = Field(default=None, description="Page number if applicable")
    token_count: Optional[int] = Field(default=None, description="Estimated token count of the chunk")
    extra: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional metadata")


class Chunk(BaseModel):
    """Representation of a discrete chunk of text extracted from source documents."""
    chunk_id: str = Field(description="Unique identifier for the chunk")
    doc_id: str = Field(description="Parent document identifier")
    text: str = Field(description="Raw text content of the chunk")
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata, description="Associated chunk metadata")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding representation")

    # Corpus & strategy provenance
    source_passage_id: Optional[str] = Field(default=None, description="ID of the source passage this chunk was derived from")
    language: Optional[str] = Field(default=None, description="BCP-47 language code of the chunk text (e.g. 'hi', 'en')")
    chunk_strategy: Optional[str] = Field(default=None, description="Chunking strategy that produced this chunk (fixed_size | semantic | small_to_big)")
    token_count: Optional[int] = Field(default=None, description="Approximate whitespace-token count of chunk text")
    parent_text: Optional[str] = Field(default=None, description="Full source passage text; populated by small_to_big strategy for context expansion")


class ScoredChunk(BaseModel):
    """A chunk scored and ranked during retrieval."""
    chunk: Chunk = Field(description="The underlying document chunk")
    score: float = Field(description="Relevance or similarity score")
    rank: int = Field(description="Rank position in the retrieval result set (1-indexed)")
    retrieval_strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.HYBRID,
        description="Strategy that fetched or scored this chunk"
    )


class RetrievalResult(BaseModel):
    """Structured output from the retrieval stage."""
    query: str = Field(description="Search query string used for retrieval")
    chunks: List[ScoredChunk] = Field(default_factory=list, description="Ranked list of scored chunks")
    strategy_used: RetrievalStrategy = Field(
        default=RetrievalStrategy.HYBRID,
        description="Retrieval strategy executed"
    )
    total_candidates_evaluated: int = Field(
        default=0,
        description="Number of candidate chunks evaluated across indexes"
    )
    latency_ms: float = Field(default=0.0, description="Retrieval latency in milliseconds")


class GuardrailResult(BaseModel):
    """Validation and moderation result from input/output guardrails."""
    passed: bool = Field(default=True, description="Whether guardrail checks passed")
    flagged_categories: List[str] = Field(
        default_factory=list,
        description="Violated guardrail categories (e.g., prompt_injection, toxicity, pii)"
    )
    sanitized_text: Optional[str] = Field(
        default=None,
        description="Sanitized or redacted text if modifications were necessary"
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Explanation if the request was blocked"
    )


from typing import Any, Dict, List, Literal, Optional


class GenerationResult(BaseModel):
    """Structured output from the LLM generation stage."""
    answer: str = Field(default="", description="Final generated answer/response from LLM")
    citations: List[str] = Field(default_factory=list, description="List of cited chunk_ids supporting the answer")
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Model confidence assessment based on context grounding"
    )
    grounded: bool = Field(default=True, description="Whether the answer is grounded in provided retrieved context")
    generation_ms: float = Field(default=0.0, description="Full generation call duration in milliseconds")
    ttft_ms: Optional[float] = Field(default=None, description="Time-to-first-token in milliseconds if streaming")
    model_name: str = Field(default="llama-3.3-70b", description="LLM model identifier used for generation")
    prompt_tokens: Optional[int] = Field(default=None, description="Number of tokens in prompt")
    completion_tokens: Optional[int] = Field(default=None, description="Number of tokens in completion")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens consumed")
    latency_ms: float = Field(default=0.0, description="Generation latency in milliseconds")
    finish_reason: Optional[str] = Field(default=None, description="Generation termination reason")

    @property
    def generated_text(self) -> str:
        """Backward compatibility alias for answer."""
        return self.answer



class LatencyBreakdown(BaseModel):
    """Granular latency measurements across pipeline stages in milliseconds."""
    stt_ms: float = Field(default=0.0, description="Speech-To-Text transcription latency")
    guardrail_input_ms: float = Field(default=0.0, description="Input guardrails latency")
    embedding_ms: float = Field(default=0.0, description="Query embedding generation latency")
    retrieval_ms: float = Field(default=0.0, description="FAISS/BM25 retrieval latency")
    generation_ms: float = Field(default=0.0, description="LLM generation latency")
    guardrail_output_ms: float = Field(default=0.0, description="Output guardrails latency")
    total_pipeline_ms: float = Field(default=0.0, description="End-to-end processing latency")


class AudioInput(BaseModel):
    """Audio container for voice-enabled queries."""
    audio_base64: Optional[str] = Field(default=None, description="Base64-encoded audio payload")
    audio_bytes: Optional[bytes] = Field(default=None, description="Raw audio byte stream")
    content_type: str = Field(default="audio/wav", description="MIME type of the audio (e.g. audio/wav, audio/webm)")
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    language_code: Optional[str] = Field(default="en-IN", description="Expected spoken language code")


class PipelineResponse(BaseModel):
    """Top-level unified response returned by the voice-enabled RAG orchestrator."""
    query: str = Field(description="Original user query (transcribed or provided as text)")
    transcription: Optional[str] = Field(
        default=None,
        description="Speech-to-text transcript if input was audio"
    )
    answer: str = Field(description="Generated response answer text")
    citations: List[str] = Field(
        default_factory=list,
        description="List of cited chunk_ids supporting the answer"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence assessment based on retrieval grounding"
    )
    grounded: bool = Field(
        default=True,
        description="Whether the answer is grounded in provided retrieved context"
    )
    status: str = Field(
        default="success",
        description="Execution status: 'success' | 'guardrail_blocked' | 'low_confidence_fallback' | 'error'"
    )

    # First-class latency metrics (200ms target evaluation)
    total_rag_core_ms: float = Field(
        default=0.0,
        description="Core RAG latency excluding STT (guardrail + embed + retrieve + gate + generation) in ms"
    )
    stt_ms: float = Field(
        default=0.0,
        description="Speech-To-Text transcription latency in ms (separate for transparency)"
    )
    timings: Dict[str, float] = Field(
        default_factory=dict,
        description="Granular per-stage timings in ms: {'stt': ..., 'guardrail': ..., 'embed': ..., 'retrieve': ..., 'gate': ..., 'generation': ..., 'total': ...}"
    )

    retrieval_result: Optional[RetrievalResult] = Field(
        default=None,
        description="Retrieved chunks and context used for generation"
    )
    generation_result: Optional[GenerationResult] = Field(
        default=None,
        description="LLM generation details and token metrics"
    )
    latency: LatencyBreakdown = Field(
        default_factory=LatencyBreakdown,
        description="Legacy latency breakdown model"
    )
    guardrails: Optional[GuardrailResult] = Field(
        default=None,
        description="Guardrail evaluation details"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of pipeline response generation"
    )
