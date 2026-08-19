"""Voice-enabled Pure Extractive RAG Pipeline package."""

from pipeline.config import Settings, get_settings
from pipeline.schemas import (
    AudioInput,
    Chunk,
    ChunkMetadata,
    GenerationResult,
    GuardrailResult,
    LatencyBreakdown,
    PipelineResponse,
    RetrievalResult,
    RetrievalStrategy,
    ScoredChunk,
)
from pipeline.chunking import (
    fixed_size_chunker,
    semantic_chunker,
    small_to_big_chunker,
    run_all_chunkers,
)
from pipeline.embed import (
    BaseEmbedder,
    SentenceTransformerEmbedder,
    embed_query,
    embed_passages,
    get_embedding_model,
)
from pipeline.retrieve import (
    BaseRetriever,
    FAISSDenseRetriever,
    BM25SparseRetriever,
    HybridRetriever,
    IndexRegistry,
    get_index_registry,
    hybrid_retrieve,
)
from pipeline.guardrails import (
    BaseGuardrail,
    SafetyGuardrails,
    input_guardrail,
    confidence_gate,
)
from pipeline.confidence import (
    compute_evidence_score,
    extract_confidence_features,
    ConfidenceResult,
    ConfidenceFeatures,
)
from pipeline.generate import (
    generate_extractive_response,
    generate_answer,
    agenerate_answer,
    select_best_evidence_sentence,
)
from pipeline.stt import (
    BaseSTT,
    SarvamSTT,
    transcribe,
    atranscribe,
)
from pipeline.orchestrator import (
    run_pipeline,
    arun_pipeline,
)

__all__ = [
    "Settings",
    "get_settings",
    # Schemas
    "AudioInput",
    "Chunk",
    "ChunkMetadata",
    "GenerationResult",
    "GuardrailResult",
    "LatencyBreakdown",
    "PipelineResponse",
    "RetrievalResult",
    "RetrievalStrategy",
    "ScoredChunk",
    # Chunking
    "fixed_size_chunker",
    "semantic_chunker",
    "small_to_big_chunker",
    "run_all_chunkers",
    # Embedder
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "embed_query",
    "embed_passages",
    "get_embedding_model",
    # Retrieval
    "BaseRetriever",
    "FAISSDenseRetriever",
    "BM25SparseRetriever",
    "HybridRetriever",
    "IndexRegistry",
    "get_index_registry",
    "hybrid_retrieve",
    # Guardrails
    "BaseGuardrail",
    "SafetyGuardrails",
    "input_guardrail",
    "confidence_gate",
    # Confidence
    "compute_evidence_score",
    "extract_confidence_features",
    "ConfidenceResult",
    "ConfidenceFeatures",
    # Extractive Generation
    "generate_extractive_response",
    "generate_answer",
    "agenerate_answer",
    "select_best_evidence_sentence",
    # STT
    "BaseSTT",
    "SarvamSTT",
    "transcribe",
    "atranscribe",
    # Orchestrator
    "run_pipeline",
    "arun_pipeline",
]
