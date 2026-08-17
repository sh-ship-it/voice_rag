"""Voice-enabled RAG Pipeline package."""

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
from pipeline.generate import (
    BaseGenerator,
    CerebrasGenerator,
    generate_answer,
    agenerate_answer,
)
from pipeline.stt import (
    BaseSTT,
    SarvamSTT,
    transcribe,
    atranscribe,
)
from pipeline.orchestrator import (
    VoiceRAGOrchestrator,
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
    # Generation
    "BaseGenerator",
    "CerebrasGenerator",
    "generate_answer",
    "agenerate_answer",
    # STT
    "BaseSTT",
    "SarvamSTT",
    "transcribe",
    "atranscribe",
    # Orchestrator
    "VoiceRAGOrchestrator",
    "run_pipeline",
    "arun_pipeline",
]


