"""Configuration loading and validation for the Pure Extractive Voice RAG system."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Automatically load .env if present
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable override support."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # Speech-to-Text API Key (Sarvam AI saaras:v3)
    sarvam_api_key: Optional[str] = Field(default=None, alias="SARVAM_API_KEY")
    sarvam_base_url: str = Field(
        default="https://api.sarvam.ai/v1",
        alias="SARVAM_BASE_URL"
    )
    sarvam_language_code: str = Field(
        default="hi-IN",
        alias="SARVAM_LANGUAGE_CODE"
    )
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")

    # Response Policy (Pure Extractive)
    answer_mode: str = Field(
        default="extractive",
        alias="ANSWER_MODE",
        description="Response mode: 'extractive' (zero LLM calls, verbatim evidence extraction)"
    )
    max_evidence_sentences: int = Field(
        default=2,
        alias="MAX_EVIDENCE_SENTENCES"
    )
    enable_rerank: bool = Field(
        default=True,
        alias="ENABLE_RERANK"
    )

    # Embedding Configuration
    embedding_model_name: str = Field(
        default="intfloat/multilingual-e5-small",
        alias="EMBEDDING_MODEL_NAME"
    )

    # Directory Paths
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    index_dir: Path = Field(default=Path("./index"), alias="INDEX_DIR")

    # Retrieval Configurations
    top_k_dense: int = Field(default=50, alias="TOP_K_DENSE")
    top_k_sparse: int = Field(default=50, alias="TOP_K_SPARSE")
    top_k_final: int = Field(default=5, alias="TOP_K_FINAL")
    rrf_k: int = Field(default=60, alias="RRF_K")

    # Server Configurations
    server_host: str = Field(default="0.0.0.0", alias="SERVER_HOST")
    server_port: int = Field(default=8000, alias="SERVER_PORT")

    # Index build parameters
    hnsw_m: int = Field(default=16, alias="HNSW_M")
    hnsw_ef_construction: int = Field(default=100, alias="HNSW_EF_CONSTRUCTION")
    index_build_batch_size: int = Field(default=256, alias="INDEX_BUILD_BATCH_SIZE")
    dataset_name: str = Field(default="ai4bharat/MSMARCO-XI", alias="DATASET_NAME")
    dataset_config: str = Field(default="default", alias="DATASET_CONFIG")
    dataset_lang: str = Field(default="hin_Deva", alias="DATASET_LANG")
    dataset_rows: int = Field(default=3000, alias="DATASET_ROWS")

    def ensure_directories(self) -> None:
        """Ensure runtime directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton factory."""
    settings = Settings()
    return settings
