"""Configuration loading and validation for the Voice RAG system."""

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

    # API Keys
    sarvam_api_key: Optional[str] = Field(default=None, alias="SARVAM_API_KEY")
    cerebras_api_key: Optional[str] = Field(default=None, alias="CEREBRAS_API_KEY")
    hf_token: Optional[str] = Field(default=None, alias="HF_TOKEN")

    # Pipeline Model Configurations
    embedding_model_name: str = Field(
        default="intfloat/multilingual-e5-small",
        alias="EMBEDDING_MODEL_NAME"
    )
    llm_provider: str = Field(
        default="cerebras",
        alias="LLM_PROVIDER"
    )
    cerebras_model: str = Field(
        default="llama-3.3-70b",
        alias="CEREBRAS_MODEL"
    )
    cerebras_base_url: str = Field(
        default="https://api.cerebras.ai/v1",
        alias="CEREBRAS_BASE_URL"
    )
    max_tokens: int = Field(
        default=120,
        alias="MAX_TOKENS"
    )
    sarvam_llm_model: str = Field(
        default="sarvam-105b-conversations",
        alias="SARVAM_LLM_MODEL"
    )
    sarvam_base_url: str = Field(
        default="https://api.sarvam.ai/v1",
        alias="SARVAM_BASE_URL"
    )
    sarvam_language_code: str = Field(
        default="hi-IN",
        alias="SARVAM_LANGUAGE_CODE"
    )

    # Directory Paths
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")
    index_dir: Path = Field(default=Path("./index"), alias="INDEX_DIR")

    # Retrieval Configurations
    top_k_dense: int = Field(default=10, alias="TOP_K_DENSE")
    top_k_sparse: int = Field(default=10, alias="TOP_K_SPARSE")
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
    dataset_lang: str = Field(default="hin_Deva", alias="DATASET_LANG")  # target_lang filter value
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
