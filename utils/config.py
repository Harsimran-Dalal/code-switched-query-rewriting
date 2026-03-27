from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])

    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data")
    documents_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "documents")
    index_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1] / "data" / "index")

    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )

    top_k: int = Field(default=5, alias="TOP_K")
    chunk_size: int = Field(default=350, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=60, alias="CHUNK_OVERLAP")

    # ASR
    asr_backend: Literal["faster-whisper", "whisper"] = Field(default="faster-whisper", alias="ASR_BACKEND")
    whisper_model_size: str = Field(default="small", alias="WHISPER_MODEL_SIZE")
    asr_device: str = Field(default="cpu", alias="ASR_DEVICE")
    asr_compute_type: str = Field(default="int8", alias="ASR_COMPUTE_TYPE")

    # Baseline generator mode (extractive only in Phase 1).
    llm_provider: Literal["extractive"] = Field(default="extractive", alias="LLM_PROVIDER")


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.index_dir.mkdir(parents=True, exist_ok=True)
    s.documents_dir.mkdir(parents=True, exist_ok=True)
    return s

