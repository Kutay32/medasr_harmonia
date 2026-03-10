"""Centralised configuration loaded from .env or environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# --- LLM backend -----------------------------------------------------------
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "openai")  # openai | ollama | huggingface | lmstudio

# OpenAI
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# LM Studio (OpenAI-compatible local server)
LMSTUDIO_BASE_URL: str = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:8080/v1")
LMSTUDIO_MODEL: str = os.getenv("LMSTUDIO_MODEL", "local-model")

# Ollama
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

# HuggingFace
HF_LLM_MODEL: str = os.getenv("HF_LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

# --- Embeddings ------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# --- ChromaDB ---------------------------------------------------------------
CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(_PROJECT_ROOT / "data" / "chroma_db"),
)

# --- Document ingestion -----------------------------------------------------
DOCS_DIR: str = os.getenv("DOCS_DIR", str(_PROJECT_ROOT / "data" / "docs"))
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
