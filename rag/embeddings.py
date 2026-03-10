"""Multilingual embedding model wrapper for RAG."""

from langchain_huggingface import HuggingFaceEmbeddings
from rag.config import EMBEDDING_MODEL


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return a multilingual sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
