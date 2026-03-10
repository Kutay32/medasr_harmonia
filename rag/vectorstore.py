"""ChromaDB vector store helpers."""

from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rag.config import CHROMA_PERSIST_DIR
from rag.embeddings import get_embedding_model

COLLECTION_NAME = "turkish_medical_docs"


def get_vectorstore() -> Chroma:
    """Return a persistent ChromaDB vector store instance."""
    Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=CHROMA_PERSIST_DIR,
    )


def similarity_search(query: str, k: int = 4) -> list:
    """Search the vector store and return the top-k relevant document chunks."""
    vs = get_vectorstore()
    return vs.similarity_search(query, k=k)


def add_document(text: str, metadata: dict = None):
    """Add a new document directly into the vector store.
    
    Useful for saving user-edited reports as future context.
    """
    vs = get_vectorstore()
    doc = Document(page_content=text, metadata=metadata or {"source": "user_feedback"})
    vs.add_documents([doc])
