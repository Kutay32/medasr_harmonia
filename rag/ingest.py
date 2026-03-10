"""Document ingestion script — reads docs from data/docs/, chunks, embeds, stores in ChromaDB."""

import sys
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import DOCS_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from rag.vectorstore import get_vectorstore


def load_documents(docs_dir: str | None = None) -> list:
    """Load .txt, .pdf, and .docx files from the documents directory."""
    docs_path = Path(docs_dir or DOCS_DIR)
    if not docs_path.exists():
        print(f"Documents directory not found: {docs_path}")
        return []

    all_docs = []

    # Text files
    txt_loader = DirectoryLoader(
        str(docs_path), glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    try:
        all_docs.extend(txt_loader.load())
    except Exception as e:
        print(f"Warning loading .txt files: {e}")

    # PDF files
    pdf_loader = DirectoryLoader(
        str(docs_path), glob="**/*.pdf", loader_cls=PyPDFLoader,
    )
    try:
        all_docs.extend(pdf_loader.load())
    except Exception as e:
        print(f"Warning loading .pdf files: {e}")

    # DOCX files
    docx_loader = DirectoryLoader(
        str(docs_path), glob="**/*.docx", loader_cls=Docx2txtLoader,
    )
    try:
        all_docs.extend(docx_loader.load())
    except Exception as e:
        print(f"Warning loading .docx files: {e}")

    return all_docs


def ingest(docs_dir: str | None = None) -> int:
    """Main ingestion pipeline: load → chunk → embed → store. Returns chunk count."""
    docs = load_documents(docs_dir)
    if not docs:
        print("No documents found. Add files to data/docs/ and try again.")
        return 0

    print(f"Loaded {len(docs)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    vs = get_vectorstore()
    vs.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks into ChromaDB.")
    return len(chunks)


if __name__ == "__main__":
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else None
    ingest(docs_dir)
