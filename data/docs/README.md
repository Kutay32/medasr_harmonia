# Turkish Medical Documents for RAG Knowledge Base

Place your Turkish medical documents in this directory. Supported formats: `.txt`, `.pdf`, `.docx`

## What to Collect

1. **Radiology report templates** — Turkish radiology report examples (BT, MR, USG raporları)
2. **Medical terminology glossary** — English-Turkish medical term mappings
3. **ICD-10 Turkish translations** — Diagnosis code descriptions in Turkish
4. **Sample clinical reports** — De-identified Turkish clinical notes
5. **Turkish medical textbook excerpts** — Anatomy, pathology terminology

## Indexing

After adding documents, run:

```bash
python -m rag.ingest
```

This will chunk, embed, and store them in the local ChromaDB vector store.
