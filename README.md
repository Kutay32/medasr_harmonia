# MedASR – Medical ASR & Turkish Report Generator

Local implementation of [MedASR](https://huggingface.co/google/medasr) by Google Health, extended with **bilingual ASR** (English + Turkish) and **RAG-powered Turkish medical report generation**.

## Features

- **English medical ASR** via Google MedASR
- **Turkish ASR** via OpenAI Whisper (large-v3) with auto-language-detection
- **Turkish medical report generation** using RAG (Retrieval-Augmented Generation) over a Turkish medical knowledge base
- **Pluggable LLM backend** — OpenAI, Ollama, LM Studio, or HuggingFace models
- **Gradio web UI** with language toggle and report generation
- **CLI** with `--lang`, `--report`, and `--llm-backend` flags

## Prerequisites

1. **Hugging Face account** – create one at https://huggingface.co/join
2. **Accept model terms** – visit https://huggingface.co/google/medasr and accept the usage conditions
3. **Hugging Face token** – generate a `read` token at https://huggingface.co/settings/tokens and log in:

```bash
huggingface-cli login
```

4. **Python 3.12 recommended** (required for `kenlm` if using the language-model mode)
5. **NVIDIA GPU strongly recommended** (CPU works but is slow)
6. **LLM API key** (if using OpenAI) or **Ollama** installed locally

## Installation

```powershell
# 1. Create a virtual environment (uv auto-selects Python 3.12)
uv venv .venv

# 2. Install core dependencies
uv pip install -r requirements.txt --python .venv\Scripts\python.exe

# 3. Activate the venv
.venv\Scripts\Activate.ps1   # PowerShell
# or
.venv\Scripts\activate.bat  # CMD

# 4. Configure LLM backend
copy .env.example .env
# Edit .env with your API keys / preferences
```

> **Language model mode (`pipeline_lm`) is not supported on Windows** due to `kenlm` build failures
> with Python 3.13 and MSVC. It requires Linux/macOS with Python 3.12.
> The other two modes (`pipeline` and `direct`) work fully on Windows.

## RAG Setup

### 1. Add Turkish medical documents

Place `.txt`, `.pdf`, or `.docx` files in `data/docs/`. Three sample files are included:
- `sample_radyoloji_raporu.txt` — sample Turkish radiology report
- `tibbi_terimler_sozlugu.txt` — English-Turkish medical terminology glossary
- `rapor_sablonlari.txt` — Turkish radiology report templates

### 2. Index documents into the vector store

```bash
python -m rag.ingest
```

This chunks the documents, embeds them with a multilingual model, and stores them in a local ChromaDB database.

### 3. Configure LLM backend

Edit `.env` to select your backend:

| Backend | Required env vars |
|---|---|
| OpenAI | `LLM_BACKEND=openai`, `OPENAI_API_KEY=sk-...` |
| Ollama | `LLM_BACKEND=ollama`, `OLLAMA_MODEL=llama3` |
| LM Studio | `LLM_BACKEND=lmstudio`, `LMSTUDIO_BASE_URL=...` |
| HuggingFace | `LLM_BACKEND=huggingface`, `HF_LLM_MODEL=...` |

## Usage

### Web UI (recommended)

```bash
python app.py
```

Open http://127.0.0.1:7860 — features:
- **Language selector** (Auto / English / Turkish)
- **Transcription** with WER evaluation
- **Turkish report generation** with LLM backend selector

### CLI — Basic transcription

```bash
python medasr_inference.py --mode pipeline
```

### CLI — Turkish audio transcription

```bash
python medasr_inference.py --lang tr --audio path/to/turkish_audio.wav
```

### CLI — Auto-detect language + generate Turkish report

```bash
python medasr_inference.py --lang auto --report --audio path/to/audio.wav
```

### CLI — Specify LLM backend

```bash
python medasr_inference.py --lang auto --report --llm-backend ollama
```

### CLI — All original English modes

```bash
python medasr_inference.py --mode all
```

### Custom audio / transcript

```powershell
python medasr_inference.py --mode pipeline `
    --audio path/to/audio.wav `
    --transcript "your reference transcript here"
```

If `--audio` is omitted, the sample audio is downloaded from the Hugging Face Hub.

## File overview

| File | Description |
|---|---|
| `app.py` | Gradio web UI — transcription + Turkish report generation |
| `medasr_inference.py` | CLI entry point — all inference modes + report generation |
| `transcriber.py` | Unified ASR interface (MedASR + Whisper) |
| `utils.py` | `normalize` / `evaluate` helpers (WER + colored diff) |
| `lm_decoder.py` | `LasrCtcBeamSearchDecoder` + `beam_search_pipe` for n-gram LM |
| `rag/config.py` | Centralised configuration from `.env` |
| `rag/embeddings.py` | Multilingual embedding model wrapper |
| `rag/vectorstore.py` | ChromaDB vector store helpers |
| `rag/llm_backend.py` | Pluggable LLM factory (OpenAI / Ollama / HuggingFace) |
| `rag/report_generator.py` | RAG-powered Turkish medical report generator |
| `rag/ingest.py` | Document ingestion script |
| `data/docs/` | Turkish medical documents for RAG knowledge base |
| `requirements.txt` | Python dependencies |
| `requirements-lm.txt` | Optional LM deps – `kenlm` + `pyctcdecode` (Linux/macOS only) |
| `.env.example` | Configuration template |

## Output

Each mode prints:
- The hypothesis text (`HYP: ...`)
- Word Error Rate with insertion / deletion / substitution counts
- A colored diff against the reference transcript (red = deletion, green = insertion)

When `--report` is used, additionally outputs a full Turkish medical report.
