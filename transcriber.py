"""Unified transcriber interface — wraps MedASR (English) and Whisper (Turkish/multilingual)."""

import torch
from transformers import pipeline as hf_pipeline

_medasr_pipe = None
_whisper_pipe = None

MEDASR_MODEL_ID = "google/medasr"
WHISPER_MODEL_ID = "openai/whisper-large-v3"


def _get_medasr_pipe():
    """Lazy-load the MedASR pipeline."""
    global _medasr_pipe
    if _medasr_pipe is None:
        print("Loading MedASR model...")
        _medasr_pipe = hf_pipeline("automatic-speech-recognition", model=MEDASR_MODEL_ID)
        print("MedASR model loaded.")
    return _medasr_pipe


def _get_whisper_pipe():
    """Lazy-load the Whisper pipeline."""
    global _whisper_pipe
    if _whisper_pipe is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Loading Whisper model on {device}...")
        _whisper_pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL_ID,
            device=device,
            torch_dtype=torch_dtype,
        )
        print("Whisper model loaded.")
    return _whisper_pipe


def detect_language(audio_path: str) -> str:
    """Detect the language of an audio file using Whisper.

    Returns "en" or "tr" (or other ISO 639-1 code).
    """
    pipe = _get_whisper_pipe()
    result = pipe(
        audio_path,
        chunk_length_s=30,
        return_timestamps=True,
        generate_kwargs={"task": "transcribe"},
    )
    # Whisper puts detected language in the chunks metadata if available
    # Fallback: use a short transcription and heuristic
    text = result.get("text", "")
    # Simple heuristic: check for Turkish-specific characters
    turkish_chars = set("çğıöşüÇĞİÖŞÜ")
    if any(c in text for c in turkish_chars):
        return "tr"
    return "en"


def transcribe(
    audio_path: str,
    language: str = "auto",
) -> dict:
    """Transcribe audio and return {"text": ..., "language": ...}.

    Parameters
    ----------
    audio_path : str
        Path to the audio file.
    language : str
        "en" to force MedASR, "tr" to force Whisper Turkish, "auto" to auto-detect.

    Returns
    -------
    dict
        {"text": str, "language": str}
    """
    if language == "auto":
        # Use Whisper for detection first with a quick pass
        pipe = _get_whisper_pipe()
        result = pipe(
            audio_path,
            chunk_length_s=20,
            stride_length_s=2,
            return_timestamps=False,
            generate_kwargs={"task": "transcribe"},
        )
        text = result.get("text", "")
        turkish_chars = set("çğıöşüÇĞİÖŞÜ")
        detected = "tr" if any(c in text for c in turkish_chars) else "en"

        if detected == "en":
            # Re-transcribe with MedASR for better English medical accuracy
            medasr = _get_medasr_pipe()
            result = medasr(audio_path, chunk_length_s=20, stride_length_s=2)
            return {"text": result["text"], "language": "en"}
        else:
            return {"text": text, "language": "tr"}

    elif language == "en":
        pipe = _get_medasr_pipe()
        result = pipe(audio_path, chunk_length_s=20, stride_length_s=2)
        return {"text": result["text"], "language": "en"}

    elif language == "tr":
        pipe = _get_whisper_pipe()
        result = pipe(
            audio_path,
            chunk_length_s=20,
            stride_length_s=2,
            generate_kwargs={"task": "transcribe", "language": "turkish"},
        )
        return {"text": result["text"], "language": "tr"}

    else:
        raise ValueError(f"Unknown language '{language}'. Use 'auto', 'en', or 'tr'.")
