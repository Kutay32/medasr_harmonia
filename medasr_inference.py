"""
MedASR Inference Script
=======================
Implements the three inference modes from the quick_start_with_hugging_face notebook:
  1. Pipeline API (simple)
  2. Pipeline API with n-gram language model (improved quality)
  3. Direct model usage (full control)

Usage:
    python medasr_inference.py --mode pipeline
    python medasr_inference.py --mode pipeline_lm
    python medasr_inference.py --mode direct
    python medasr_inference.py --mode pipeline --audio path/to/your/audio.wav
"""

import argparse
import os

import huggingface_hub
import librosa
import torch
from transformers import AutoModelForCTC, AutoProcessor, pipeline

from utils import evaluate
from transcriber import transcribe as unified_transcribe

MODEL_ID = "google/medasr"

SAMPLE_TRANSCRIPT = (
    "Exam type CT chest PE protocol period. Indication 54 year old female, "
    "shortness of breath, evaluate for PE period. Technique standard protocol period. "
    "Findings colon. Pulmonary vasculature colon. The main PA is patent period. "
    "There are filling defects in the segmental branches of the right lower lobe comma "
    "compatible with acute PE period. No saddle embolus period. Lungs colon. "
    "No pneumothorax period. Small bilateral effusions comma right greater than left period. "
    "New paragraph. Impression colon Acute segmental PE right lower lobe period."
)


def get_sample_audio() -> str:
    """Download the sample audio from Hugging Face Hub and return its local path."""
    print("Downloading sample audio from Hugging Face Hub...")
    audio_path = huggingface_hub.hf_hub_download(MODEL_ID, "test_audio.wav")
    print(f"Audio saved to: {audio_path}")
    return audio_path


def run_pipeline(audio_path: str, ref_text: str) -> None:
    """Run inference using the Hugging Face pipeline API."""
    print("\n--- Pipeline API inference ---")
    pipe = pipeline("automatic-speech-recognition", model=MODEL_ID)
    result = pipe(audio_path, chunk_length_s=20, stride_length_s=2)
    print(result)
    evaluate(ref_text=ref_text, hyp_text=result["text"])


def run_pipeline_with_lm(audio_path: str, ref_text: str) -> None:
    """Run inference using the pipeline API with the n-gram language model."""
    print("\n--- Pipeline API + language model inference ---")
    try:
        from lm_decoder import beam_search_pipe
    except ImportError as e:
        print(f"Could not import lm_decoder: {e}")
        print("Make sure kenlm and pyctcdecode are installed (see requirements.txt).")
        return

    lm_path = huggingface_hub.hf_hub_download(MODEL_ID, filename="lm_6.kenlm")
    pipe_with_lm = beam_search_pipe(MODEL_ID, lm_path)
    result = pipe_with_lm(
        audio_path,
        chunk_length_s=20,
        stride_length_s=2,
        decoder_kwargs=dict(beam_width=8),
    )
    evaluate(ref_text=ref_text, hyp_text=result["text"])


def run_direct(audio_path: str, ref_text: str) -> None:
    """Run inference by using the model directly (full control over pre/post-processing)."""
    print("\n--- Direct model inference ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(MODEL_ID).to(device)

    speech, sample_rate = librosa.load(audio_path, sr=16000)

    inputs = processor(speech, sampling_rate=sample_rate)
    inputs = inputs.to(device)

    outputs = model.generate(**inputs)
    decoded_text = processor.batch_decode(outputs)[0]
    evaluate(ref_text=ref_text, hyp_text=decoded_text)


def run_turkish_report(transcript: str, llm_backend: str | None = None) -> None:
    """Generate a Turkish medical report from the given transcript using RAG."""
    print("\n--- Turkish Report Generation (RAG) ---")
    try:
        from rag.report_generator import generate_turkish_report
        report = generate_turkish_report(
            transcript=transcript,
            language="auto",
            llm_backend=llm_backend,
        )
        print("\n" + report)
    except Exception as e:
        print(f"Error generating Turkish report: {e}")


def main():
    parser = argparse.ArgumentParser(description="MedASR inference demo")
    parser.add_argument(
        "--mode",
        choices=["pipeline", "pipeline_lm", "direct", "all"],
        default="pipeline",
        help="Inference mode to run (default: pipeline)",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Path to audio file. If not provided, the sample audio is downloaded.",
    )
    parser.add_argument(
        "--transcript",
        default=None,
        help="Reference transcript for WER evaluation. Uses built-in sample if omitted.",
    )
    parser.add_argument(
        "--lang",
        choices=["auto", "en", "tr"],
        default="auto",
        help="Audio language: auto-detect, English, or Turkish (default: auto)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate a Turkish medical report from the transcription using RAG",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["openai", "ollama", "lmstudio", "huggingface"],
        default=None,
        help="LLM backend for report generation (overrides .env setting)",
    )
    args = parser.parse_args()

    audio_path = args.audio if args.audio else get_sample_audio()
    ref_text = args.transcript if args.transcript else SAMPLE_TRANSCRIPT

    # If --lang is tr or auto, use unified transcriber for language-aware ASR
    if args.lang in ("tr", "auto") and args.lang != "en":
        print(f"\n--- Unified transcription (lang={args.lang}) ---")
        result = unified_transcribe(audio_path, language=args.lang)
        print(f"Detected language: {result['language']}")
        print(f"Transcript: {result['text']}")
        if args.report:
            run_turkish_report(result["text"], llm_backend=args.llm_backend)
        if ref_text:
            evaluate(ref_text=ref_text, hyp_text=result["text"])
        return

    # Original English-only modes
    if args.mode == "pipeline" or args.mode == "all":
        run_pipeline(audio_path, ref_text)
    if args.mode == "pipeline_lm" or args.mode == "all":
        run_pipeline_with_lm(audio_path, ref_text)
    if args.mode == "direct" or args.mode == "all":
        run_direct(audio_path, ref_text)

    # Generate Turkish report from last pipeline result if requested
    if args.report:
        pipe_result = pipeline("automatic-speech-recognition", model=MODEL_ID)
        result = pipe_result(audio_path, chunk_length_s=20, stride_length_s=2)
        run_turkish_report(result["text"], llm_backend=args.llm_backend)


if __name__ == "__main__":
    main()
