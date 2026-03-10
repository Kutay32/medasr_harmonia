"""Pluggable LLM backend factory — supports OpenAI, Ollama, and HuggingFace."""

from rag.config import (
    LLM_BACKEND,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MODEL,
    HF_LLM_MODEL,
)


def get_llm(backend: str | None = None):
    """Return a LangChain-compatible LLM based on the configured backend.

    Parameters
    ----------
    backend : str, optional
        Override the backend from config.  One of "openai", "ollama", "huggingface".
    """
    backend = (backend or LLM_BACKEND).lower().strip()

    if backend == "openai":
        from langchain_openai import ChatOpenAI

        if not OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file or environment."
            )
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.3,
            max_tokens=1000,
        )

    elif backend == "lmstudio":
        from langchain_openai import ChatOpenAI
        from rag.config import LMSTUDIO_BASE_URL, LMSTUDIO_MODEL

        return ChatOpenAI(
            base_url=LMSTUDIO_BASE_URL,
            model=LMSTUDIO_MODEL,
            api_key="lm-studio",  # API key is required but any string works
            temperature=0.3,
            max_tokens=1000,
        )

    elif backend == "ollama":
        from langchain_ollama import ChatOllama
        from rag.config import OLLAMA_BASE_URL, OLLAMA_MODEL

        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
        )

    elif backend == "huggingface":
        from langchain_huggingface import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

        tokenizer = AutoTokenizer.from_pretrained(HF_LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            HF_LLM_MODEL,
            device_map="auto",
            torch_dtype="auto",
        )
        pipe = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            temperature=0.3,
            do_sample=True,
        )
        return HuggingFacePipeline(pipeline=pipe)

    else:
        raise ValueError(
            f"Unknown LLM_BACKEND '{backend}'. Choose from: openai, ollama, huggingface"
        )
