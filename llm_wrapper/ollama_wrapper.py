from __future__ import annotations

import os

from langchain_ollama import ChatOllama


def create_llm(
    model: str = "qwen2.5:7b",
    base_url: str | None = None,
    temperature: float = 0.0,
) -> ChatOllama:
    """Create a configured Ollama chat model instance (model, base_url, temperature)."""
    resolved_base_url = base_url or os.getenv("OLLAMA_BASE_URL") or "http://127.0.0.1:11434"
    return ChatOllama(
        model=model,
        base_url=resolved_base_url,
        temperature=temperature,
    )


__all__ = ["create_llm", "ChatOllama"]
