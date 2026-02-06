from __future__ import annotations

from langchain_ollama import ChatOllama


def create_llm(
    model: str = "qwen2.5:7b",
    base_url: str = "http://127.0.0.1:11434",
    temperature: float = 0.0,
) -> ChatOllama:
    """Create a configured Ollama chat model instance (model, base_url, temperature)."""
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )


__all__ = ["create_llm", "ChatOllama"]
