from __future__ import annotations

import os
from typing import Optional


def configure_langsmith(
    *,
    enabled: Optional[bool] = None,
    project: Optional[str] = None,
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> bool:
    """
    Enable LangSmith tracing for LangChain runs (LLM calls, chains, etc).

    This project uses a local LLM (Ollama) for inference, but LangSmith itself is a remote
    service. When enabled, prompts/outputs and run metadata are sent to LangSmith.

    Configuration is controlled via environment variables, but can be overridden via args:
    - LANGCHAIN_TRACING_V2=true|false
    - LANGCHAIN_API_KEY=...
    - LANGCHAIN_PROJECT=...
    - LANGCHAIN_ENDPOINT=...  (optional; defaults to LangSmith cloud)

    Convenience aliases (supported by this helper):
    - LANGSMITH_API_KEY, LANGSMITH_PROJECT, LANGSMITH_ENDPOINT

    Returns True if tracing is enabled, False otherwise.
    """

    env_enabled = os.getenv("LANGCHAIN_TRACING_V2")
    env_api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    env_project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT")
    env_endpoint = os.getenv("LANGCHAIN_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT")

    if enabled is None:
        # Auto-enable if an API key is present and user didn't explicitly disable tracing.
        if env_enabled is not None:
            enabled = env_enabled.strip().lower() in {"1", "true", "yes", "y", "on"}
        else:
            enabled = bool(env_api_key)

    if not enabled:
        return False

    resolved_api_key = api_key or env_api_key
    if not resolved_api_key:
        # Tracing cannot work without a key; treat as disabled.
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGCHAIN_API_KEY", resolved_api_key)

    resolved_project = project or env_project
    if resolved_project:
        os.environ.setdefault("LANGCHAIN_PROJECT", resolved_project)

    resolved_endpoint = endpoint or env_endpoint
    if resolved_endpoint:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", resolved_endpoint)

    return True

