from __future__ import annotations

import copy
from typing import Any, Iterable


def sanitize_text(value: Any) -> str:
    """
    Make text safe for UTF-8 encoding / JSON transport.

    Fixes rare cases where strings contain Unicode surrogate code points
    (U+D800..U+DFFF). Such characters cannot be encoded to UTF-8 and may
    crash HTTP/JSON libraries with errors like:
      "'utf-8' codec can't encode character '\\udcd1': surrogates not allowed"
    """

    text = value if isinstance(value, str) else str(value)

    # Replace any surrogate code points with the Unicode replacement character.
    if any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
        text = "".join("\uFFFD" if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in text)

    # Defensive: ensure text is round-trippable as UTF-8.
    # (After surrogate cleanup it should always succeed.)
    return text.encode("utf-8", errors="replace").decode("utf-8")


def sanitize_obj(value: Any) -> Any:
    """Recursively sanitize strings inside common container types."""
    if value is None:
        return None
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {sanitize_obj(k): sanitize_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_obj(v) for v in value]
    if isinstance(value, tuple):
        return tuple(sanitize_obj(v) for v in value)
    if isinstance(value, set):
        return {sanitize_obj(v) for v in value}
    return value


def sanitize_messages(messages: Iterable[Any]) -> list[Any]:
    """
    Best-effort sanitize of LangChain message objects before sending to LLM.
    We sanitize `.content` and common metadata fields if present.
    """
    sanitized: list[Any] = []
    for msg in messages:
        try:
            # LangChain messages are often pydantic models with `model_copy`.
            if hasattr(msg, "model_copy"):
                new_msg = msg.model_copy(deep=True)  # type: ignore[attr-defined]
            else:
                new_msg = copy.deepcopy(msg)

            if hasattr(new_msg, "content"):
                try:
                    new_msg.content = sanitize_obj(getattr(new_msg, "content"))
                except Exception:
                    pass

            for attr in ("additional_kwargs", "response_metadata"):
                if hasattr(new_msg, attr):
                    try:
                        setattr(new_msg, attr, sanitize_obj(getattr(new_msg, attr)))
                    except Exception:
                        pass

            sanitized.append(new_msg)
        except Exception:
            sanitized.append(msg)
    return sanitized
