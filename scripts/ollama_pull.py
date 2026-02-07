from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


def _env_models() -> list[str]:
    raw = os.getenv("OLLAMA_PULL_MODELS", "").strip()
    if not raw:
        return []
    models = []
    for part in raw.split(","):
        m = part.strip()
        if m:
            models.append(m)
    # de-dup while keeping order
    seen: set[str] = set()
    ordered: list[str] = []
    for m in models:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def _http_json(method: str, url: str, payload: dict | None = None, timeout_s: int = 30) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body) if body else {}


def wait_for_ollama(base_url: str, *, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_err: str | None = None
    while time.time() < deadline:
        try:
            _http_json("GET", f"{base_url}/api/tags", timeout_s=5)
            print("Ollama: готово.")
            return
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(1)
    raise TimeoutError(f"Ollama не стал готов за {timeout_s}s. Последняя ошибка: {last_err}")


def pull_model(base_url: str, model: str, *, timeout_s: int) -> None:
    print(f"Pull model: {model}")
    # stream=false => one response after completion
    try:
        result = _http_json(
            "POST",
            f"{base_url}/api/pull",
            payload={"name": model, "stream": False},
            timeout_s=timeout_s,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore") if hasattr(exc, "read") else ""
        raise RuntimeError(f"HTTP {exc.code} при pull {model}: {body}".strip()) from exc
    status = str(result.get("status", "")).strip()
    if status:
        print(f"  status: {status}")


def main() -> int:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
    models = _env_models()
    timeout_s = int(os.getenv("OLLAMA_PULL_TIMEOUT_S", "3600"))  # per-model
    wait_timeout_s = int(os.getenv("OLLAMA_WAIT_TIMEOUT_S", "300"))

    if not models:
        print("OLLAMA_PULL_MODELS пуст — пропускаю скачивание моделей.")
        return 0

    print(f"Ollama base_url: {base_url}")
    wait_for_ollama(base_url, timeout_s=wait_timeout_s)

    for model in models:
        pull_model(base_url, model, timeout_s=timeout_s)

    print("Все модели скачаны.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

