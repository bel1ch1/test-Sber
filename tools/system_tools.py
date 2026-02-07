"""System tools for the assistant.

Tools from this module are exposed to the model as "tools" and are intended for short,
deterministic observations about the runtime environment and time.

The call protocol used in this project is the plain-text ReAct block:
- `Action: <tool name>`
- `Action Input: <string>`

To make tool usage stable and reduce hallucinated tool names, each tool docstring includes
few-shot examples of choosing `Action` and filling `Action Input` (usually empty).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


@tool
def system_load(query: str = "") -> str:
    """Current CPU load and RAM usage.

    **When to use**
    - The user asks about system load, slowness, CPU/RAM, memory usage, "how much is free".
    - You need a quick "right now" snapshot (not a historical metric).

    **Arguments**
    - `query` (str, optional): ignored; kept for tool schema compatibility.

    **Returns**
    - A single line with CPU% and RAM (used/total + percent). Example:
      `CPU load: 12.3% | Memory used: 5.10 GiB / 31.86 GiB (16.0%).`

    **Few-shot (ReAct)**
    Example 1 (load question):
    Thought: I should measure the current system load.
    Action: system_load
    Action Input:
    Final Answer: I'll check the current load.

    Example 2 (not a load question — no tool needed):
    Thought: This is a general question; a tool is not required.
    Action: none
    Action Input:
    Final Answer: ...
    """
    if psutil is None:
        return (
            "Информация о загрузке системы недоступна: модуль 'psutil' не установлен. "
            "Установите его командой 'pip install psutil'."
        )

    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024**3)
    total_gb = mem.total / (1024**3)

    return (
        f"CPU load: {cpu_percent:.1f}% | "
        f"Memory used: {used_gb:.2f} GiB / {total_gb:.2f} GiB ({mem.percent:.1f}%)."
    )


@tool
def moscow_time(query: str = "") -> str:
    """Current date and time in Moscow (MSK).

    **When to use**
    - The user asks "what time is it in Moscow", "Moscow time now", "current MSK time".

    **Arguments**
    - `query` (str, optional): ignored; kept for tool schema compatibility.

    **Returns**
    - A single line with current date/time formatted as:
      `Текущее время в Москве: YYYY-MM-DD HH:MM:SS (MSK).`

    **Few-shot (ReAct)**
    Example (time question):
    Thought: I should get the current time in the Moscow timezone.
    Action: moscow_time
    Action Input:
    Final Answer: I'll check the current time in Moscow.
    """
    tz = ZoneInfo("Europe/Moscow")
    now = datetime.now(tz)
    return now.strftime("Текущее время в Москве: %Y-%m-%d %H:%M:%S (MSK).")
