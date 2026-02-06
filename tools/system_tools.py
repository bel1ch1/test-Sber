"""Tools for system load and Moscow time (Ubuntu 24.04 / Python 3.12+)."""

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
    """Returns current CPU load and RAM usage. Use for questions about system load or memory."""
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
    """Returns current date and time in Moscow (MSK). Use for questions about Moscow time."""
    tz = ZoneInfo("Europe/Moscow")
    now = datetime.now(tz)
    return now.strftime("Текущее время в Москве: %Y-%m-%d %H:%M:%S (MSK).")
