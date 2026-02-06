FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Устанавливаем базовые утилиты и Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Установка зависимостей проекта
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m venv /venv && \
    . /venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r /workspace/requirements.txt

ENV PATH="/venv/bin:${PATH}"

# Копируем исходники внутрь контейнера (Dev Containers всё равно будет монтировать volume,
# но это позволяет запускать образ и отдельно)
COPY . /workspace

CMD [ "bash" ]
