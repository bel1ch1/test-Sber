FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8:replace

WORKDIR /app

# System deps for common Python wheels / builds
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app

CMD ["python", "main.py"]
