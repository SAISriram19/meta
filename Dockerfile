FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860

# Only lightweight system deps — the core env is pure python.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Core runtime deps only. sentence-transformers (90MB+ model) is optional;
# the memory subsystem falls back to a deterministic hash embedding if absent,
# which is fine for the server — agents use semantic retrieval via the KG + PPR.
COPY requirements-server.txt ./requirements-server.txt
RUN pip install -r requirements-server.txt
# OpenEnv SDK — required by env/openenv_compat.py (server entrypoint).
RUN pip install openenv

# Copy code (everything server-side needs at runtime).
COPY env ./env
COPY generator ./generator
COPY eval ./eval
COPY server ./server

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health').read()" || exit 1

CMD ["sh", "-c", "uvicorn server.openenv_main:app --host 0.0.0.0 --port ${PORT:-7860}"]
