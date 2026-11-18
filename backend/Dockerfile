FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (build-essential, poppler/mupdf deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for rapid deterministic dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml .
RUN uv pip install --system \
    fastapi \
    uvicorn \
    pydantic \
    pydantic-settings \
    aiohttp \
    aiosqlite \
    asyncpg \
    sqlalchemy \
    pymupdf \
    rank-bm25 \
    pillow \
    python-multipart

COPY backend ./backend
COPY eval ./eval

RUN mkdir -p /app/data/uploads /app/data/crops /app/data/states

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

