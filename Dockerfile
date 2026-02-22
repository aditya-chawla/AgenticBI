# AgenticBI: Vizro frontend + agents. Run with docker-compose; DB and Ollama required.
FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2 and build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY src/ ./src/

# ChromaDB / schema index lives in src/agents/chroma_db_data (mount or build at runtime)
ENV PYTHONPATH=/app
EXPOSE 8050

# Default: run Vizro app. Override to run schema ingestion or tests.
CMD ["python", "-m", "app.main"]
