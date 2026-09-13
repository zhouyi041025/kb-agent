FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    KB_INDEX_DIR=/app/.kb_index \
    KB_KNOWLEDGE_DIR=/app/data/knowledge

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY eval ./eval
COPY data ./data

RUN python scripts/build_index.py

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["python", "-m", "kb_agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
