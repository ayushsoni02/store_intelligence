FROM python:3.11-slim

# System deps for healthcheck curl
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install API-only Python deps (no torch/ultralytics — detection runs locally)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy source
COPY app/       ./app/
COPY pipeline/  ./pipeline/
COPY data/raw/  ./data/raw/

# data/ is mounted as a volume at runtime
RUN mkdir -p data/raw data/processed

EXPOSE 8000

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
