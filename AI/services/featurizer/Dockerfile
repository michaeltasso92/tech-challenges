# syntax=docker/dockerfile:1.7-labs
FROM python:3.11.9-slim
WORKDIR /app
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
COPY feature_text.py .
CMD ["python","feature_text.py","--help"]
