# syntax=docker/dockerfile:1.7-labs
FROM python:3.11.9-slim
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    CUDA_VISIBLE_DEVICES="" \
    DGLBACKEND="pytorch"

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY train_gnn.py .
CMD ["python","-u","train_gnn.py","--interim","/app/data/interim","--artifacts","/app/models/artifacts"]
