# syntax=docker/dockerfile:1.7-labs
FROM python:3.11.9-slim
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
COPY app ./app
ENV PYTHONPATH=/app
CMD ["uvicorn","app.api:app","--host","0.0.0.0","--port","8000"]
