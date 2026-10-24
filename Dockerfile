FROM python:3.13-slim

# CPU-only torch wheel: ~200 MB instead of several GB with CUDA.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch==2.14.1 --index-url ${TORCH_INDEX_URL}

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
# artifacts/ is mounted at runtime by docker-compose (train first).
ENV ARTIFACTS_DIR=/app/artifacts
EXPOSE 8000
USER nobody
CMD ["uvicorn", "--app-dir", "src", "app:app", "--host", "0.0.0.0", "--port", "8000"]
