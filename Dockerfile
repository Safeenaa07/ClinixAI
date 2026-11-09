# Build:  docker build -t clinix-ai .
# Run:    docker run -p 8000:8000 -v clinix-data:/app/backend/data clinix-ai
# Add  --build-arg WITH_MODEL=1  to include torch + the real X-ray model (large image).
FROM python:3.12-slim
ARG WITH_MODEL=0
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt backend/requirements-model.txt /app/backend/
RUN pip install --no-cache-dir -r backend/requirements.txt && \
    if [ "$WITH_MODEL" = "1" ]; then \
      pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir torchxrayvision; fi
COPY backend /app/backend
COPY evaluation /app/evaluation
WORKDIR /app/backend
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
