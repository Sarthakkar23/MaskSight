FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY masksight ./masksight
RUN pip install --no-cache-dir ".[serve]"
# Pre-fetch backbone weights so the container starts offline.
RUN python -c "import torchvision; torchvision.models.get_model('wide_resnet50_2', weights='DEFAULT')"
COPY serve ./serve
COPY models ./models
ENV MASKSIGHT_MODELS=/app/models
EXPOSE 8000
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
