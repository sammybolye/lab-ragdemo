FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install system dependencies (build-essential for some python packages if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# We install:
# - jupyterlab: The IDE
# - langchain, langchain-community: RAG framework
# - qdrant-client: Vector DB client
# - sentence-transformers: For local embeddings
# - pypdf: For reading PDFs
# - pandas: Data manipulation
# - httpx: Async HTTP client (often needed)
RUN pip install --no-cache-dir \
    jupyterlab \
    langchain \
    langchain-community \
    langchain-core \
    langchain-text-splitters \
    langchain-ollama \
    langchain-qdrant \
    langchain-huggingface \
    qdrant-client \
    sentence-transformers \
    pypdf \
    pandas \
    httpx \
    ipywidgets \
    torch \
    transformers \
    peft \
    trl \
    bitsandbytes \
    datasets \
    accelerate \
    scipy \
    unstructured \
    markdown

# Expose Jupyter port
EXPOSE 8888

# Start JupyterLab
# Copy Jupyter config
COPY jupyter_server_config.py /root/.jupyter/

# Start JupyterLab (config file will control auth)
CMD ["jupyter", "lab"]
