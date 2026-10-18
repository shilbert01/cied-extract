# CiedOCR — pacemaker report OCR service
# Build:   docker build -t ciedocr .
# Run GPU: docker run --gpus all -v /path/to/pdfs:/data ciedocr /data/report.pdf
# Run CPU: docker run -v /path/to/pdfs:/data ciedocr /data/report.pdf --device cpu

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System dependencies for pdf2image (poppler) and PDF rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code (HTTP service: server.py → per-vendor extractors + Qwen3 VQA)
COPY server.py vqa.py ./
COPY extractors/ ./extractors/

# Ground-truth annotator (served on a separate port via docker-compose.api.yml)
COPY annotator/ ./annotator/

# HuggingFace cache survives container restarts when mounted
# (Qwen3-VL weights download here on the first Abbott request.)
ENV HF_HOME=/cache/huggingface
VOLUME ["/cache"]

# The image is the HTTP service. Per-vendor extractors run pdfplumber (CPU); the
# Abbott path lazy-loads Qwen3-VL onto the GPU on first use.
ENTRYPOINT ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
