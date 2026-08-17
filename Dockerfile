FROM python:3.11-slim

# Install system audio and compiler dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install ultra-lightweight CPU-only PyTorch (saves 2.5 GB of disk space)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-built indices
COPY index/ ./index/
COPY data/ ./data/
COPY pipeline/ ./pipeline/
COPY ui/ ./ui/
COPY app/ ./app/
COPY .env .

# Set CPU threading environment variables for low-latency embedding
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Start high-performance FastAPI server
CMD ["uvicorn", "ui.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
