# RunPod Serverless: faster-whisper large-v3-turbo
# CUDA 12.1 — stabiel op RunPod L4/RTX4090/A100
# Model wordt tijdens build gedownload → geen cold-start wachten op HuggingFace

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# HuggingFace cache → in image, niet opnieuw downloaden bij elke start
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models
WORKDIR /app

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3 /usr/bin/python \
    && pip install --no-cache-dir --upgrade pip

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model tijdens build (geen GPU nodig voor download)
# Zodat eerste request direct kan starten
RUN python3 -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Systran/faster-whisper-large-v3-turbo', local_dir='/app/models/faster-whisper-large-v3-turbo')"

COPY rp_handler.py /app/

CMD ["python", "-u", "rp_handler.py"]
