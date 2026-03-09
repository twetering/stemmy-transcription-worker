# RunPod Serverless: faster-whisper only – geen WhisperX, geen PyTorch
# ~5GB i.p.v. 15GB

FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Python, ffmpeg, pkg-config (voor PyAV/av)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    ffmpeg \
    pkg-config libavformat-dev libavcodec-dev libavutil-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf python3 /usr/bin/python && pip install --no-cache-dir --upgrade pip

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY rp_handler.py /app/

CMD ["python", "-u", "rp_handler.py"]
