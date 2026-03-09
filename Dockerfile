# RunPod Serverless: WhisperX transcription worker
# GPU required (L4 / RTX 4090)

FROM runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# FFmpeg for audio loading
RUN apt-get update -y && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Install transformers first (compatible with whisperx/pyannote Pipeline import)
RUN pip install --no-cache-dir "transformers>=4.30.0,<4.46.0"

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY rp_handler.py /app/

CMD ["python", "-u", "rp_handler.py"]
