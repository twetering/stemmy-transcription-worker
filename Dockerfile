# RunPod Serverless: WhisperX transcription worker
# GPU required (L4 / RTX 4090)

FROM runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# FFmpeg for audio loading
RUN apt-get update -y && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Fix: whisperx 3.8+ requires transformers>=4.48 (Pipeline removed there)
# Pin transformers 4.36 for Pipeline; do NOT reinstall torchvision (breaks torch pairing)
RUN pip install --no-cache-dir "transformers==4.36.2"

COPY rp_handler.py /app/

CMD ["python", "-u", "rp_handler.py"]
