# Honest Orb on Hugging Face Spaces (Docker, free CPU).
#
# Python slim + system ffmpeg, CPU-only torch, then the app's own
# requirements. The image contains NO model weights and NO secrets: the
# TTS model (TTS_MODEL_ID) downloads from the Hugging Face Hub at
# container startup, using the HF_TOKEN secret only when the model is
# private. MODEL_API_KEY, HF_TOKEN and TTS_MODEL_ID are read from the
# environment (Space secrets) at runtime.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements.txt api/requirements.txt

# CPU-only torch first: the bare `torch` line in requirements.txt is then
# already satisfied, so pip never pulls the large CUDA wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r api/requirements.txt

COPY api/ api/
COPY web/ web/
COPY data/sections.json data/sections.json

EXPOSE 7860

ENV PORT=7860

CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
