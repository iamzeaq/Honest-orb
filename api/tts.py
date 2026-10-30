"""Honest Orb mouth: text -> WAV file via Meta MMS-TTS.

Uses facebook/mms-tts-eng (VitsModel) from Hugging Face Transformers,
running on CPU, for both English and Pidgin text. The model is loaded once
and reused. Output is mono 16-bit PCM WAV.
"""

from __future__ import annotations

import re
import time
import wave
from pathlib import Path

MODEL_ID = "facebook/mms-tts-eng"

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    from transformers import AutoTokenizer, VitsModel

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _model = VitsModel.from_pretrained(MODEL_ID)
    _model.eval()
    return _model, _tokenizer


def _chunks(text: str, limit: int = 400) -> list[str]:
    """Split long text on sentence boundaries for stable synthesis."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= limit:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                out.append(current)
            current = sentence
    if current:
        out.append(current)
    return out or [text]


def speak(text: str, out_path: str | None = None) -> str:
    """Synthesize text to a WAV file. Returns the file path."""
    import torch

    started = time.perf_counter()
    model, tokenizer = _load()
    sample_rate = int(model.config.sampling_rate)

    import numpy as np

    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for chunk in _chunks(text):
            inputs = tokenizer(chunk, return_tensors="pt")
            waveform = model(**inputs).waveform[0].cpu().numpy()
            pieces.append(waveform)
    audio = np.concatenate(pieces) if pieces else np.zeros(0)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)

    if out_path is None:
        import tempfile

        fd, out_path = tempfile.mkstemp(suffix=".wav")
        import os

        os.close(fd)
    out_file = str(out_path)
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(out_file, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())

    elapsed = time.perf_counter() - started
    print(f"TTS: synthesized {len(text)} chars in {elapsed:.2f}s -> {out_file}")
    return out_file
