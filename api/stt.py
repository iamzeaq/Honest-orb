"""Honest Orb ears: audio file -> text via Muse Voice Transcribe.

Uses the file endpoint POST https://api.meta.ai/v1/asr/transcribe with
model muse-voice-transcribe-1.0 (see meta-model-cookbook 06_muse_voice).
Any input container (m4a, ogg, mp3, wav, webm, ...) is first converted to
mono 16-bit PCM WAV at 16 kHz with ffmpeg.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

TRANSCRIBE_URL = "https://api.meta.ai/v1/asr/transcribe"
MODEL = "muse-voice-transcribe-1.0"

# English bias plus legal + Pidgin vocabulary biasing.
LANGUAGE_BIAS = ["english"]
KEYWORDS = [
    "police",
    "warrant",
    "bail",
    "arrest",
    "station",
    "lawyer",
    "Legal Aid",
    "search",
    "checkpoint",
    "dem",
    "wetin",
    "abeg",
    "oga",
    "wan",
    "dey",
]

# Pidgin markers used for the language_hint heuristic.
_PIDGIN_MARKERS = {
    "dem",
    "dey",
    "wan",
    "wetin",
    "abeg",
    "una",
    "na",
    "abi",
    "oga",
    "don",
    "fit",
    "suppose",
    "carry",
    "go",
    "wey",
    "make",
    "e",
}


def _load_env_file() -> None:
    try:
        text = ENV_PATH.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "ffmpeg is not installed and imageio-ffmpeg is unavailable"
        ) from exc


class AudioTooShortError(ValueError):
    """Raised when the recording has no usable speech (empty/silent)."""


MIN_SPEECH_SECONDS = 0.3
MIN_INPUT_BYTES = 2000


def _convert_attempt(exe: str, src_path: str, out: str,
                     extra_input: list[str] | None = None) -> str | None:
    """One ffmpeg conversion attempt; returns stderr tail on failure."""
    cmd = [exe, "-y", "-v", "error"]
    if extra_input:
        cmd.extend(extra_input)
    cmd += [ "-i", src_path, "-ac", "1", "-ar", "16000",
             "-sample_fmt", "s16", out ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        return exc.stderr.decode()[:300]
    if not os.path.exists(out) or os.path.getsize(out) == 0:
        return "ffmpeg produced no output"
    return None


def _to_wav_16k(src: str | os.PathLike) -> tuple[str, bool]:
    """Convert any audio file to mono 16-bit PCM WAV at 16 kHz.

    Phone recorders emit streaming containers (Android: webm/Opus with
    unknown-length EBML; iPhone: mp4/AAC) that defeat a plain ffmpeg
    invocation, so this tries progressively more tolerant strategies.
    Returns (path, is_temp). Already-conforming WAV files are used as-is.
    """
    src_path = str(src)
    try:
        with wave.open(src_path, "rb") as w:
            if (
                w.getnchannels() == 1
                and w.getsampwidth() == 2
                and w.getframerate() == 16000
            ):
                return src_path, False
    except (wave.Error, OSError):
        pass
    try:
        size = os.path.getsize(src_path)
    except OSError:
        size = 0
    if size < MIN_INPUT_BYTES:
        raise AudioTooShortError(
            f"recording is empty ({size} bytes)")
    fd, out = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    exe = _ffmpeg_exe()
    errors: list[str] = []
    # 1. Plain conversion (clean files).
    # 2. Lenient streaming parse: ignore index, generate timestamps, probe
    #    deep (fixes phone MediaRecorder webm/mp4 with unknown lengths).
    for extra in (None, ["-fflags", "+ignidx+genpts", "-err_detect",
                         "ignore_err", "-analyzeduration", "20M",
                         "-probesize", "20M"]):
        err = _convert_attempt(exe, src_path, out, extra)
        if err is None:
            return out, True
        errors.append(err)
    os.path.exists(out) and os.remove(out)
    raise RuntimeError(
        f"ffmpeg conversion failed for {size}-byte input: {errors[-1]}"
    )


def _wav_seconds(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate() or 1)


def _guess_language_hint(text: str) -> str:
    words = set(re.findall(r"[a-z]+", text.lower()))
    hits = len(words & _PIDGIN_MARKERS - {"e", "go", "make", "na"})
    if hits >= 2 or ("dem" in words and "dey" in words):
        return "pcm"
    return "en"


def transcribe(audio_path: str | os.PathLike) -> dict:
    """Transcribe an audio file. Returns {text, language_hint, seconds}."""
    import requests

    _load_env_file()
    api_key = os.environ.get("MODEL_API_KEY", "")
    if not api_key:
        raise RuntimeError("MODEL_API_KEY is not set (expected in .env)")

    wav_path, is_temp = _to_wav_16k(audio_path)
    try:
        seconds = _wav_seconds(wav_path)
        if seconds < MIN_SPEECH_SECONDS:
            raise AudioTooShortError(
                f"recording has no usable speech ({seconds:.2f}s)")
        request_part = {
            "mode": "PUSH_TO_TALK",
            "model": MODEL,
            "audioEncoding": "WAV",
            "keywords": KEYWORDS,
            "languageBias": LANGUAGE_BIAS,
        }
        last_error: Exception | None = None
        data: dict | None = None
        for _ in range(3):  # initial attempt + up to 2 retries
            try:
                with open(wav_path, "rb") as audio:
                    response = requests.post(
                        TRANSCRIBE_URL,
                        headers={"Authorization": f"Bearer {api_key}"},
                        files={
                            "request": (
                                None,
                                json.dumps(request_part),
                                "application/json",
                            ),
                            "audio": (
                                os.path.basename(wav_path),
                                audio,
                                "audio/wav",
                            ),
                        },
                        timeout=120,
                    )
                if response.status_code >= 500:
                    last_error = RuntimeError(
                        f"transcribe HTTP {response.status_code}: "
                        f"{response.text[:200]}"
                    )
                    time.sleep(2)
                    continue
                response.raise_for_status()
                data = response.json()
                break
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                time.sleep(2)
        if data is None:
            assert last_error is not None
            raise last_error
        text = str(data.get("transcript", "") or "").strip()
        duration_ms = data.get("audioDurationMs")
        if duration_ms:
            seconds = float(duration_ms) / 1000.0
        return {
            "text": text,
            "language_hint": _guess_language_hint(text),
            "seconds": round(seconds, 2),
        }
    finally:
        if is_temp and os.path.exists(wav_path):
            os.remove(wav_path)
