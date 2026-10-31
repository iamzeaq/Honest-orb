"""Honest Orb server: one FastAPI app serving both the API and the web UI.

Endpoints:
    GET  /health            liveness check
    POST /api/ask_text      {"text"} -> brain + TTS answer
    POST /api/ask           audio upload (webm/ogg/m4a/wav) -> stt -> brain -> tts
    GET  /audio/<id>.wav    generated answer audio
    GET  /api/section/<id>  section text behind a citation chip
    GET  /                  the web UI (web/ static files)
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import tts  # noqa: E402
from api.brain import answer as brain_answer  # noqa: E402
from api.pipeline import run as pipeline_run  # noqa: E402

AUDIO_OUT_DIR = ROOT / "data" / "audio_out"
SECTIONS_PATH = ROOT / "data" / "sections.json"
WEB_DIR = ROOT / "web"

ALLOWED_SUFFIXES = {".webm", ".ogg", ".oga", ".m4a", ".mp4", ".wav", ".mp3", ".mpeg"}

# Bound how long an answer may take so one stalled upstream call (model API,
# transcription, synthesis) fails as a 504 instead of hanging the worker.
# Voice uploads allow longer: they include audio upload + transcription.
BRAIN_TIMEOUT_S = 180.0
PIPELINE_TIMEOUT_S = 420.0

_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="orb")

logger = logging.getLogger("honest_orb")


def _run_timed(fn, *args, timeout_s: float):
    try:
        return _pool.submit(fn, *args).result(timeout=timeout_s)
    except FutureTimeout as exc:
        raise HTTPException(
            status_code=504, detail="answer took too long, try again"
        ) from exc


def _reason(exc: Exception) -> str:
    """Short, safe-to-return reason for a failed answer (no credentials)."""
    text = str(exc).strip().replace("\n", " ")
    return (text[:300] or type(exc).__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    tts._load()  # load the TTS model once at startup
    yield


app = FastAPI(title="Honest Orb")


def _load_sections() -> list[dict]:
    with open(SECTIONS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


SECTIONS: list[dict] = _load_sections()
SECTIONS_BY_ID: dict[str, dict] = {s["id"]: s for s in SECTIONS}


def _prune_old_audio(max_age_s: float = 2 * 3600, keep: int = 200) -> None:
    """Best-effort cleanup of old generated clips so the disk stays bounded."""
    try:
        wavs = sorted(
            AUDIO_OUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime
        )
    except OSError:
        return
    now = time.time()
    for path in wavs[:-keep]:
        try:
            if now - path.stat().st_mtime > max_age_s:
                path.unlink()
        except OSError:
            pass


def _store_clip(src_path: str) -> str:
    """Move a generated WAV into the served dir; return its audio id."""
    audio_id = uuid.uuid4().hex
    dest = AUDIO_OUT_DIR / f"{audio_id}.wav"
    shutil.move(src_path, dest)
    _prune_old_audio()
    return audio_id


def _speak_to_url(spoken: str, timings: dict, key: str = "tts") -> str:
    started = time.perf_counter()
    tmp_path = tts.speak(spoken)
    timings[key] = round(time.perf_counter() - started, 2)
    audio_id = _store_clip(tmp_path)
    return f"/audio/{audio_id}.wav"


class AskTextBody(BaseModel):
    text: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ask_text")
def ask_text(body: AskTextBody) -> dict:
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="text is too long")
    timings: dict[str, float] = {}
    overall = time.perf_counter()

    started = time.perf_counter()
    try:
        ans = _run_timed(brain_answer, text, timeout_s=BRAIN_TIMEOUT_S)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ask_text brain failed")
        raise HTTPException(
            status_code=502, detail=f"could not answer the text: {_reason(exc)}"
        ) from exc
    timings["brain"] = round(time.perf_counter() - started, 2)

    try:
        audio_url = _speak_to_url(ans["spoken"], timings)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ask_text tts failed")
        raise HTTPException(
            status_code=502, detail=f"could not speak the answer: {_reason(exc)}"
        ) from exc
    timings["total"] = round(time.perf_counter() - overall, 2)
    return {
        "transcript": text,
        "answer": ans,
        "audio_url": audio_url,
        "timings": timings,
    }


@app.post("/api/ask")
async def ask(audio: UploadFile = File(...)) -> dict:
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        # Fall back to the declared content type for extensionless uploads.
        ctype = (audio.content_type or "").lower()
        guess = {
            "audio/webm": ".webm",
            "audio/ogg": ".ogg",
            "audio/m4a": ".m4a",
            "audio/x-m4a": ".m4a",
            "audio/mp4": ".m4a",
            "audio/wav": ".wav",
            "audio/x-wav": ".wav",
            "audio/mpeg": ".mp3",
        }.get(ctype.split(";")[0].strip())
        if guess is None:
            raise HTTPException(
                status_code=400,
                detail="audio must be one of: webm, ogg, m4a, wav, mp3",
            )
        suffix = guess
    fd, tmp_in = tempfile.mkstemp(suffix=suffix)
    try:
        with open(fd, "wb") as fh:
            shutil.copyfileobj(audio.file, fh)
        try:
            result = _run_timed(
                pipeline_run, tmp_in, timeout_s=PIPELINE_TIMEOUT_S
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("voice pipeline failed")
            raise HTTPException(
                status_code=502,
                detail=f"could not process the recording: {_reason(exc)}",
            ) from exc
    finally:
        try:
            Path(tmp_in).unlink()
        except OSError:
            pass
    audio_id = _store_clip(result["audio_out"])
    return {
        "transcript": result["transcript"],
        "answer": result["answer"],
        "audio_url": f"/audio/{audio_id}.wav",
        "timings": result["timings"],
    }


@app.get("/audio/{audio_id}.wav")
def get_audio(audio_id: str) -> FileResponse:
    if not audio_id or len(audio_id) > 64 or not audio_id.replace("_", "").isalnum():
        raise HTTPException(status_code=404, detail="not found")
    path = AUDIO_OUT_DIR / f"{audio_id}.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/section/{section_id}")
def get_section(section_id: str) -> dict:
    section = SECTIONS_BY_ID.get(section_id)
    if section is None:
        raise HTTPException(status_code=404, detail="unknown section id")
    return {
        "id": section["id"],
        "label": f"{section['source']}, s.{section['section']}",
        "title": section["title"],
        "text": section["text"],
    }


app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
