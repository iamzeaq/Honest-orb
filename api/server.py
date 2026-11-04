"""Honest Orb server: one FastAPI app serving both the API and the web UI.

Endpoints:
    GET  /health            liveness check
    POST /api/ask_text      {"text"} -> brain + TTS answer
    POST /api/ask           audio upload (webm/ogg/m4a/wav) -> stt -> brain
                            -> tts, streamed NDJSON just like /api/ask_text_stream
                            (meta first, then one audio line per sentence);
                            ?format=json (or an in-process call, or an
                            Accept: application/json request) keeps the
                            legacy single-JSON answer as a fallback
    GET  /audio/<id>.wav    generated answer audio
    GET  /api/section/<id>  section text behind a citation chip
    GET  /                  the web UI (web/ static files)
"""

from __future__ import annotations

import base64
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

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api import tts  # noqa: E402
from api.brain import answer as brain_answer, converse as brain_converse  # noqa: E402
from api.pipeline import run as pipeline_run  # noqa: E402
from api.stt import AudioTooShortError  # noqa: E402
from api.stt import transcribe as stt_transcribe  # noqa: E402

AUDIO_OUT_DIR = ROOT / "data" / "audio_out"
# Create eagerly at import: the Modal image only bakes in
# data/sections.json, so this directory does not exist in the container
# until we make it (the lifespan below re-ensures it on startup).
AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
SECTIONS_PATH = ROOT / "data" / "sections.json"
WEB_DIR = ROOT / "web"

ALLOWED_SUFFIXES = {".webm", ".ogg", ".oga", ".m4a", ".mp4", ".wav", ".mp3", ".mpeg"}

# Bound how long an answer may take so one stalled upstream call (model API,
# transcription, synthesis) fails as a 504 instead of hanging the worker.
# Voice uploads allow longer: they include audio upload + transcription.
BRAIN_TIMEOUT_S = 180.0
STT_TIMEOUT_S = 240.0
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
    try:
        # Warm the TTS model once at startup (this downloads it on first
        # boot, so the private token is only needed at runtime, never at
        # image build time).
        tts._load()
    except Exception:
        # A private TTS_MODEL_ID without HF_TOKEN (or no network at boot)
        # must not take the whole app down: /health stays green and the
        # model loads lazily on the first synthesis instead.
        logger.warning("tts preload failed; loading lazily on first use",
                       exc_info=True)
        yield
        return
    try:
        # Pre-generate one short acknowledgement clip in the orb voice
        # ("Mm, let me check."). The browser plays it the moment speech
        # ends so the user hears something instantly while the brain runs.
        ack_path = AUDIO_OUT_DIR / "ack.wav"
        if not ack_path.is_file():
            tmp = tts.speak_sentence("Mm, let me check.", str(ack_path) + ".tmp")
            if tmp is not None:
                shutil.move(tmp, ack_path)
    except Exception:
        logger.warning("ack clip preload failed; continuing without it",
                       exc_info=True)
    yield


app = FastAPI(title="Honest Orb", lifespan=lifespan)


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
    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_id = uuid.uuid4().hex
    dest = AUDIO_OUT_DIR / f"{audio_id}.wav"
    shutil.move(src_path, dest)
    _prune_old_audio()
    return audio_id


def _clip_data_uri(path: str) -> str:
    """Read a WAV clip as a base64 data URI for inline delivery.

    Clips on disk live on the container that synthesized them, and a
    later ``GET /audio/...`` may land on a different container. The
    inline copy travels with the answer, so playback never depends on
    which container serves ``/audio/``.
    """
    with open(path, "rb") as fh:
        raw = base64.b64encode(fh.read()).decode("ascii")
    return f"data:audio/wav;base64,{raw}"


def _store_clip_with_b64(path: str) -> tuple[str, str]:
    """Store a clip for ``/audio/``; return (audio_id, data_uri).

    Bytes are read before the move, so the inline copy survives even if
    the stored file later prunes or lives on another container.
    """
    data_uri = _clip_data_uri(path)
    return _store_clip(path), data_uri


def _speak_to_urls(
    spoken: str, timings: dict
) -> tuple[str, list[str], list[str], str, list[str]]:
    """Synthesize one WAV per sentence; return (full_url, urls, sentences,
    full_b64, b64s).

    ``audio_url`` is the whole answer as one clip (previous clients keep
    working); ``audio_urls`` is the per-sentence queue so the browser can
    start playing sentence 1 while later sentences are still rendering.
    ``audio_b64``/``audio_b64s`` carry the same clips as base64 data
    URIs, so playback works even when ``/audio/`` lands on another
    container. ``timings["tts_first"]`` marks when the first sentence
    was ready.
    """
    import tempfile

    sentences = tts.split_sentences(spoken) or [spoken]
    started = time.perf_counter()
    first_s: dict[str, float] = {}
    tmpdir = tempfile.mkdtemp(prefix="orb_sent_")
    try:
        paths = tts.speak_sentences(
            spoken,
            out_dir=tmpdir,
            on_sentence=lambda i, _p: first_s.setdefault(
                "t", round(time.perf_counter() - started, 2)
            ),
        )
        timings["tts_first"] = first_s.get(
            "t", round(time.perf_counter() - started, 2)
        )
        if not paths:
            raise RuntimeError("all sentences failed synthesis")
        full_tmp = tempfile.mktemp(suffix=".wav")
        tts.concat_wav_files(paths, full_tmp)
        timings["tts"] = round(time.perf_counter() - started, 2)
        stored = [_store_clip_with_b64(p) for p in paths]
        full_id, full_b64 = _store_clip_with_b64(full_tmp)
        urls = [f"/audio/{audio_id}.wav" for audio_id, _b64 in stored]
        b64s = [b64 for _audio_id, b64 in stored]
        full_url = f"/audio/{full_id}.wav"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return full_url, urls, sentences, full_b64, b64s


class AskTextBody(BaseModel):
    text: str


class ChatBody(BaseModel):
    text: str
    session_id: str = "default"


# converse() modes -> legacy answer statuses (the web card and old clients
# speak the legacy vocabulary; the raw turn rides along as "turn").
MODE_TO_STATUS = {
    "chat": "chat",
    "clarify": "clarify",
    "legal": "answered",
    "emergency": "emergency",
    "not_covered": "not_sure",
    "redirect": "off_topic",
}


def _turn_to_legacy(turn: dict) -> dict:
    details = turn.get("details") or {}
    return {
        "language": turn.get("language", "en"),
        "status": MODE_TO_STATUS.get(turn.get("mode"), "not_sure"),
        "mode": turn.get("mode"),
        "right": details.get("right", ""),
        "say_this": details.get("what_to_say", ""),
        "do_this": details.get("what_to_do", ""),
        "citations": turn.get("citations", []),
        "spoken": turn.get("spoken", ""),
        "facts": turn.get("facts", {}),
        "turn": turn,
    }


def _ack_url() -> str | None:
    path = AUDIO_OUT_DIR / "ack.wav"
    return "/audio/ack.wav" if path.is_file() else None


TOO_SHORT_SPOKEN = "Sorry, I didn't catch that. Could you please say it again?"


def _too_short_answer() -> dict:
    """Legacy answer shape for an empty/too-short recording.

    Spoken by voice like any answer, so the user hears the prompt and the
    orb keeps listening afterwards instead of erroring out.
    """
    return {
        "language": "en",
        "status": "chat",
        "mode": "chat",
        "right": "",
        "say_this": "",
        "do_this": "",
        "citations": [],
        "spoken": TOO_SHORT_SPOKEN,
        "facts": {},
    }


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
        audio_url, audio_urls, sentences, audio_b64, audio_b64s = _speak_to_urls(
            ans["spoken"], timings
        )
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
        "audio_urls": audio_urls,
        "audio_b64": audio_b64,
        "audio_b64s": audio_b64s,
        "sentences": sentences,
        "timings": timings,
    }


@app.post("/api/ask_text_stream")
def ask_text_stream(body: AskTextBody):
    """Streaming twin of /api/ask_text: NDJSON lines over one response.

    First line: ``{"type": "meta", "transcript", "answer", "sentences",
    "timings"}`` as soon as the brain answer is ready. Then one
    ``{"type": "audio", "index", "audio_url", "sentence", "tts_s"}`` line
    per sentence, each sent the moment that sentence's audio is ready, so
    the browser can play sentence 1 while later sentences still render.
    """
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="text is too long")
    timings: dict[str, float] = {}

    started = time.perf_counter()
    try:
        ans = _run_timed(brain_answer, text, timeout_s=BRAIN_TIMEOUT_S)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("ask_text_stream brain failed")
        raise HTTPException(
            status_code=502, detail=f"could not answer the text: {_reason(exc)}"
        ) from exc
    timings["brain"] = round(time.perf_counter() - started, 2)

    sentences = tts.split_sentences(ans["spoken"]) or [ans["spoken"]]
    meta = {
        "type": "meta",
        "transcript": text,
        "answer": ans,
        "sentences": sentences,
        "timings": timings,
    }

    def gen():
        import tempfile

        yield json.dumps(meta) + "\n"
        tts_started = time.perf_counter()
        tmpdir = Path(tempfile.mkdtemp(prefix="orb_stream_"))
        try:
            for i, sentence in enumerate(sentences):
                try:
                    path = tts.speak_sentence(
                        sentence, str(tmpdir / f"s{i:02d}.wav")
                    )
                except Exception as exc:
                    logger.exception("ask_text_stream tts failed")
                    yield json.dumps(
                        {"type": "error", "index": i, "detail": _reason(exc)}
                    ) + "\n"
                    continue  # keep the rest of the reply playing
                if path is None:
                    continue
                try:
                    audio_id, audio_b64 = _store_clip_with_b64(path)
                except Exception as exc:
                    logger.exception("ask_text_stream store failed")
                    yield json.dumps(
                        {"type": "error", "index": i, "detail": _reason(exc)}
                    ) + "\n"
                    return
                yield json.dumps(
                    {
                        "type": "audio",
                        "index": i,
                        "audio_url": f"/audio/{audio_id}.wav",
                        "audio_b64": audio_b64,
                        "sentence": sentence,
                        "tts_s": round(time.perf_counter() - tts_started, 2),
                    }
                ) + "\n"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/chat")
def chat(body: ChatBody) -> dict:
    """Conversational turn: session memory + voice answer.

    Same answer audio shape as /api/ask_text, plus the mapped legacy
    "answer" (status/right/say_this/do_this for the card), the raw "turn"
    (mode/spoken/details/citations/facts), session "facts", and an
    "ack_url" short acknowledgement clip the browser plays instantly.
    """
    text = (body.text or "").strip()
    session_id = (body.session_id or "default").strip() or "default"
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="text is too long")
    timings: dict[str, float] = {}
    overall = time.perf_counter()

    started = time.perf_counter()
    try:
        turn = _run_timed(brain_converse, session_id, text,
                          timeout_s=BRAIN_TIMEOUT_S)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat brain failed")
        raise HTTPException(
            status_code=502, detail=f"could not answer the text: {_reason(exc)}"
        ) from exc
    timings["brain"] = round(time.perf_counter() - started, 2)
    ans = _turn_to_legacy(turn)

    try:
        audio_url, audio_urls, sentences, audio_b64, audio_b64s = _speak_to_urls(
            ans["spoken"], timings
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat tts failed")
        raise HTTPException(
            status_code=502, detail=f"could not speak the answer: {_reason(exc)}"
        ) from exc
    timings["total"] = round(time.perf_counter() - overall, 2)
    return {
        "transcript": text,
        "session_id": session_id,
        "answer": ans,
        "turn": turn,
        "facts": turn.get("facts", {}),
        "ack_url": _ack_url(),
        "audio_url": audio_url,
        "audio_urls": audio_urls,
        "audio_b64": audio_b64,
        "audio_b64s": audio_b64s,
        "sentences": sentences,
        "timings": timings,
    }


@app.post("/api/chat_stream")
def chat_stream(body: ChatBody):
    """Streaming twin of /api/chat: NDJSON meta line (answer + ack_url),
    then one audio line per sentence as each finishes rendering."""
    text = (body.text or "").strip()
    session_id = (body.session_id or "default").strip() or "default"
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="text is too long")
    timings: dict[str, float] = {}

    started = time.perf_counter()
    try:
        turn = _run_timed(brain_converse, session_id, text,
                          timeout_s=BRAIN_TIMEOUT_S)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("chat_stream brain failed")
        raise HTTPException(
            status_code=502, detail=f"could not answer the text: {_reason(exc)}"
        ) from exc
    timings["brain"] = round(time.perf_counter() - started, 2)
    ans = _turn_to_legacy(turn)

    sentences = tts.split_sentences(ans["spoken"]) or [ans["spoken"]]
    meta = {
        "type": "meta",
        "transcript": text,
        "session_id": session_id,
        "answer": ans,
        "turn": turn,
        "facts": turn.get("facts", {}),
        "ack_url": _ack_url(),
        "sentences": sentences,
        "timings": timings,
    }

    def gen():
        import tempfile

        yield json.dumps(meta) + "\n"
        tts_started = time.perf_counter()
        tmpdir = Path(tempfile.mkdtemp(prefix="orb_chat_stream_"))
        try:
            for i, sentence in enumerate(sentences):
                try:
                    path = tts.speak_sentence(
                        sentence, str(tmpdir / f"s{i:02d}.wav")
                    )
                except Exception as exc:
                    logger.exception("chat_stream tts failed")
                    yield json.dumps(
                        {"type": "error", "index": i, "detail": _reason(exc)}
                    ) + "\n"
                    continue  # keep the rest of the reply playing
                if path is None:
                    continue
                try:
                    audio_id, audio_b64 = _store_clip_with_b64(path)
                except Exception as exc:
                    logger.exception("chat_stream store failed")
                    yield json.dumps(
                        {"type": "error", "index": i, "detail": _reason(exc)}
                    ) + "\n"
                    return
                yield json.dumps(
                    {
                        "type": "audio",
                        "index": i,
                        "audio_url": f"/audio/{audio_id}.wav",
                        "audio_b64": audio_b64,
                        "sentence": sentence,
                        "tts_s": round(time.perf_counter() - tts_started, 2),
                    }
                ) + "\n"
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/ask")
async def ask(
    audio: UploadFile = File(...),
    request: Request = None,  # type: ignore[assignment]
    format: str | None = None,
    session_id: str | None = None,
):
    """Voice question: transcribe, answer, and stream the spoken answer.

    Same sentence-by-sentence NDJSON stream as ``/api/ask_text_stream``:
    a ``{"type": "meta", ...}`` line as soon as transcription + brain are
    done, then one ``{"type": "audio", ...}`` line per sentence, each sent
    the moment that sentence's audio is ready — so the browser plays the
    first sentence while later ones still render.

    JSON fallback: ``?format=json``, an ``Accept: application/json``
    request, or an in-process call without a request answers with the
    legacy single-JSON dict instead.
    """
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

        accept = ""
        if request is not None:
            accept = (request.headers.get("accept", "") or "").lower()
        wants_json = (
            format == "json"
            or request is None
            or ("application/json" in accept and "x-ndjson" not in accept)
        )
        if wants_json:
            if session_id:
                # Session voice turn: transcribe, converse with memory,
                # then speak — same JSON shape as the pipeline path.
                timings = {}
                started = time.perf_counter()
                too_short = False
                try:
                    stt = _run_timed(stt_transcribe, tmp_in,
                                     timeout_s=STT_TIMEOUT_S)
                    if not (stt.get("text") or "").strip():
                        too_short = True
                except AudioTooShortError as exc:
                    logger.info("voice turn too short: %s", exc)
                    too_short = True
                    stt = {"text": "", "language_hint": "en", "seconds": 0.0}
                except Exception as exc:
                    logger.exception("voice stt failed")
                    raise HTTPException(
                        status_code=502,
                        detail=f"could not transcribe the recording: {_reason(exc)}",
                    ) from exc
                timings["stt"] = round(time.perf_counter() - started, 2)
                if too_short:
                    turn = {"mode": "chat", "spoken": TOO_SHORT_SPOKEN,
                            "details": {}, "citations": [], "facts": {},
                            "summary": ""}
                    ans = _too_short_answer()
                if not too_short:
                    started = time.perf_counter()
                    try:
                        turn = _run_timed(brain_converse, session_id,
                                          stt["text"], timeout_s=BRAIN_TIMEOUT_S)
                    except Exception as exc:
                        logger.exception("voice converse failed")
                        raise HTTPException(
                            status_code=502,
                            detail=f"could not answer the voice question: {_reason(exc)}",
                        ) from exc
                    timings["brain"] = round(time.perf_counter() - started, 2)
                    ans = _turn_to_legacy(turn)
                try:
                    (audio_url, audio_urls, sentences,
                     audio_b64, audio_b64s) = _speak_to_urls(ans["spoken"], timings)
                except Exception as exc:
                    logger.exception("voice tts failed")
                    raise HTTPException(
                        status_code=502,
                        detail=f"could not speak the answer: {_reason(exc)}",
                    ) from exc
                timings["total"] = round(
                    sum(v for v in timings.values() if isinstance(v, (int, float))), 2)
                return {
                    "transcript": stt["text"],
                    "session_id": session_id,
                    "answer": ans,
                    "turn": turn,
                    "facts": turn.get("facts", {}),
                    "ack_url": _ack_url(),
                    "audio_url": audio_url,
                    "audio_urls": audio_urls,
                    "audio_b64": audio_b64,
                    "audio_b64s": audio_b64s,
                    "sentences": sentences,
                    "timings": timings,
                }
            try:
                result = _run_timed(
                    pipeline_run, tmp_in, timeout_s=PIPELINE_TIMEOUT_S
                )
            except AudioTooShortError as exc:
                # Empty/silent recording: answer by voice and keep the
                # conversation going instead of returning an error.
                logger.info("voice turn too short: %s", exc)
                timings: dict[str, float] = {}
                (audio_url, audio_urls, sentences,
                 audio_b64, audio_b64s) = _speak_to_urls(
                    TOO_SHORT_SPOKEN, timings)
                timings["total"] = round(
                    sum(v for v in timings.values()
                        if isinstance(v, (int, float))), 2)
                return {
                    "transcript": "",
                    "answer": {
                        "language": "en",
                        "status": "chat",
                        "right": "",
                        "say_this": "",
                        "do_this": "",
                        "cited": [],
                        "spoken": TOO_SHORT_SPOKEN,
                    },
                    "too_short": True,
                    "audio_url": audio_url,
                    "audio_urls": audio_urls,
                    "audio_b64": audio_b64,
                    "audio_b64s": audio_b64s,
                    "sentences": sentences,
                    "timings": timings,
                }
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("voice pipeline failed")
                raise HTTPException(
                    status_code=502,
                    detail=f"could not process the recording: {_reason(exc)}",
                ) from exc
            audio_id, audio_b64 = _store_clip_with_b64(result["audio_out"])
            audio_urls = []
            audio_b64s = []
            for p in result.get("sentence_outs") or []:
                try:
                    sent_id, sent_b64 = _store_clip_with_b64(p)
                except OSError:
                    logger.exception("ask sentence clip store failed")
                else:
                    audio_urls.append(f"/audio/{sent_id}.wav")
                    audio_b64s.append(sent_b64)
            return {
                "transcript": result["transcript"],
                "answer": result["answer"],
                "audio_url": f"/audio/{audio_id}.wav",
                "audio_urls": audio_urls,
                "audio_b64": audio_b64,
                "audio_b64s": audio_b64s,
                "sentences": result.get("sentences") or [],
                "timings": result["timings"],
            }

        # Streaming path: transcribe + answer while the upload still
        # exists, then stream one audio line per sentence as each sentence
        # finishes rendering (the generator only needs sentence text).
        timings: dict[str, float] = {}
        started = time.perf_counter()
        too_short = False
        try:
            stt = _run_timed(stt_transcribe, tmp_in, timeout_s=STT_TIMEOUT_S)
            if not (stt.get("text") or "").strip():
                too_short = True
        except AudioTooShortError as exc:
            logger.info("voice turn too short: %s", exc)
            too_short = True
            stt = {"text": "", "language_hint": "en", "seconds": 0.0}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("voice stt failed")
            raise HTTPException(
                status_code=502,
                detail=f"could not transcribe the recording: {_reason(exc)}",
            ) from exc
        timings["stt"] = round(time.perf_counter() - started, 2)
        if too_short:
            ans = _too_short_answer()
        else:
            started = time.perf_counter()
            try:
                if session_id:
                    turn = _run_timed(brain_converse, session_id, stt["text"],
                                      timeout_s=BRAIN_TIMEOUT_S)
                    ans = _turn_to_legacy(turn)
                else:
                    ans = _run_timed(
                        brain_answer, stt["text"], timeout_s=BRAIN_TIMEOUT_S
                    )
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("voice brain failed")
                raise HTTPException(
                    status_code=502,
                    detail=f"could not answer the voice question: {_reason(exc)}",
                ) from exc
            timings["brain"] = round(time.perf_counter() - started, 2)

        sentences = tts.split_sentences(ans["spoken"]) or [ans["spoken"]]
        meta = {
            "type": "meta",
            "transcript": stt["text"],
            "language_hint": stt.get("language_hint"),
            "answer": ans,
            "turn": ans.get("turn") if isinstance(ans, dict) else None,
            "facts": (ans.get("facts") or {}) if isinstance(ans, dict) else {},
            "session_id": session_id,
            "ack_url": _ack_url(),
            "too_short": too_short,
            "sentences": sentences,
            "timings": timings,
        }

        def gen():
            yield json.dumps(meta) + "\n"
            tts_started = time.perf_counter()
            tmpdir = Path(tempfile.mkdtemp(prefix="orb_ask_stream_"))
            try:
                for i, sentence in enumerate(sentences):
                    try:
                        path = tts.speak_sentence(
                            sentence, str(tmpdir / f"s{i:02d}.wav")
                        )
                    except Exception as exc:
                        logger.exception("ask stream tts failed")
                        yield json.dumps(
                            {"type": "error", "index": i, "detail": _reason(exc)}
                        ) + "\n"
                        continue  # keep the rest of the reply playing
                    if path is None:
                        continue
                    try:
                        audio_id, audio_b64 = _store_clip_with_b64(path)
                    except Exception as exc:
                        logger.exception("ask stream store failed")
                        yield json.dumps(
                            {"type": "error", "index": i, "detail": _reason(exc)}
                        ) + "\n"
                        return
                    yield json.dumps(
                        {
                            "type": "audio",
                            "index": i,
                            "audio_url": f"/audio/{audio_id}.wav",
                            "audio_b64": audio_b64,
                            "sentence": sentence,
                            "tts_s": round(time.perf_counter() - tts_started, 2),
                        }
                    ) + "\n"
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

        return StreamingResponse(gen(), media_type="application/x-ndjson")
    finally:
        try:
            Path(tmp_in).unlink()
        except OSError:
            pass


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
