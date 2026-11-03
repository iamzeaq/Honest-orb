"""Regression tests for POST /api/ask with real browser-style recordings.

Uses only the standard library (``python -m unittest api.test_ask_upload``).
The fixture ``data/audio/test_browser.webm`` is a genuine WebM/Opus file
(the container Chrome's MediaRecorder produces), encoded from
``data/audio/test_en.wav``. No test here needs the network: the pipeline
is stubbed at the server boundary, so these run offline and fast.
"""

import asyncio
import io
import os
import sys
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import HTTPException, UploadFile  # noqa: E402

import api.server as srv  # noqa: E402
from api.stt import _to_wav_16k, _wav_seconds  # noqa: E402

FIXTURE = ROOT / "data" / "audio" / "test_browser.webm"


def _upload(filename="question.webm", data=b"RIFF....",
            content_type="audio/webm;codecs=opus"):
    return UploadFile(
        file=io.BytesIO(data),
        filename=filename,
        headers={"content-type": content_type},
    )


class AskUploadTest(unittest.TestCase):
    def test_webm_opus_converts_to_16k_wav(self):
        """Chrome's WebM/Opus must convert to the 16k mono WAV STT needs."""
        self.assertTrue(FIXTURE.is_file(), f"missing fixture {FIXTURE}")
        wav_path, is_temp = _to_wav_16k(str(FIXTURE))
        try:
            with wave.open(wav_path, "rb") as w:
                self.assertEqual(w.getnchannels(), 1)
                self.assertEqual(w.getsampwidth(), 2)
                self.assertEqual(w.getframerate(), 16000)
            self.assertGreater(_wav_seconds(wav_path), 3.0)
        finally:
            if is_temp and os.path.exists(wav_path):
                os.remove(wav_path)

    def test_ask_rejects_bad_upload(self):
        async def go():
            with self.assertRaises(HTTPException) as ctx:
                await srv.ask(_upload(filename="evil.exe",
                                      content_type="application/x-msdownload"))
            self.assertEqual(ctx.exception.status_code, 400)
        asyncio.run(go())

    def test_ask_accepts_webm_upload(self):
        """The multipart shape browsers send must reach the pipeline."""
        seen = {}

        def fake_run(path):
            import shutil
            import tempfile

            seen["path"] = path
            self.assertTrue(path.endswith(".webm"), path)
            self.assertTrue(os.path.getsize(path) > 1000)
            fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            shutil.copy(str(ROOT / "data" / "audio" / "test_en.wav"), tmp_wav)
            seen["tmp_wav"] = tmp_wav
            return {"transcript": "hello orb",
                    "answer": {"status": "answered"},
                    "audio_out": tmp_wav,
                    "timings": {}}

        old, srv.pipeline_run = srv.pipeline_run, fake_run
        try:
            async def go():
                return await srv.ask(
                    _upload(data=FIXTURE.read_bytes()))
            data = asyncio.run(go())
        finally:
            srv.pipeline_run = old
        self.assertEqual(data["transcript"], "hello orb")
        self.assertTrue(data["audio_url"].startswith("/audio/"))
        self.assertIn("path", seen)
        stored = ROOT / "data" / "audio_out" / (
            data["audio_url"].split("/")[-1])
        self.assertTrue(stored.is_file())
        stored.unlink()  # the clip is a test artefact, not a real answer

    def test_ask_pipeline_failure_is_502_with_reason(self):
        """An STT/brain/TTS blowup must not surface as an opaque 500."""

        def boom(path):
            raise RuntimeError("transcribe HTTP 500: upstream exploded")

        old, srv.pipeline_run = srv.pipeline_run, boom
        try:
            async def go():
                await srv.ask(_upload(data=FIXTURE.read_bytes()))
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(go())
        finally:
            srv.pipeline_run = old
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("transcribe", ctx.exception.detail)

    def test_ask_text_brain_failure_is_502_with_reason(self):
        def boom(question):
            raise RuntimeError("model API overloaded")

        old, srv.brain_answer = srv.brain_answer, boom
        try:
            with self.assertRaises(HTTPException) as ctx:
                srv.ask_text(srv.AskTextBody(text="Can police search me?"))
        finally:
            srv.brain_answer = old
        self.assertEqual(ctx.exception.status_code, 502)
        self.assertIn("overloaded", ctx.exception.detail)


def _ndjson_request(accept="application/x-ndjson"):
    from starlette.requests import Request

    return Request(
        {
            "type": "http",
            "method": "POST",
            "headers": [(b"accept", accept.encode("latin-1"))],
        }
    )


async def _collect_lines(resp):
    """Read a StreamingResponse body into parsed NDJSON dicts."""
    import json as _json

    text = ""
    async for chunk in resp.body_iterator:
        text += chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
    out = []
    for line in text.split("\n"):
        if line.strip():
            out.append(_json.loads(line))
    return out


class AskStreamTest(unittest.TestCase):
    """POST /api/ask streams meta first, then one audio line per sentence."""

    def _stubs(self):
        import os
        import tempfile

        def fake_stt(path):
            return {"text": "can police search me", "seconds": 1.0}

        def fake_brain(question):
            return {
                "language": "en",
                "status": "answered",
                "right": "You have a right.",
                "say_this": "Say it.",
                "do_this": "Do it.",
                "citations": [],
                "spoken": "Stay calm. Ask for a name. Call 112 first!",
            }

        def fake_speak_sentence(text, out_path=None):
            if out_path is None:
                fd, out_path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
            with wave.open(out_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(b"\x00\x00" * 160)
            return out_path

        ids = {"n": 0}

        def fake_store(path):
            ids["n"] += 1
            return f"clip{ids['n']}"

        return fake_stt, fake_brain, fake_speak_sentence, fake_store

    def test_ask_streams_meta_then_audio(self):
        from fastapi.responses import StreamingResponse

        fake_stt, fake_brain, fake_speak, fake_store = self._stubs()
        old = (
            srv.stt_transcribe,
            srv.brain_answer,
            srv.tts.speak_sentence,
            srv._store_clip,
        )
        srv.stt_transcribe = fake_stt
        srv.brain_answer = fake_brain
        srv.tts.speak_sentence = fake_speak
        srv._store_clip = fake_store
        try:
            async def go():
                return await srv.ask(
                    _upload(data=FIXTURE.read_bytes()),
                    _ndjson_request(),
                )
            resp = asyncio.run(go())
        finally:
            (
                srv.stt_transcribe,
                srv.brain_answer,
                srv.tts.speak_sentence,
                srv._store_clip,
            ) = old
        self.assertIsInstance(resp, StreamingResponse)
        self.assertIn("x-ndjson", resp.media_type)
        lines = asyncio.run(_collect_lines(resp))
        self.assertGreaterEqual(len(lines), 2)
        # Meta first: transcript, answer and sentences, like ask_text_stream.
        self.assertEqual(lines[0]["type"], "meta")
        self.assertEqual(lines[0]["transcript"], "can police search me")
        self.assertEqual(
            lines[0]["sentences"],
            ["Stay calm.", "Ask for a name.", "Call 112 first!"],
        )
        self.assertIn("stt", lines[0]["timings"])
        self.assertIn("brain", lines[0]["timings"])
        # Then one audio line per sentence, in order, starting with the
        # first sentence so the browser can play it right away.
        audio = [ln for ln in lines if ln["type"] == "audio"]
        self.assertEqual(len(audio), 3)
        self.assertEqual(
            [ln["sentence"] for ln in audio],
            ["Stay calm.", "Ask for a name.", "Call 112 first!"],
        )
        self.assertEqual([ln["index"] for ln in audio], [0, 1, 2])
        for ln in audio:
            self.assertTrue(ln["audio_url"].startswith("/audio/"))

    def test_ask_format_json_keeps_dict_fallback(self):
        """?format=json answers with the legacy dict, not NDJSON."""

        def fake_run(path):
            import shutil
            import tempfile

            fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            shutil.copy(str(ROOT / "data" / "audio" / "test_en.wav"), tmp_wav)
            return {"transcript": "hello orb",
                    "answer": {"status": "answered"},
                    "audio_out": tmp_wav,
                    "timings": {}}

        old, srv.pipeline_run = srv.pipeline_run, fake_run
        try:
            async def go():
                return await srv.ask(
                    _upload(data=FIXTURE.read_bytes()),
                    _ndjson_request(),
                    format="json",
                )
            data = asyncio.run(go())
        finally:
            srv.pipeline_run = old
        self.assertIsInstance(data, dict)
        self.assertEqual(data["transcript"], "hello orb")
        stored = ROOT / "data" / "audio_out" / (data["audio_url"].split("/")[-1])
        self.assertTrue(stored.is_file())
        stored.unlink()


if __name__ == "__main__":
    unittest.main()
