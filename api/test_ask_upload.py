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


if __name__ == "__main__":
    unittest.main()
