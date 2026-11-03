"""Regression tests for live audio delivery (Modal: no baked audio dir).

The live app returned the written answer but no audio: the Modal image
only bakes in ``data/sections.json``, so ``data/audio_out/`` does not
exist in the container, and the ``lifespan`` that would create it was
never registered on the FastAPI app. ``_store_clip`` then failed with
ENOENT and every answer came back as a 502 with no audio at all.

These tests reproduce the Modal condition (an ``AUDIO_OUT_DIR`` that
does not exist yet) and require both the fix and the container-proof
delivery: answers must carry the audio inline as base64 data URIs, so
playback never depends on which container serves ``/audio/``.

Uses only the standard library (``python -m unittest
api.test_audio_inline``). The brain and TTS are stubbed at the server
boundary, so these run offline and fast.
"""

import base64
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api.server as srv  # noqa: E402

FAKE_SPOKEN = "Stay calm. Ask for a name. Call 112 first!"


def _write_tiny_wav(path):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 160)
    return path


def _fake_brain(question):
    return {
        "language": "en",
        "status": "answered",
        "right": "You have a right.",
        "say_this": "Say it.",
        "do_this": "Do it.",
        "citations": [],
        "spoken": FAKE_SPOKEN,
    }


def _fake_speak_sentences(text, out_dir=None, on_sentence=None):
    from api.tts import split_sentences

    paths = []
    for i, _sent in enumerate(split_sentences(text)):
        fd, p = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        _write_tiny_wav(p)
        paths.append(p)
        if on_sentence is not None:
            on_sentence(i, p)
    return paths


def _fake_speak_sentence(text, out_path=None):
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    return _write_tiny_wav(out_path)


def _decode_data_uri(uri):
    prefix = "data:audio/wav;base64,"
    assert uri.startswith(prefix), uri[:60]
    return base64.b64decode(uri[len(prefix):])


class AudioInlineTest(unittest.TestCase):
    def _missing_dir(self):
        missing = Path(tempfile.mkdtemp(prefix="orb_no_audio_")) / "audio_out"
        assert not missing.exists()
        return missing

    def _enter_missing_dir(self, missing):
        old = srv.AUDIO_OUT_DIR
        srv.AUDIO_OUT_DIR = missing
        return old

    def test_ask_text_works_when_audio_dir_missing(self):
        """Modal has no baked data/audio_out: ask_text must still speak."""
        from fastapi import HTTPException

        missing = self._missing_dir()
        old_dir, old_brain, old_speak = (
            srv.AUDIO_OUT_DIR, srv.brain_answer, srv.tts.speak_sentences,
        )
        srv.AUDIO_OUT_DIR, srv.brain_answer = missing, _fake_brain
        srv.tts.speak_sentences = _fake_speak_sentences
        try:
            try:
                data = srv.ask_text(srv.AskTextBody(text="Can police search me?"))
            except HTTPException as exc:
                self.fail(f"ask_text raised {exc.status_code}: {exc.detail}")
            # The whole-answer clip and the per-sentence queue are stored.
            for url in [data["audio_url"]] + data["audio_urls"]:
                self.assertTrue(url.startswith("/audio/"))
                stored = missing / url.split("/")[-1]
                self.assertTrue(stored.is_file(), stored)
                # ... and served back through the /audio/ endpoint.
                resp = srv.get_audio(url.split("/")[-1].removesuffix(".wav"))
                self.assertTrue(Path(resp.path).is_file())
        finally:
            srv.AUDIO_OUT_DIR = old_dir
            srv.brain_answer = old_brain
            srv.tts.speak_sentences = old_speak
        try:
            # The same audio also rides inline, so playback never depends
            # on which container serves /audio/.
            full = _decode_data_uri(data["audio_b64"])
            self.assertTrue(full.startswith(b"RIFF"))
            self.assertEqual(len(data["audio_b64s"]), len(data["audio_urls"]))
            for uri in data["audio_b64s"]:
                self.assertTrue(_decode_data_uri(uri).startswith(b"RIFF"))
        finally:
            import shutil

            shutil.rmtree(missing, ignore_errors=True)

    def test_ask_text_stream_lines_carry_inline_audio(self):
        """Each stream audio line must carry playable inline bytes."""
        import asyncio
        import json as _json

        missing = self._missing_dir()
        old = (
            srv.AUDIO_OUT_DIR,
            srv.brain_answer,
            srv.tts.speak_sentence,
        )
        srv.AUDIO_OUT_DIR = missing
        srv.brain_answer = _fake_brain
        srv.tts.speak_sentence = _fake_speak_sentence
        try:
            resp = srv.ask_text_stream(
                srv.AskTextBody(text="Can police search me?"))

            async def _collect():
                text = ""
                async for chunk in resp.body_iterator:
                    text += chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                return text

            text = asyncio.run(_collect())
            lines = [_json.loads(ln) for ln in text.split("\n") if ln.strip()]
        finally:
            (
                srv.AUDIO_OUT_DIR,
                srv.brain_answer,
                srv.tts.speak_sentence,
            ) = old
        try:
            audio = [ln for ln in lines if ln["type"] == "audio"]
            self.assertEqual(len(audio), 3)
            for ln in audio:
                self.assertTrue(ln["audio_url"].startswith("/audio/"))
                raw = _decode_data_uri(ln["audio_b64"])
                self.assertTrue(raw.startswith(b"RIFF"))
        finally:
            import shutil

            shutil.rmtree(missing, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
