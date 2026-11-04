"""Tests for the TTS never-silent fallbacks (api/tts.py).

Run from the repo root inside the venv::

    python -m unittest api.test_tts_fallback -v

Only pure helpers are exercised, plus speak_sentence with a stubbed
synthesizer (no model load, no network): a sentence that fails twice is
retried once in a shorter form before it is allowed to fail.
"""

import os
import tempfile
import unittest
import wave

from api import tts


def _fake_waves(text):
    import numpy as np

    sr = 16000
    n = max(sr // 4, len(text) * 80)
    return [0.3 * np.sin(2.0 * np.pi * 220.0 *
                         __import__("numpy").arange(n) / sr)], sr


class ShortenTest(unittest.TestCase):
    def test_long_sentence_cuts_at_word_boundary(self):
        text = ("The Constitution guarantees your privacy including "
                "telephone conversations and your home and property "
                "against unlawful search and seizure by anyone at all.")
        short = tts.shorten_for_retry(text, limit=80)
        self.assertTrue(short)
        self.assertLessEqual(len(short), 80)
        self.assertTrue(text.startswith(short))
        self.assertNotIn(" ", short[-1:])

    def test_short_sentence_falls_back_to_first_clause(self):
        short = tts.shorten_for_retry("Stay calm, ask for the name.", limit=140)
        self.assertEqual(short, "Stay calm")

    def test_no_shorter_version_returns_empty(self):
        self.assertEqual(tts.shorten_for_retry("Stay calm.", limit=140), "")
        self.assertEqual(tts.shorten_for_retry("", limit=140), "")


class SpeakSentenceFallbackTest(unittest.TestCase):
    def setUp(self):
        self.real = tts._synth_normalized
        self.calls = []

    def tearDown(self):
        tts._synth_normalized = self.real

    def _wav_seconds(self, path):
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())

    def test_retry_then_shorter_version_succeeds(self):
        text = ("Stay calm and ask for the officer's name and station "
                "and reason, and write everything down so that your lawyer "
                "can challenge any unlawful search or seizure later in court.")
        full_norm = tts.normalize_for_speech(text)
        short_norm = tts.normalize_for_speech(tts.shorten_for_retry(text))
        self.assertTrue(short_norm and short_norm != full_norm)

        def stub(normalized):
            self.calls.append(normalized)
            if normalized == short_norm:
                return _fake_waves(normalized)
            raise RuntimeError("boom")

        tts._synth_normalized = stub
        fd, out = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        got = tts.speak_sentence(text, out)
        self.assertEqual(got, out)
        # Two full attempts, then the shortened attempt.
        self.assertEqual(self.calls,
                         [full_norm, full_norm, short_norm])
        self.assertGreater(self._wav_seconds(out), 0.1)
        os.remove(out)

    def test_total_failure_raises_original_error(self):
        def stub(normalized):
            raise RuntimeError("engine down")

        tts._synth_normalized = stub
        with self.assertRaisesRegex(RuntimeError, "engine down"):
            tts.speak_sentence("Stay calm and ask for the name.", None)


if __name__ == "__main__":
    unittest.main()
