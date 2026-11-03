"""Focused tests for TTS clarity: normalize_for_speech + sentence split.

Run from the repo root inside the venv::

    python -m unittest api.test_tts_clarity -v

Only pure functions are exercised here (no model load, no network).
"""

import unittest

from api.tts import normalize_for_speech, split_sentences


class TestNormalizeForSpeech(unittest.TestCase):
    def test_plain_numbers_become_words(self):
        out = normalize_for_speech("See section 35.")
        self.assertIn("thirty-five", out)
        self.assertNotRegex(out, r"\d")

    def test_emergency_number_is_digit_by_digit(self):
        self.assertEqual(
            normalize_for_speech("Call 112 first."),
            "call one one two first",
        )

    def test_section_abbreviations_expanded(self):
        self.assertEqual(
            normalize_for_speech("Police Act 2020, s.35"),
            "police act two thousand twenty section thirty-five",
        )
        self.assertEqual(
            normalize_for_speech("ss.49-51"),
            "sections forty-nine to fifty-one",
        )

    def test_brackets_and_symbols_stripped(self):
        out = normalize_for_speech("[Police Act] #Rights! (calm) 50%")
        self.assertNotRegex(out, r"[\[\]#()*%!]")
        self.assertIn("fifty percent", out)

    def test_abbreviations_expanded(self):
        out = normalize_for_speech("Mr Smith & Dr Jones")
        self.assertIn("mister", out)
        self.assertIn("and", out)
        self.assertIn("doctor", out)

    def test_missing_vocab_letters_mapped(self):
        # The mms-tts-pcm vocab has no 'q': map to the closest sound 'k'.
        out = normalize_for_speech("Be quick and quiet.")
        self.assertNotIn("q", out)
        self.assertIn("kuick", out)
        self.assertIn("kuiet", out)

    def test_missing_digits_never_reach_tokenizer(self):
        # Vocab has no 6/8/9: every digit must become a word.
        out = normalize_for_speech("Quiz s.69 (8) 100% §9")
        self.assertNotRegex(out, r"[0-9q]")
        self.assertRegex(out, r"^[a-z' \-]*$")
        for word in ("sixty-nine", "eight", "nine", "one hundred percent"):
            self.assertIn(word, out)


class TestSplitSentences(unittest.TestCase):
    def test_three_sentences(self):
        text = "Stay calm. Ask for their name. Call 112 first!"
        self.assertEqual(len(split_sentences(text)), 3)

    def test_empty_is_empty(self):
        self.assertEqual(split_sentences("   "), [])


class TestLoudnessNormalization(unittest.TestCase):
    """Every synthesized sentence must be phone-speaker loud.

    Target is about -16 LUFS integrated loudness with true peak at or
    below -1 dBFS. Pure synthesis-free checks on synthetic signals.
    """

    def _speech_like(self, seconds=2.0, sr=16000, level=0.2):
        import numpy as np

        t = np.arange(int(sr * seconds)) / sr
        # Voiced-ish wobble: a low tone with slow amplitude modulation so
        # gating sees real speech-like energy, not digital silence.
        return (
            level * np.sin(2 * np.pi * 220 * t)
            * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
        ).astype(np.float64)

    def test_sentence_lands_near_minus_16_lufs(self):
        from api.tts import measure_lufs, normalize_loudness

        out = normalize_loudness(self._speech_like(), 16000)
        measured = measure_lufs(out, 16000)
        self.assertIsNotNone(measured)
        self.assertAlmostEqual(measured, -16.0, delta=1.0)

    def test_quiet_sentence_is_lifted_loud_one_is_kept(self):
        from api.tts import measure_lufs, normalize_loudness

        for level in (0.02, 0.5):
            out = normalize_loudness(self._speech_like(level=level), 16000)
            measured = measure_lufs(out, 16000)
            self.assertIsNotNone(measured)
            self.assertAlmostEqual(measured, -16.0, delta=1.5)

    def test_peak_never_exceeds_minus_1_dbfs(self):
        import numpy as np

        from api.tts import normalize_loudness

        hot = self._speech_like(level=0.99)
        out = normalize_loudness(hot, 16000)
        self.assertTrue(bool(np.all(np.isfinite(out))))
        self.assertLessEqual(float(np.max(np.abs(out))), 10 ** (-1 / 20) + 1e-6)
        # A full-scale transient too short to measure must still be limited.
        click = np.zeros(16000, dtype=np.float64)
        click[8000] = 1.0
        out = normalize_loudness(click, 16000)
        self.assertTrue(bool(np.all(np.isfinite(out))))
        self.assertLessEqual(float(np.max(np.abs(out))), 10 ** (-1 / 20) + 1e-6)

    def test_silence_stays_silent(self):
        import numpy as np

        from api.tts import measure_lufs, normalize_loudness

        silence = np.zeros(16000, dtype=np.float64)
        out = normalize_loudness(silence, 16000)
        self.assertTrue(bool(np.all(np.isfinite(out))))
        self.assertTrue(bool((out == 0).all()))
        self.assertIsNone(measure_lufs(silence, 16000))


class TestAskTextSentenceQueue(unittest.TestCase):
    def test_ask_text_returns_audio_url_queue(self):
        import tempfile
        import wave
        from pathlib import Path

        import api.server as srv

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

        def fake_speak_sentences(text, out_dir=None, on_sentence=None):
            import os

            from api.tts import split_sentences

            paths = []
            for i, sent in enumerate(split_sentences(text)):
                fd, p = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                with wave.open(p, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    w.writeframes(b"\x00\x00" * 160)
                paths.append(p)
                if on_sentence is not None:
                    on_sentence(i, p)
            return paths

        old_brain, old_speak = srv.brain_answer, srv.tts.speak_sentences
        srv.brain_answer, srv.tts.speak_sentences = fake_brain, fake_speak_sentences
        try:
            data = srv.ask_text(srv.AskTextBody(text="Can police search me?"))
        finally:
            srv.brain_answer, srv.tts.speak_sentences = old_brain, old_speak
        try:
            self.assertEqual(len(data["audio_urls"]), 3)
            # audio_url is the whole answer as one clip; audio_urls is the
            # per-sentence queue starting with sentence 1.
            self.assertTrue(data["audio_url"].startswith("/audio/"))
            self.assertNotEqual(data["audio_url"], data["audio_urls"][0])
            for url in [data["audio_url"]] + data["audio_urls"]:
                self.assertTrue(url.startswith("/audio/"))
                self.assertTrue(
                    (Path(srv.AUDIO_OUT_DIR) / (url.split("/")[-1])).is_file()
                )
            self.assertEqual(
                data["sentences"],
                ["Stay calm.", "Ask for a name.", "Call 112 first!"],
            )
            self.assertIn("tts_first", data["timings"])
            self.assertLessEqual(
                data["timings"]["tts_first"], data["timings"]["tts"]
            )
        finally:
            for url in [data.get("audio_url")] + data.get("audio_urls", []):
                if not url:
                    continue
                p = Path(srv.AUDIO_OUT_DIR) / (url.split("/")[-1])
                if p.is_file():
                    p.unlink()


if __name__ == "__main__":
    unittest.main()
