"""Tests for VITS env settings + VOICE_STYLE post-processing chain.

Run from the repo root inside the venv::

    python -m unittest api.test_voice_style -v

Only pure functions are exercised here (no model load, no network).
"""

import os
import unittest
from contextlib import contextmanager


@contextmanager
def _env(**values):
    """Temporarily set/unset env vars (None unsets)."""
    saved = {key: os.environ.get(key) for key in values}
    try:
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _tone(freq_hz, seconds=2.0, sr=16000, level=0.5):
    import numpy as np

    t = np.arange(int(sr * seconds)) / sr
    return (level * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float64)


def _speech_like(seconds=2.0, sr=16000, level=0.2):
    import numpy as np

    t = np.arange(int(sr * seconds)) / sr
    return (
        level * np.sin(2.0 * np.pi * 220 * t)
        * (0.6 + 0.4 * np.sin(2.0 * np.pi * 3 * t))
    ).astype(np.float64)


class TestVitsEnvSettings(unittest.TestCase):
    def test_defaults_match_mms_builtin(self):
        from api import tts

        with _env(TTS_NOISE_SCALE=None, NOISE_SCALE=None,
                  TTS_NOISE_SCALE_DURATION=None, NOISE_SCALE_DURATION=None,
                  TTS_SPEAKING_RATE=None, SPEAKING_RATE=None):
            self.assertAlmostEqual(tts.noise_scale(), 0.667)
            self.assertAlmostEqual(tts.noise_scale_duration(), 0.8)
            self.assertAlmostEqual(tts.speaking_rate(), 1.0)

    def test_env_values_are_used(self):
        from api import tts

        with _env(TTS_NOISE_SCALE="0.4", TTS_NOISE_SCALE_DURATION="0.5",
                  TTS_SPEAKING_RATE="0.9"):
            self.assertAlmostEqual(tts.noise_scale(), 0.4)
            self.assertAlmostEqual(tts.noise_scale_duration(), 0.5)
            self.assertAlmostEqual(tts.speaking_rate(), 0.9)

    def test_bad_values_fall_back_to_defaults(self):
        from api import tts

        with _env(TTS_NOISE_SCALE="loud", TTS_SPEAKING_RATE="-2"):
            self.assertAlmostEqual(tts.noise_scale(), 0.667)
            self.assertAlmostEqual(tts.speaking_rate(), 1.0)


class TestStyleFlags(unittest.TestCase):
    def test_loud_is_loudness_only(self):
        from api.tts import _style_flags

        self.assertEqual(_style_flags("loud"), (True, False, False))
        self.assertEqual(_style_flags("loudness"), (True, False, False))

    def test_clear_adds_eq(self):
        from api.tts import _style_flags

        self.assertEqual(_style_flags("clear"), (True, True, False))

    def test_orb_adds_reverb_on_top_of_eq(self):
        from api.tts import _style_flags

        self.assertEqual(_style_flags("orb"), (True, True, True))
        self.assertEqual(_style_flags("clear+reverb"), (True, True, True))

    def test_off_disables_everything(self):
        from api.tts import _style_flags

        self.assertEqual(_style_flags("off"), (False, False, False))

    def test_default_style_is_clear(self):
        from api import tts

        with _env(VOICE_STYLE=None):
            self.assertEqual(tts.voice_style(), "clear")
            self.assertEqual(tts._style_flags(), (True, True, False))


class TestEqualizer(unittest.TestCase):
    def test_highpass_removes_rumble_keeps_voice(self):
        import numpy as np

        from api.tts import highpass_80hz

        sr = 16000
        rumble = _tone(40, sr=sr)
        voice = _tone(500, sr=sr)
        rumble_ratio = float(np.sum(highpass_80hz(rumble, sr) ** 2) / np.sum(rumble**2))
        voice_ratio = float(np.sum(highpass_80hz(voice, sr) ** 2) / np.sum(voice**2))
        self.assertLess(rumble_ratio, 0.25)
        self.assertGreater(voice_ratio, 0.95)

    def test_presence_boosts_3khz_not_lows(self):
        import numpy as np

        from api.tts import presence_boost

        sr = 16000
        high = _tone(3000, sr=sr)
        low = _tone(300, sr=sr)
        high_ratio = float(np.sum(presence_boost(high, sr) ** 2) / np.sum(high**2))
        low_ratio = float(np.sum(presence_boost(low, sr) ** 2) / np.sum(low**2))
        # +3 dB doubles energy; allow tolerance for spectral leakage.
        self.assertGreater(high_ratio, 1.7)
        self.assertLess(high_ratio, 2.3)
        self.assertAlmostEqual(low_ratio, 1.0, delta=0.05)


class TestOrbReverb(unittest.TestCase):
    def test_same_length_finite_and_subtle(self):
        import numpy as np

        from api.tts import orb_reverb

        sr = 16000
        x = _speech_like(sr=sr)
        y = orb_reverb(x, sr)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(bool(np.all(np.isfinite(y))))
        change = float(np.linalg.norm(y - x) / np.linalg.norm(x))
        self.assertGreater(change, 0.0)
        # About -20 dB wet: clearly audible as space, never masking speech.
        self.assertLess(change, 0.35)

    def test_deterministic(self):
        import numpy as np

        from api.tts import orb_reverb

        x = _speech_like()
        self.assertTrue(bool((orb_reverb(x, 16000) == orb_reverb(x, 16000)).all()))


class TestVoiceStyleChain(unittest.TestCase):
    def test_every_style_lands_near_minus_16_lufs(self):
        from api.tts import apply_voice_style, measure_lufs

        for style in ("loud", "clear", "orb"):
            out = apply_voice_style(_speech_like(), 16000, style=style)
            measured = measure_lufs(out, 16000)
            self.assertIsNotNone(measured, style)
            self.assertAlmostEqual(measured, -16.0, delta=1.5, msg=style)

    def test_every_style_peak_limited(self):
        import numpy as np

        from api.tts import apply_voice_style

        for style in ("loud", "clear", "orb"):
            out = apply_voice_style(_speech_like(level=0.9), 16000, style=style)
            self.assertLessEqual(
                float(np.max(np.abs(out))), 10 ** (-1 / 20) + 1e-6, msg=style
            )

    def test_off_passes_through(self):
        import numpy as np

        from api.tts import apply_voice_style

        x = _speech_like()
        out = apply_voice_style(x, 16000, style="off")
        self.assertTrue(bool(np.allclose(out, x, rtol=1e-6, atol=1e-7)))

    def test_style_comes_from_env(self):
        from api.tts import apply_voice_style, measure_lufs

        with _env(VOICE_STYLE="orb"):
            out = apply_voice_style(_speech_like(), 16000)
            self.assertAlmostEqual(measure_lufs(out, 16000), -16.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
