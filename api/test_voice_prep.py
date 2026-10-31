"""Focused tests for the monotonic script alignment in api/voice_prep.py.

Run from the repo root inside the venv::

    python -m unittest api.test_voice_prep -v

Only pure functions are exercised here (no audio files, no network).
"""

import unittest

from api.voice_prep import align_pieces, normalize_words, word_overlap


def script(*texts):
    return [(i + 1, t) for i, t in enumerate(texts)]


class TestNormalize(unittest.TestCase):
    def test_punctuation_and_case_ignored(self):
        self.assertEqual(
            normalize_words("Oga, Officer! no FIT search?"),
            ["oga", "officer", "no", "fit", "search"],
        )


class TestWordOverlap(unittest.TestCase):
    def test_identical_is_one(self):
        words = normalize_words("bail na free dem no fit bill you")
        self.assertEqual(word_overlap(words, words), 1.0)

    def test_empty_is_zero(self):
        self.assertEqual(word_overlap([], ["bail"]), 0.0)
        self.assertEqual(word_overlap(["bail"], []), 0.0)

    def test_partial(self):
        full = normalize_words("bail na free dem no fit bill you")
        # Short fragment: below threshold so it gets merged, not kept alone.
        frag = normalize_words("bail na free")
        self.assertLess(word_overlap(frag, full), 0.6)
        # Near-complete reading: above threshold.
        near = normalize_words("bail na free dem no fit bill")
        self.assertGreaterEqual(word_overlap(near, full), 0.6)
        self.assertLess(word_overlap(near, full), 1.0)


class TestAlignPieces(unittest.TestCase):
    LINES = script(
        "Oga officer no fit search your phone without reason stay calm",
        "Abeg ask for im name and im station before you answer",
        "Police no fit arrest you for another person matter at all",
    )

    def test_in_order_pieces_match_in_order(self):
        pieces = [
            "oga officer no fit search your phone without reason stay calm",
            "abeg ask for im name and im station before you answer",
            "police no fit arrest you for another person matter at all",
        ]
        kept, dropped, pos = align_pieces(pieces, self.LINES, start=0)
        self.assertEqual([line for line, _, _ in kept], [1, 2, 3])
        self.assertEqual(dropped, [])
        self.assertEqual(pos, 3)

    def test_restart_piece_dropped_and_reported(self):
        pieces = [
            "oga officer no fit",  # false start: too little overlap
            "oga officer no fit search your phone without reason stay calm",
        ]
        kept, dropped, _ = align_pieces(pieces, self.LINES, start=0)
        self.assertEqual([line for line, _, _ in kept], [1])
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0][0], 0)  # piece index

    def test_midline_pause_merged_greedily(self):
        pieces = [
            "oga officer no fit search your",
            "phone without reason stay calm",
            "abeg ask for im name and im station before you answer",
        ]
        kept, dropped, _ = align_pieces(pieces, self.LINES, start=0)
        self.assertEqual([line for line, _, _ in kept], [1, 2])
        self.assertEqual([span for _, span, _ in kept][0], (0, 2))
        self.assertEqual(dropped, [])

    def test_monotonic_never_goes_backwards(self):
        pieces = [
            "police no fit arrest you for another person matter at all",
            "abeg ask for im name and im station before you answer",
        ]
        kept, _, pos = align_pieces(pieces, self.LINES, start=0)
        # First piece jumps to line 3; second piece is behind the pointer.
        self.assertEqual([line for line, _, _ in kept], [3])
        self.assertEqual(pos, 3)

    def test_duplicate_line_keeps_better_score(self):
        pieces = [
            "abeg ask for im name and im station",  # partial but >= 0.6
            "abeg ask for im name and im station before you answer",  # full
        ]
        kept, _, _ = align_pieces(pieces, self.LINES, start=1)
        line2 = [k for k in kept if k[0] == 2]
        self.assertEqual(len(line2), 1)
        self.assertEqual(line2[0][1], (1, 2))  # span of the better piece
        self.assertGreater(line2[0][2], 0.9)


class TestTrimAndPad(unittest.TestCase):
    def test_noise_burst_plus_gap_keeps_full_reading(self):
        import numpy as np

        from api.voice_prep import SAMPLE_RATE, trim_and_pad

        sr = SAMPLE_RATE
        rng = np.random.default_rng(0)
        burst = (rng.standard_normal(int(0.08 * sr)) * 0.5).astype(np.float32)
        gap = np.zeros(int(0.4 * sr), dtype=np.float32)
        t = np.arange(int(2.5 * sr)) / sr
        speech = (
            0.4 * np.sin(2 * np.pi * 150 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
        ).astype(np.float32)
        out = trim_and_pad(np.concatenate([burst, gap, speech]))
        # Full ~3 s span plus 2x0.15 s padding, not just the 80 ms burst.
        self.assertGreater(len(out) / sr, 2.5)

    def test_outage_aborts_but_occasional_failure_passes(self):
        from api.voice_prep import TranscribeOutage, check_transcribe_health

        check_transcribe_health(20, 3, "batch_01.m4a")  # occasional: fine
        check_transcribe_health(0, 0, "batch_01.m4a")  # empty: fine
        with self.assertRaises(TranscribeOutage):
            check_transcribe_health(20, 19, "batch_01.m4a")  # outage: abort

    def test_pure_silence_returns_padded_input(self):
        import numpy as np

        from api.voice_prep import SAMPLE_RATE, trim_and_pad

        out = trim_and_pad(np.zeros(SAMPLE_RATE, dtype=np.float32))
        self.assertGreater(len(out), SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
