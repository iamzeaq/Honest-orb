"""Clear-voice variant bake-off: synthesize, transcribe, score.

Generates the same 3 sentences for each variant into
``data/audio_out/clear_<variant>_<n>.wav`` (n = 1..3):

- A: facebook/mms-tts-eng, VITS defaults + loudness only
- B: mms-tts-eng, noise_scale 0.4, noise_scale_duration 0.5,
  speaking_rate 0.9 + loudness + EQ
- C: same as B + subtle orb reverb
- D: facebook/mms-tts-pcm with B's settings + loudness + EQ

Then transcribes every file with ``api/stt.py`` and prints the word
accuracy vs the input sentence (clarity score) plus a per-variant
average table. All voice knobs go through the environment
(TTS_MODEL_ID, TTS_NOISE_SCALE, TTS_NOISE_SCALE_DURATION,
TTS_SPEAKING_RATE, VOICE_STYLE), so variants switch without code
changes.

Run from the repo root inside the venv::

    python -m api.clear_voice_eval
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "audio_out"

SENTENCES = [
    "Abeg officer, show me the warrant before you search my car.",
    "No wahala, Honest Orb go guide you step by step today.",
    "You have the right to remain silent until your lawyer arrives.",
]

VARIANTS = {
    # variant: (model, noise_scale, noise_scale_duration, speaking_rate, style)
    "A": ("facebook/mms-tts-eng", 0.667, 0.8, 1.0, "loud"),
    "B": ("facebook/mms-tts-eng", 0.4, 0.5, 0.9, "clear"),
    "C": ("facebook/mms-tts-eng", 0.4, 0.5, 0.9, "orb"),
    "D": ("facebook/mms-tts-pcm", 0.4, 0.5, 0.9, "clear"),
}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def word_accuracy(reference: str, hypothesis: str) -> float:
    """Word accuracy in [0, 1]: 1 - WER, floored at 0."""
    ref = _words(reference)
    if not ref:
        return 1.0 if not _words(hypothesis) else 0.0
    return max(0.0, (len(ref) - _edit_distance(ref, _words(hypothesis))) / len(ref))


def main() -> int:
    from transformers import set_seed

    from api.stt import transcribe
    from api.tts import speak_sentence

    results: dict[str, list[tuple[str, str, float | None]]] = {}
    for variant, (model, ns, nsd, rate, style) in VARIANTS.items():
        os.environ["TTS_MODEL_ID"] = model
        os.environ["TTS_NOISE_SCALE"] = str(ns)
        os.environ["TTS_NOISE_SCALE_DURATION"] = str(nsd)
        os.environ["TTS_SPEAKING_RATE"] = str(rate)
        os.environ["VOICE_STYLE"] = style
        print(f"--- variant {variant}: {model} ns={ns} nsd={nsd} "
              f"rate={rate} style={style}", flush=True)
        rows = []
        for n, sentence in enumerate(SENTENCES, 1):
            out_path = OUT_DIR / f"clear_{variant}_{n}.wav"
            set_seed(1234)  # same draw for every variant: fair comparison
            path = speak_sentence(sentence, str(out_path))
            transcript = ""
            score: float | None = None
            try:
                transcript = transcribe(path)["text"]
                score = word_accuracy(sentence, transcript)
            except Exception as exc:  # STT needs MODEL_API_KEY + network
                transcript = f"<transcribe failed: {exc}>"
            print(f"  [{variant}{n}] {out_path.name}")
            print(f"    ref : {sentence}")
            print(f"    heard: {transcript}")
            print(f"    clarity: {'n/a' if score is None else f'{100 * score:.1f}%'}",
                  flush=True)
            rows.append((out_path.name, transcript, score))
        results[variant] = rows

    print("\nvariant | avg clarity score")
    print("--------|------------------")
    for variant in VARIANTS:
        scores = [s for _, _, s in results[variant] if s is not None]
        avg = "n/a (transcribe unavailable)" if not scores else f"{100 * sum(scores) / len(scores):.1f}%"
        print(f"   {variant}    | {avg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
