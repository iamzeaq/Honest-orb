"""Verify data/voice/dataset clips before training.

For each row in ``metadata.csv``::

1. Duration vs text length: flag if under 1.0 s total or shorter than
   ~0.25 s per word, or if mostly silence (energy VAD speech fraction).
2. Transcribe the final clip with ``api/stt.py`` and compute word-overlap
   similarity with its script text; flag if < 0.6.
3. Remove every flagged clip from ``metadata.csv`` (kept in the report).

Rewrites ``metadata.csv`` without flagged rows and writes a Markdown
report (default ``data/voice/verify_report.md``). Prints a summary.

Usage (run from the repo root, inside the project venv)::

    python api/verify_voice.py [--dataset ...] [--report ...]
"""

from __future__ import annotations

import argparse
import csv
import sys
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DEFAULT = REPO_ROOT / "data" / "voice" / "dataset"
REPORT_DEFAULT = REPO_ROOT / "data" / "voice" / "verify_report.md"

MIN_TOTAL_S = 1.0  # any real line reading is longer than this
MIN_S_PER_WORD = 0.25  # expected speaking rate floor
MIN_SPEECH_FRAC = 0.5  # energy-VAD speech fraction floor
MIN_OVERLAP = 0.6  # transcript vs script word-overlap floor


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        pcm = np.frombuffer(
            w.readframes(w.getnframes()), dtype=np.int16
        ).astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            pcm = pcm.reshape(-1, 2).mean(axis=1)
        return pcm, w.getframerate()


def speech_fraction(audio: np.ndarray, sr: int, frame_ms: int = 20) -> float:
    """Fraction of 20 ms frames above an energy threshold."""
    frame = max(1, int(sr * frame_ms / 1000))
    n = max(1, len(audio) // frame)
    rms = np.sqrt((audio[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    if float(rms.max()) <= 1e-6:
        return 0.0
    return float((rms >= max(0.01, float(rms.max()) * 0.05)).mean())


def transcribe_clip(path: Path) -> str:
    try:
        from stt import transcribe as stt_transcribe  # run as api/verify_voice.py
    except ImportError:  # run as a module: python -m api.verify_voice
        from api.stt import transcribe as stt_transcribe
    return str(stt_transcribe(path).get("text", "") or "").strip()


def main(argv: list[str] | None = None) -> int:
    try:
        from voice_prep import normalize_words, word_overlap  # noqa
    except ImportError:
        from api.voice_prep import normalize_words, word_overlap  # noqa

    parser = argparse.ArgumentParser(description="Verify data/voice/dataset clips.")
    parser.add_argument("--dataset", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--report", type=Path, default=REPORT_DEFAULT)
    args = parser.parse_args(argv)

    meta = args.dataset / "metadata.csv"
    try:
        with meta.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="|"))
    except OSError as exc:
        print(f"ERROR: cannot read {meta}: {exc}")
        return 1
    print(f"Checking {len(rows)} clips in {args.dataset} ...", flush=True)

    kept: list[dict] = []
    removed: list[tuple[str, str]] = []  # (file, reason)
    passing_seconds = 0.0
    stt_errors = 0
    for row in rows:
        name, text = row["file"], row["text"]
        reasons: list[str] = []
        try:
            audio, sr = read_wav_mono(args.dataset / name)
        except (OSError, wave.Error, ValueError) as exc:
            removed.append((name, f"unreadable audio ({exc})"))
            print(f"  {name}: REMOVED (unreadable audio)")
            continue
        seconds = len(audio) / float(sr or 1)
        n_words = len(normalize_words(text))
        if seconds < MIN_TOTAL_S:
            reasons.append(f"too short total ({seconds:.2f}s < {MIN_TOTAL_S:.1f}s)")
        if seconds < MIN_S_PER_WORD * n_words:
            reasons.append(
                f"too short for {n_words} words "
                f"({seconds:.2f}s < {MIN_S_PER_WORD * n_words:.2f}s)"
            )
        frac = speech_fraction(audio, sr)
        if frac < MIN_SPEECH_FRAC:
            reasons.append(f"mostly silence (speech {frac:.0%})")
        try:
            transcript = transcribe_clip(args.dataset / name)
            transcribed_ok = True
        except Exception as exc:
            stt_errors += 1
            reasons.append(f"transcribe FAILED ({exc})")
            transcript, transcribed_ok = "", False
        if transcribed_ok:
            score = word_overlap(
                normalize_words(transcript), normalize_words(text)
            )
            if score < MIN_OVERLAP:
                shown = transcript if transcript else "<empty>"
                reasons.append(
                    f"transcript overlap {score:.2f} < {MIN_OVERLAP} ({shown[:80]})"
                )
        if reasons:
            removed.append((name, "; ".join(reasons)))
            print(f"  {name}: REMOVED ({'; '.join(reasons)})")
        else:
            kept.append(row)
            passing_seconds += seconds
            print(f"  {name}: ok ({seconds:.1f}s)")
        if stt_errors > max(5, len(rows) // 10):
            print("ERROR: too many transcribe failures, not touching metadata.csv")
            return 1

    with meta.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "text"], delimiter="|")
        writer.writeheader()
        writer.writerows(kept)

    with args.report.open("w", encoding="utf-8") as f:
        f.write("# Voice verification report\n\n")
        f.write(f"Clips checked: {len(rows)}\n\n")
        f.write(f"Clips passed: {len(kept)}\n\n")
        f.write("## Removed clips\n\n")
        if removed:
            for name, reason in removed:
                f.write(f"- `{name}`: {reason}\n")
        else:
            f.write("None — every clip passed.\n")
        f.write(f"\nTotal minutes of passing audio: {passing_seconds / 60:.1f}\n")

    print(f"\nChecked: {len(rows)}, passed: {len(kept)}, "
          f"removed: {len(removed)}")
    print(f"Total minutes of passing audio: {passing_seconds / 60:.1f}")
    print(f"metadata.csv rewritten ({len(kept)} rows); report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
