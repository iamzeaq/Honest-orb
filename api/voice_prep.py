"""Prepare the Honest Orb voice dataset from recorded batches.

Reads ``data/voice/script.txt`` (180 numbered lines) and the recordings in
``data/voice/raw/`` (``batch_01.m4a`` ... ``batch_09.m4a``, 20 lines each),
converts each batch to 16 kHz mono WAV, splits it on silence into clips,
pairs clip *i* of a batch with script line *i* of that batch in order, and
writes ``data/voice/dataset/`` (``wavs/line_NNN.wav`` + ``metadata.csv``
with ``file|text`` rows). Batches whose clip count does not match the line
count are flagged in the printed report and only the overlapping pairs are
kept, so a bad batch can never silently misalign the dataset.

Usage (run from the repo root, inside the project venv)::

    python api/voice_prep.py [--raw-dir ...] [--out-dir ...] [--script ...]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DEFAULT = REPO_ROOT / "data" / "voice" / "script.txt"
RAW_DEFAULT = REPO_ROOT / "data" / "voice" / "raw"
OUT_DEFAULT = REPO_ROOT / "data" / "voice" / "dataset"
SAMPLE_RATE = 16000
AUDIO_EXTS = (".m4a", ".wav", ".mp3", ".ogg", ".flac", ".aac")


def parse_script(script_path: Path) -> list[tuple[int, str]]:
    """Parse ``NNN text`` lines into ``[(line_no, text), ...]``."""
    entries: list[tuple[int, str]] = []
    for raw in script_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        match = re.match(r"^(\d+)\s+(.*\S)\s*$", raw)
        if not match:
            raise ValueError(f"Bad script line (want 'NNN text'): {raw!r}")
        entries.append((int(match.group(1)), match.group(2).strip()))
    return entries


def to_mono_16k(src: Path, dst: Path) -> None:
    """Convert any audio file to 16 kHz mono WAV via the bundled ffmpeg."""
    from imageio_ffmpeg import get_ffmpeg_exe

    cmd = [
        get_ffmpeg_exe(), "-y", "-v", "error",
        "-i", str(src),
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-sample_fmt", "s16",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.exists():
        raise RuntimeError(f"ffmpeg failed on {src.name}: {proc.stderr.strip()}")


def read_wav_mono(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() not in (1, 2) or w.getsampwidth() != 2:
            raise ValueError(f"Unsupported WAV format: {path}")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            pcm = pcm.reshape(-1, 2).mean(axis=1)
        return pcm


def split_on_silence(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: int = 20,
    threshold_ratio: float = 0.05,
    floor: float = 0.01,
    min_silence_s: float = 0.4,
    min_clip_s: float = 0.3,
    pad_s: float = 0.15,
) -> list[np.ndarray]:
    """Split audio into speech clips on silent gaps (energy VAD)."""
    frame = max(1, int(sr * frame_ms / 1000))
    n_frames = max(1, len(audio) // frame)
    trimmed = audio[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt((trimmed ** 2).mean(axis=1) + 1e-12)
    threshold = max(floor, float(rms.max()) * threshold_ratio)
    speech = rms >= threshold

    min_sil_frames = max(1, int(min_silence_s * 1000 / frame_ms))
    clips: list[tuple[int, int]] = []
    start: int | None = None
    silence_run = 0
    for i, is_speech in enumerate(speech):
        if is_speech:
            if start is None:
                start = i
            silence_run = 0
        elif start is not None:
            silence_run += 1
            if silence_run >= min_sil_frames:
                clips.append((start, i - silence_run + 1))
                start = None
                silence_run = 0
    if start is not None:
        clips.append((start, n_frames))

    pad = int(pad_s * sr)
    out: list[np.ndarray] = []
    for s, e in clips:
        s0 = max(0, s * frame - pad)
        e0 = min(len(audio), e * frame + pad)
        if (e0 - s0) / sr < min_clip_s:
            continue
        chunk = audio[s0:e0].astype(np.float64)
        chunk -= chunk.mean()  # drop DC offset from the phone mic
        peak = float(np.abs(chunk).max())
        if peak > 1e-6:
            chunk = chunk / peak * 0.9  # consistent loudness per clip
        out.append(chunk.astype(np.float32))
    return out


def write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())


def process_batch(
    batch_file: Path, lines: list[tuple[int, str]], wavs_dir: Path, tmp_dir: Path
) -> tuple[list[tuple[str, str]], int, float]:
    """Convert, split, and pair one batch. Returns (pairs, clip_count, minutes)."""
    conv = tmp_dir / f"{batch_file.stem}_16k.wav"
    to_mono_16k(batch_file, conv)
    audio = read_wav_mono(conv)
    clips = split_on_silence(audio)
    pairs = []
    for clip, (line_no, text) in zip(clips, lines):
        name = f"wavs/line_{line_no:03d}.wav"
        write_wav(wavs_dir / f"line_{line_no:03d}.wav", clip)
        pairs.append((name, text))
    minutes = sum(len(c) for c in clips) / SAMPLE_RATE / 60.0
    return pairs, len(clips), minutes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build data/voice/dataset from raw batches.")
    parser.add_argument("--script", type=Path, default=SCRIPT_DEFAULT)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--lines-per-batch", type=int, default=20)
    args = parser.parse_args(argv)

    try:
        script = parse_script(args.script)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Script: {len(script)} lines from {args.script}")

    batch_files = sorted(
        p for p in args.raw_dir.glob("batch_*") if p.suffix.lower() in AUDIO_EXTS
    ) if args.raw_dir.is_dir() else []
    if not batch_files:
        print(f"No batch files (batch_*.m4a, 20 lines each) found in {args.raw_dir}.")
        print("Record them per data/voice/README.md, then re-run.")
        return 1

    wavs_dir = args.out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    all_pairs: list[tuple[str, str]] = []
    mismatches: list[str] = []
    total_minutes = 0.0
    print(f"\n{'batch':<14}{'expected':>9}{'clips':>7}  status")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for idx, batch_file in enumerate(batch_files):
            expected = script[idx * args.lines_per_batch:(idx + 1) * args.lines_per_batch]
            if not expected:
                print(f"{batch_file.name:<14}  (no script lines left for this batch)  SKIPPED")
                mismatches.append(batch_file.name)
                continue
            try:
                pairs, n_clips, minutes = process_batch(batch_file, expected, wavs_dir, tmp_dir)
            except (RuntimeError, ValueError, OSError) as exc:
                print(f"{batch_file.name:<14}{len(expected):>9}{'--':>7}  ERROR: {exc}")
                mismatches.append(batch_file.name)
                continue
            total_minutes += minutes
            status = "OK" if n_clips == len(expected) else (
                f"MISMATCH (kept first {len(pairs)})"
            )
            if n_clips != len(expected):
                mismatches.append(f"{batch_file.name} (expected {len(expected)}, got {n_clips})")
            all_pairs.extend(pairs)
            print(f"{batch_file.name:<14}{len(expected):>9}{n_clips:>7}  {status}")

    meta = args.out_dir / "metadata.csv"
    with meta.open("w", encoding="utf-8") as f:
        f.write("file|text\n")
        for name, text in all_pairs:
            f.write(f"{name}|{text}\n")

    print(f"\nWrote {len(all_pairs)} clips + {meta.name} to {args.out_dir}")
    print(f"Total speech: {total_minutes:.1f} min across {len(batch_files)} batch(es)")
    if mismatches:
        print("FLAGGED batches (clip count != line count, re-record these):")
        for item in mismatches:
            print(f"  - {item}")
        return 2
    print("All batches matched their line counts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
