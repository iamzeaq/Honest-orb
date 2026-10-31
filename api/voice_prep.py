"""Prepare the Honest Orb voice dataset from recorded batches.

Reads ``data/voice/script.txt`` (180 numbered lines) and the recordings in
``data/voice/raw/`` (``batch_01.m4a`` ... ``batch_09.m4a``). The speaker read
the script in order (001->180) across the batches, but batches do NOT hold
exactly 20 lines each: there may be extra/missing lines per batch, mid-line
pauses and restarts. Do not re-record; align instead:

1. Process batches in order. Split each on silence with a short min-silence
   (~0.5 s) into small pieces.
2. Transcribe each piece with Muse Voice Transcribe (see ``api/stt.py``),
   which already applies the English language bias + Pidgin keywords.
3. Align pieces to the FULL 180-line script by text similarity, moving
   forward through the script in order (one monotonic pointer across all
   batches, never 20 per batch). Consecutive pieces are greedily merged
   into one clip when the merged transcript matches a script line better.
   Restarts, false starts and noise score low and are skipped.
4. A clip is kept only if its similarity to the matched line is high
   (normalized word overlap >= 0.6); otherwise it is dropped and logged.
   If a line was read twice, the better match wins.
5. Kept clips are trimmed of leading/trailing silence, padded with ~0.15 s
   of silence, and written as 16 kHz mono WAV.

Output: ``data/voice/dataset/`` (``wavs/line_NNN.wav`` + ``metadata.csv``
with ``file|text`` rows using the SCRIPT text, not the transcript).

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
from collections.abc import Collection
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DEFAULT = REPO_ROOT / "data" / "voice" / "script.txt"
RAW_DEFAULT = REPO_ROOT / "data" / "voice" / "raw"
OUT_DEFAULT = REPO_ROOT / "data" / "voice" / "dataset"
SAMPLE_RATE = 16000
AUDIO_EXTS = (".m4a", ".wav", ".mp3", ".ogg", ".flac", ".aac")

MIN_SILENCE_S = 0.5  # short gap: splits mid-line pauses into small pieces
MIN_SCORE = 0.6  # normalized word overlap needed to keep a clip
LOOKAHEAD = 8  # how many script lines ahead a piece may match
MAX_MERGE = 4  # max consecutive pieces merged into one clip
TRIM_PAD_S = 0.15  # silence padding on kept clips

_WORD_RE = re.compile(r"[a-z0-9']+")


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


def normalize_words(text: str) -> list[str]:
    """Lowercase alphanumeric word tokens (punctuation-insensitive)."""
    return _WORD_RE.findall(text.lower())


def word_overlap(a: list[str], b: list[str]) -> float:
    """Normalized word overlap: shared words / longer side (0..1).

    Multiset intersection (``min`` counts), so repeats count on both
    sides. Short fragments of a long line score low, which is what lets
    the aligner drop false starts and merge mid-line pauses.
    """
    if not a or not b:
        return 0.0
    counts: dict[str, int] = {}
    for w in a:
        counts[w] = counts.get(w, 0) + 1
    shared = 0
    for w in b:
        if counts.get(w, 0):
            counts[w] -= 1
            shared += 1
    return shared / max(len(a), len(b))


def align_pieces(
    transcripts: list[str],
    script: list[tuple[int, str]],
    start: int = 0,
    lookahead: int = LOOKAHEAD,
    max_merge: int = MAX_MERGE,
    min_score: float | None = None,
    re_readable: Collection[int] = (),
) -> tuple[list[tuple[int, tuple[int, int], float]], list[tuple[int, str, float]], int]:
    """Align piece transcripts to the script, moving only forward.

    Args:
        transcripts: one transcript per audio piece, in time order.
        script: ``[(line_no, text), ...]`` full script in reading order.
        start: 0-based index of the next unmatched script line.
        re_readable: 0-based line indices matched by earlier batches that
            may be re-read here (a twice-read line keeps its better match).

    Returns ``(kept, dropped, pos)`` where ``kept`` holds
    ``(line_no, (first_piece, one_past_last_piece), score)`` with at most
    one entry per line (a twice-read line keeps its better match),
    ``dropped`` holds ``(piece_idx, transcript, best_score)`` for pieces
    below threshold, and ``pos`` is the updated 0-based pointer.
    """
    thr = MIN_SCORE if min_score is None else min_score
    normed_script = [normalize_words(text) for _, text in script]
    normed_pieces = [normalize_words(t) for t in transcripts]
    best: dict[int, tuple[tuple[int, int], float]] = {}
    dropped: list[tuple[int, str, float]] = []
    pos = start
    i = 0
    while i < len(transcripts):
        window = range(pos, min(len(script), pos + lookahead))
        # Already-kept lines behind the pointer may be re-read (keep the
        # better match); unmatched lines behind it stay unreachable so the
        # alignment never moves backwards.
        candidates = list(window) + sorted(
            j for j in set(re_readable) | set(best) if 0 <= j < pos
        )

        def window_best(words: list[str]) -> tuple[int, float]:
            line_idx, score = pos, 0.0
            for j in candidates:
                s = word_overlap(words, normed_script[j])
                if s > score:
                    line_idx, score = j, s
            return line_idx, score

        line_idx, score = window_best(normed_pieces[i])
        span = (i, i + 1)
        # Greedily merge following pieces while the merged transcript
        # matches some script line better than the pieces do alone.
        merged_words = list(normed_pieces[i])
        for k in range(i + 1, min(len(transcripts), i + max_merge)):
            merged_words = merged_words + normed_pieces[k]
            m_line, m_score = window_best(merged_words)
            if m_score > score:
                line_idx, score, span = m_line, m_score, (i, k + 1)
        # A leading restart inflates a winning span (restart + clean
        # re-read): shrink from the left while the trailing pieces match
        # the same line better on their own, logging the restart pieces.
        while span[1] - span[0] > 1:
            tail = [w for p in range(span[0] + 1, span[1]) for w in normed_pieces[p]]
            tail_score = word_overlap(tail, normed_script[line_idx])
            if tail_score > score:
                _, single_best = window_best(normed_pieces[span[0]])
                dropped.append((span[0], transcripts[span[0]], single_best))
                span = (span[0] + 1, span[1])
                score = tail_score
            else:
                break
        if score >= thr:
            prev = best.get(line_idx)
            if prev is None or score > prev[1]:
                best[line_idx] = (span, score)
            pos = max(pos, line_idx + 1)
            i = span[1]
        else:
            dropped.append((i, transcripts[i], score))
            i += 1
    kept = [
        (script[j][0], span, score)
        for j, (span, score) in sorted(best.items())
    ]
    return kept, dropped, pos


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


def split_ranges(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: int = 20,
    threshold_ratio: float = 0.05,
    floor: float = 0.01,
    min_silence_s: float = MIN_SILENCE_S,
    min_clip_s: float = 0.2,
) -> list[tuple[int, int]]:
    """Speech (start, end) sample ranges split on silent gaps (energy VAD)."""
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
                clips.append((start * frame, (i - silence_run + 1) * frame))
                start = None
                silence_run = 0
    if start is not None:
        clips.append((start * frame, len(audio)))
    return [
        (s, e) for s, e in clips if (e - s) / sr >= min_clip_s
    ]


def split_on_silence(audio: np.ndarray, **kwargs) -> list[np.ndarray]:
    """Split audio into speech chunks (see :func:`split_ranges`)."""
    return [audio[s:e] for s, e in split_ranges(audio, **kwargs)]


def trim_and_pad(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Trim edge silence, normalize, and add ~0.15 s silence padding.

    Trims from the first speech onset to the last speech end: a matched
    span may start with a noise burst separated by a gap from the real
    reading, and keeping only the first fragment would write a noise
    clip whose transcript (of the whole span) matched fine.
    """
    ranges = split_ranges(audio, sr=sr, min_silence_s=0.1, min_clip_s=0.0)
    chunk = audio[ranges[0][0]:ranges[-1][1]] if ranges else audio
    chunk = chunk.astype(np.float64)
    chunk -= chunk.mean()  # drop DC offset from the phone mic
    peak = float(np.abs(chunk).max())
    if peak > 1e-6:
        chunk = chunk / peak * 0.9  # consistent loudness per clip
    pad = np.zeros(int(TRIM_PAD_S * sr), dtype=np.float64)
    return np.concatenate([pad, chunk, pad]).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm16.tobytes())


def transcribe_piece(audio: np.ndarray, tmp_dir: Path, tag: str) -> str:
    """Transcribe one audio piece via Muse Voice Transcribe (api/stt.py)."""
    try:
        from stt import transcribe as stt_transcribe  # run as api/voice_prep.py
    except ImportError:  # run as a module: python -m api.voice_prep
        from api.stt import transcribe as stt_transcribe
    piece = tmp_dir / f"{tag}.wav"
    write_wav(piece, audio)
    try:
        return str(stt_transcribe(piece).get("text", "") or "").strip()
    finally:
        piece.unlink(missing_ok=True)


class TranscribeOutage(Exception):
    """Raised when STT fails systemically: abort, do not write a dataset."""


def check_transcribe_health(n_pieces: int, n_failed: int, batch_name: str) -> None:
    """Abort a batch when most pieces fail to transcribe (API outage).

    A few failures are normal (noise pieces); a majority means the STT
    service is unreachable, and continuing would silently produce an
    empty or garbage dataset.
    """
    if n_pieces and n_failed > n_pieces / 2:
        raise TranscribeOutage(
            f"{batch_name}: {n_failed}/{n_pieces} pieces failed to transcribe; "
            "aborting instead of writing a partial dataset"
        )


def process_batch(
    batch_file: Path,
    script: list[tuple[int, str]],
    pos: int,
    wavs_dir: Path,
    tmp_dir: Path,
    kept_global: dict[int, tuple[np.ndarray, float]],
    verbose: bool = True,
) -> tuple[list[int], int, list[tuple[int, str, float]], int]:
    """Split, transcribe and monotonically align one batch.

    Returns ``(matched_lines, n_pieces, dropped, pos)``. Audio for kept
    clips is staged in ``kept_global`` (line_no -> (audio, score)) so a
    later, better reading of the same line can replace an earlier one.
    """
    conv = tmp_dir / f"{batch_file.stem}_16k.wav"
    to_mono_16k(batch_file, conv)
    audio = read_wav_mono(conv)
    ranges = split_ranges(audio)
    n_pieces = len(ranges)
    if verbose:
        print(f"  split into {n_pieces} pieces, transcribing ...", flush=True)
    transcripts: list[str] = []
    n_failed = 0
    for idx, (s, e) in enumerate(ranges):
        try:
            text = transcribe_piece(audio[s:e], tmp_dir, f"{batch_file.stem}_{idx:03d}")
        except Exception as exc:  # network/API failure: drop this piece
            print(f"  [piece {idx}] transcribe FAILED ({exc}); skipped")
            text = ""
            n_failed += 1
        transcripts.append(text)
        if verbose and text:
            print(f"  [piece {idx}] {text[:90]}", flush=True)
    check_transcribe_health(n_pieces, n_failed, batch_file.name)

    re_readable = {
        j for j, (line_no, _) in enumerate(script) if line_no in kept_global
    }
    kept, dropped, pos = align_pieces(
        transcripts, script, start=pos, re_readable=re_readable
    )
    matched: list[int] = []
    for line_no, (first, past_last), score in kept:
        s0 = ranges[first][0]
        e0 = ranges[past_last - 1][1]
        clip = trim_and_pad(audio[s0:e0])
        prev = kept_global.get(line_no)
        if prev is None or score > prev[1]:
            kept_global[line_no] = (clip, score)
        matched.append(line_no)
    for idx, text, score in dropped:
        shown = text if text else "<empty>"
        print(f"  [piece {idx}] DROPPED (score {score:.2f}): {shown[:90]}")
    _ = wavs_dir  # clips are written once at the end (dupes resolved)
    return sorted(matched), n_pieces, dropped, pos


def main(argv: list[str] | None = None) -> int:
    global MIN_SCORE
    parser = argparse.ArgumentParser(description="Build data/voice/dataset from raw batches.")
    parser.add_argument("--script", type=Path, default=SCRIPT_DEFAULT)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--min-score", type=float, default=MIN_SCORE)
    args = parser.parse_args(argv)
    MIN_SCORE = args.min_score

    try:
        script = parse_script(args.script)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"Script: {len(script)} lines from {args.script}")
    by_no = {no: text for no, text in script}

    batch_files = sorted(
        p for p in args.raw_dir.glob("batch_*") if p.suffix.lower() in AUDIO_EXTS
    ) if args.raw_dir.is_dir() else []
    if not batch_files:
        print(f"No batch files (batch_*.m4a) found in {args.raw_dir}.")
        print("Record them per data/voice/README.md, then re-run.")
        return 1

    wavs_dir = args.out_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    kept_global: dict[int, tuple[np.ndarray, float]] = {}
    pos = 0
    batch_reports: list[tuple[str, list[int], int]] = []
    total_dropped = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for batch_file in batch_files:
            print(f"\n{batch_file.name}:")
            try:
                matched, n_pieces, dropped, pos = process_batch(
                    batch_file, script, pos, wavs_dir, tmp_dir, kept_global,
                )
            except TranscribeOutage as exc:
                # Systemic STT failure: abort without touching the dataset.
                print(f"ERROR: {exc}")
                return 1
            except (RuntimeError, ValueError, OSError) as exc:
                print(f"  ERROR: {exc}")
                batch_reports.append((batch_file.name, [], 0))
                continue
            total_dropped += len(dropped)
            batch_reports.append((batch_file.name, matched, len(matched)))
            if matched:
                print(f"  -> lines {matched[0]:03d}..{matched[-1]:03d}, "
                      f"{len(matched)} clip(s) kept from {n_pieces} pieces")
            else:
                print(f"  -> no lines matched ({n_pieces} pieces, all dropped)")

    # Clear stale clips only after successful processing, so an aborted
    # run can never leave a half-written or wiped dataset behind.
    for stale in wavs_dir.glob("line_*.wav"):
        if stale.stem[5:].isdigit() and int(stale.stem[5:]) not in kept_global:
            stale.unlink()

    total_seconds = 0.0
    for line_no in sorted(kept_global):
        clip, _ = kept_global[line_no]
        write_wav(wavs_dir / f"line_{line_no:03d}.wav", clip)
        total_seconds += len(clip) / SAMPLE_RATE

    meta = args.out_dir / "metadata.csv"
    with meta.open("w", encoding="utf-8") as f:
        f.write("file|text\n")
        for line_no in sorted(kept_global):
            f.write(f"wavs/line_{line_no:03d}.wav|{by_no[line_no]}\n")

    matched_set = set(kept_global)
    missing = [no for no, _ in script if no not in matched_set]

    print("\n==== per-batch coverage ====")
    for name, matched, n_kept in batch_reports:
        span = f"{matched[0]:03d}..{matched[-1]:03d}" if matched else "--"
        print(f"{name:<14} lines {span}  clips kept: {n_kept}")
    print(f"\nTotal clips kept: {len(kept_global)} / {len(script)} "
          f"({total_dropped} pieces dropped)")
    print(f"Total speech: {total_seconds / 60.0:.1f} min across "
          f"{len(batch_files)} batch(es)")
    if missing:
        print(f"Never matched ({len(missing)}): "
              + ", ".join(f"{n:03d}" for n in missing))
        return 2
    print("Every script line was matched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
