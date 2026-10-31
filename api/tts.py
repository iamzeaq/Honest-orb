"""Honest Orb mouth: text -> WAV file via Meta MMS-TTS.

Uses facebook/mms-tts-pcm (VitsModel) from Hugging Face Transformers,
running on CPU, for both English and Pidgin text. The model is loaded once
and reused. Output is mono 16-bit PCM WAV.

Clarity: every synthesis path runs the text through
``normalize_for_speech`` first, because the mms-tts-pcm tokenizer vocab
(37 tokens) silently drops anything it does not know:

- missing letter ``q`` -> mapped to the closest sound ``k``
- missing digits ``6``/``8``/``9`` -> every number becomes words first,
  so no digit ever reaches the tokenizer
- uppercase is fine (the tokenizer lowercases), ``-`` and ``'`` are
  kept, everything else (brackets, ``#*/%`` ...) is stripped

Speed: ``speak_sentences`` / ``speak_sentence`` synthesize one WAV per
sentence so the server can hand the first sentence's audio to the browser
as soon as it is ready.
"""

from __future__ import annotations

import os
import re
import time
import wave
from pathlib import Path

# Single config value for the voice: swap in a fine-tuned checkpoint here.
MODEL_ID = os.environ.get("TTS_MODEL_ID", "facebook/mms-tts-pcm")

# Keep model downloads inside the repo when the caller did not configure a
# Hugging Face cache (e.g. hosts whose default HF cache is not writable).
if not os.environ.get("HF_HOME") and not os.environ.get("HF_HUB_CACHE"):
    _repo_cache = Path(__file__).resolve().parent.parent / ".hf-cache"
    _repo_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HUB_CACHE"] = str(_repo_cache / "hub")

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is not None:
        return _model, _tokenizer
    from transformers import AutoTokenizer, VitsModel

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    _model = VitsModel.from_pretrained(MODEL_ID)
    _model.eval()
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# Speech normalization
# ---------------------------------------------------------------------------

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = {
    2: "twenty", 3: "thirty", 4: "forty", 5: "fifty",
    6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety",
}


def _num_words(n: int) -> str:
    """Cardinal words for a non-negative int (supports large values)."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]}-{_ONES[ones]}"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        tail = "" if rest == 0 else " " + _num_words(rest)
        return f"{_ONES[hundreds]} hundred{tail}"
    if n < 1_000_000:
        thousands, rest = divmod(n, 1000)
        tail = "" if rest == 0 else " " + _num_words(rest)
        return f"{_num_words(thousands)} thousand{tail}"
    return " ".join(_ONES[int(d)] for d in str(n))


_SECTION_RANGE_SS = re.compile(r"\bss\.?\s*(\d+)\s*[-\u2013\u2014]\s*(\d+)", re.IGNORECASE)
_SECTION_RANGE_S = re.compile(r"\bs\.\s*(\d+)\s*[-\u2013\u2014]\s*(\d+)", re.IGNORECASE)
_SECTION_SINGLE_SS = re.compile(r"\bss\.?\s*(\d+)", re.IGNORECASE)
_SECTION_SINGLE_S = re.compile(r"\bs\.\s*(\d+)", re.IGNORECASE)
_SECTION_BARE_S = re.compile(r"\bs(\d{1,4})\b", re.IGNORECASE)
_GENERIC_RANGE = re.compile(r"\b(\d+)\s*[-\u2013\u2014]\s*(\d+)")
_DECIMAL = re.compile(r"\b(\d+)\.(\d+)\b")
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_NUMBER_AFTER = re.compile(r"\d+")


def normalize_for_speech(text: str) -> str:
    """Rewrite text into words the mms-tts-pcm tokenizer can speak.

    Numbers become words (``35`` -> ``thirty-five``), the emergency
    number is read digit by digit (``112`` -> ``one one two``), legal
    short-hands expand (``s.35`` -> ``section thirty-five``,
    ``ss.49-51`` -> ``sections forty-nine to fifty-one``), brackets and
    symbols are stripped, and common abbreviations are expanded. The
    result only contains tokenizer-safe characters (``a``-``z`` except
    ``q``, space, apostrophe, hyphen); the missing letter ``q`` is mapped
    to its closest supported sound ``k``.
    """
    text = text.lower()

    # Legal section shorthands first (they consume their digits).
    text = _SECTION_RANGE_SS.sub(
        lambda m: f"sections {_num_words(int(m.group(1)))} to "
        f"{_num_words(int(m.group(2)))}",
        text,
    )
    text = _SECTION_RANGE_S.sub(
        lambda m: f"sections {_num_words(int(m.group(1)))} to "
        f"{_num_words(int(m.group(2)))}",
        text,
    )
    text = _SECTION_SINGLE_SS.sub(
        lambda m: f"sections {_num_words(int(m.group(1)))}", text
    )
    text = _SECTION_SINGLE_S.sub(
        lambda m: f"section {_num_words(int(m.group(1)))}", text
    )
    text = _SECTION_BARE_S.sub(
        lambda m: f"section {_num_words(int(m.group(1)))}", text
    )

    # Emergency number: read digit by digit, never "one hundred twelve".
    text = re.sub(r"\b112\b", "one one two", text)

    # Units and abbreviations.
    text = re.sub(r"(\d)\s*km\b", r"\1 kilometers", text)
    text = text.replace("km", "kilometers")
    text = re.sub(r"\be\.g\.", "for example", text)
    text = re.sub(r"\bi\.e\.", "that is", text)
    text = re.sub(r"\betc\.?", "etcetera", text)
    text = re.sub(r"\bmrs\.?(?!\w)", "missus", text)
    text = re.sub(r"\bmr\.?(?!\w)", "mister", text)
    text = re.sub(r"\bdr\.?(?!\w)", "doctor", text)
    text = re.sub(r"\bvs\.?(?!\w)", "versus", text)
    text = re.sub(
        r"\bno\.\s*(\d+)", lambda m: f"number {_num_words(int(m.group(1)))}", text
    )
    text = text.replace("§", " section ")
    text = text.replace("&", " and ")
    text = text.replace("%", " percent ")
    text = text.replace("+", " plus ")
    text = text.replace("=", " equals ")
    text = text.replace("@", " at ")
    text = text.replace("/", " or ")
    text = text.replace("₦", " naira ")
    text = re.sub(r"\bngn\b", "naira", text)
    text = text.replace("$", " dollars ")
    text = text.replace("°", " degrees ")

    # Numbers: strip thousand separators, read decimals digit by digit
    # after the point, ordinals as cardinals, then plain integers.
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = _DECIMAL.sub(
        lambda m: f"{_num_words(int(m.group(1)))} point "
        + " ".join(_ONES[int(d)] for d in m.group(2)),
        text,
    )
    text = _ORDINAL.sub(lambda m: _num_words(int(m.group(1))), text)
    text = _GENERIC_RANGE.sub(
        lambda m: f"{_num_words(int(m.group(1)))} to "
        f"{_num_words(int(m.group(2)))}",
        text,
    )
    text = _NUMBER_AFTER.sub(lambda m: _num_words(int(m.group())), text)

    # Missing vocab letter -> closest supported sound.
    text = text.replace("q", "k")

    # Drop anything the tokenizer would silently swallow anyway.
    text = re.sub(r"[^a-z' \-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def split_sentences(text: str) -> list[str]:
    """Split spoken text into sentences for per-sentence synthesis."""
    return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]


def _chunks(text: str, limit: int = 400) -> list[str]:
    """Split long text on sentence boundaries for stable synthesis."""
    sentences = split_sentences(text)
    out: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= limit:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                out.append(current)
            current = sentence
    if current:
        out.append(current)
    return out or [text]


def _write_wav(path: str, pcm16, sample_rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())


def _synth_normalized(normalized: str) -> tuple[list, int]:
    """Synthesize one already-normalized sentence. Returns (waves, sr)."""
    import torch

    import numpy as np

    model, tokenizer = _load()
    sample_rate = int(model.config.sampling_rate)
    waves: list = []
    with torch.no_grad():
        for chunk in _chunks(normalized):
            if not chunk.strip():
                continue
            inputs = tokenizer(chunk, return_tensors="pt")
            if inputs["input_ids"].numel() == 0:
                continue
            waveform = model(**inputs).waveform[0].cpu().numpy()
            waves.append(waveform)
    return waves, sample_rate


def speak_sentence(text: str, out_path: str | None = None) -> str | None:
    """Synthesize a single sentence to a WAV file. Returns path or None."""
    import tempfile

    import numpy as np

    normalized = normalize_for_speech(text)
    if not normalized:
        return None
    waves, sample_rate = _synth_normalized(normalized)
    if not waves:
        return None
    audio = np.concatenate(waves)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".wav")
        import os

        os.close(fd)
    _write_wav(str(out_path), pcm16, sample_rate)
    return str(out_path)


def speak_sentences(
    text: str,
    out_dir: str | None = None,
    on_sentence=None,
) -> list[str]:
    """Synthesize each sentence to its own WAV file. Returns file paths.

    ``on_sentence`` (optional) is called as ``on_sentence(index, path)``
    right after each sentence file is written, so a server can hand the
    first sentence's audio to the browser while later ones still render.
    """
    import tempfile

    sentences = split_sentences(text)
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="orb_sent_")
    paths: list[str] = []
    for i, sentence in enumerate(sentences):
        path = speak_sentence(sentence, str(Path(out_dir) / f"s{i:02d}.wav"))
        if path is None:
            continue
        paths.append(path)
        if on_sentence is not None:
            on_sentence(i, path)
    return paths


def concat_wav_files(paths: list[str], out_path: str) -> str:
    """Concatenate same-format WAV files into one. Returns out_path."""
    frames_list: list[bytes] = []
    sample_rate = 0
    for path in paths:
        with wave.open(path, "rb") as w:
            assert w.getnchannels() == 1 and w.getsampwidth() == 2
            sample_rate = sample_rate or w.getframerate()
            assert w.getframerate() == sample_rate
            frames_list.append(w.readframes(w.getnframes()))
    pcm = b"".join(frames_list)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(out_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate or 16000)
        w.writeframes(pcm)
    return out_path


def speak(text: str, out_path: str | None = None) -> str:
    """Synthesize text to a WAV file. Returns the file path."""
    import tempfile

    import numpy as np

    started = time.perf_counter()
    tmpdir = tempfile.mkdtemp(prefix="orb_speak_")
    try:
        paths = speak_sentences(text, out_dir=tmpdir)
        if not paths:
            # Nothing synthesizable: write a short silence clip.
            model, _ = _load()
            sample_rate = int(model.config.sampling_rate)
            pcm16 = np.zeros(sample_rate // 2, dtype=np.int16)
            if out_path is None:
                fd, out_path = tempfile.mkstemp(suffix=".wav")
                import os

                os.close(fd)
            _write_wav(str(out_path), pcm16, sample_rate)
            return str(out_path)
        if out_path is None:
            fd, out_path = tempfile.mkstemp(suffix=".wav")
            import os

            os.close(fd)
        concat_wav_files(paths, str(out_path))
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed = time.perf_counter() - started
    print(f"TTS: synthesized {len(text)} chars in {elapsed:.2f}s -> {out_path}")
    return str(out_path)
