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

# Voice config, all from the environment (Space secrets at runtime):
# TTS_MODEL_ID (default facebook/mms-tts-pcm) and HF_TOKEN, which is only
# needed when TTS_MODEL_ID points at a private/gated Hugging Face model
# (e.g. iamzeaq/honest-orb-mms-pcm). Tokens are never hardcoded here.
#
# VITS synthesis settings (also env, no code changes needed to retune):
# TTS_NOISE_SCALE (default 0.667), TTS_NOISE_SCALE_DURATION (default 0.8),
# TTS_SPEAKING_RATE (default 1.0; >1 is faster). Lower noise values sound
# calmer/clearer; a slightly slower rate (e.g. 0.9) aids intelligibility.
#
# VOICE_STYLE selects the per-sentence post-processing chain (env, default
# "clear"): "loud" is loudness normalization only; "clear" adds an 80 Hz
# high-pass plus a gentle 2-4 kHz presence boost; "orb" is "clear" plus a
# very subtle short reverb that stays intelligible. "off" skips everything.
DEFAULT_MODEL_ID = "facebook/mms-tts-pcm"
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_NOISE_SCALE_DURATION = 0.8
DEFAULT_SPEAKING_RATE = 1.0
DEFAULT_VOICE_STYLE = "clear"


def _model_id() -> str:
    return os.environ.get("TTS_MODEL_ID", "").strip() or DEFAULT_MODEL_ID


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def _env_float(*names: str, default: float, minimum: float) -> float:
    """First finite env value >= minimum wins, else default (never raises)."""
    import math

    for name in names:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        try:
            value = float(raw.strip())
        except ValueError:
            continue
        if math.isfinite(value) and value >= minimum:
            return value
    return default


def noise_scale() -> float:
    """VITS prior noise scale (env TTS_NOISE_SCALE, default 0.667)."""
    return _env_float("TTS_NOISE_SCALE", "NOISE_SCALE",
                      default=DEFAULT_NOISE_SCALE, minimum=0.0)


def noise_scale_duration() -> float:
    """VITS duration-predictor noise scale (env, default 0.8)."""
    return _env_float("TTS_NOISE_SCALE_DURATION", "NOISE_SCALE_DURATION",
                      default=DEFAULT_NOISE_SCALE_DURATION, minimum=0.0)


def speaking_rate() -> float:
    """VITS speaking rate (env TTS_SPEAKING_RATE, default 1.0)."""
    return _env_float("TTS_SPEAKING_RATE", "SPEAKING_RATE",
                      default=DEFAULT_SPEAKING_RATE, minimum=0.05)


def voice_style() -> str:
    """Post-processing style name (env VOICE_STYLE, default "clear")."""
    return os.environ.get("VOICE_STYLE", "").strip().lower() or DEFAULT_VOICE_STYLE


def _style_flags(style: str | None = None) -> tuple[bool, bool, bool]:
    """Parse a VOICE_STYLE value into (loudness, eq, reverb) flags.

    Loudness normalization is on for every named style; "off"/"none"/"raw"
    disables the whole chain. "orb"/"reverb" implies the EQ stages.
    Separators ``+``, ``,`` and whitespace are equivalent, so
    ``VOICE_STYLE=clear+reverb`` works too.
    """
    import re as _re

    tokens = set(_re.split(r"[+,\s;]+", (style or voice_style()).lower()))
    tokens.discard("")
    if tokens & {"off", "none", "raw", "bypass"}:
        return (False, False, False)
    reverb = bool(tokens & {"orb", "reverb", "reverby", "space"})
    eq = reverb or bool(tokens & {"eq", "clear", "presence", "present", "bright"})
    # Unknown tokens (e.g. "loud"/"loudness") fall through to loudness only.
    return (True, eq, reverb)


# Keep model downloads inside the repo when the caller did not configure a
# Hugging Face cache (e.g. hosts whose default HF cache is not writable).
# The directory is also passed as cache_dir on every load: huggingface_hub
# snapshots its cache location at import time, so setting the env var here
# alone would be ignored when transformers was imported first.
_REPO_CACHE = Path(__file__).resolve().parent.parent / ".hf-cache" / "hub"


def _cache_dir() -> str:
    configured = os.environ.get("HF_HUB_CACHE", "").strip()
    if configured:
        return configured
    home = os.environ.get("HF_HOME", "").strip()
    if home:
        return str(Path(home) / "hub")
    _REPO_CACHE.mkdir(parents=True, exist_ok=True)
    return str(_REPO_CACHE)


if not os.environ.get("HF_HOME") and not os.environ.get("HF_HUB_CACHE"):
    os.environ["HF_HUB_CACHE"] = str(_REPO_CACHE)

_model = None
_tokenizer = None
_loaded_model_id: str | None = None


def _load():
    global _model, _tokenizer, _loaded_model_id
    model_id = _model_id()
    if _model is not None and _loaded_model_id == model_id:
        return _model, _tokenizer
    from transformers import AutoTokenizer, VitsModel

    # token=None means anonymous access (public models); a private model
    # id raises an auth error here unless HF_TOKEN is set in the env.
    cache_dir = _cache_dir()
    _tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=_hf_token(), cache_dir=cache_dir
    )
    _model = VitsModel.from_pretrained(model_id, token=_hf_token(), cache_dir=cache_dir)
    _model.eval()
    _loaded_model_id = model_id
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


# ---------------------------------------------------------------------------
# Loudness normalization
# ---------------------------------------------------------------------------
#
# Every synthesized sentence goes through ``normalize_loudness`` before it
# is written to WAV, so answers are clearly audible on a phone speaker:
# integrated loudness lands at about ``TARGET_LUFS`` with true peak
# limited to ``PEAK_DBFS``. Implemented in pure numpy (BS.1770-style
# gated K-weighted measurement, linear gain, peak limit) so it works in
# every runtime, including the Docker image, with no extra dependency.

TARGET_LUFS = -16.0
PEAK_DBFS = -1.0

_BLOCK_S = 0.400  # BS.1770 integration block
_HOP_S = 0.100  # 75% overlap
_ABS_GATE_LUFS = -70.0
_REL_GATE_LU = 10.0

# K-weighting corners (BS.1770 pre-filter + RLB high-pass), realized as
# RBJ cookbook biquads evaluated at the block's bin frequencies.
_SHELF_F0 = 1681.974450955533
_SHELF_GAIN_DB = 3.986843055468915
_HP_F0 = 38.13547087602444
_HP_Q = 0.5003270373238773


def _rbj_highpass(f0: float, q: float, sr: float):
    """RBJ cookbook high-pass biquad. Returns (b, a), a0 normalized to 1."""
    import math

    w0 = 2.0 * math.pi * f0 / sr
    cos_w0 = math.cos(w0)
    alpha = math.sin(w0) / (2.0 * q)
    b = [(1.0 + cos_w0) / 2.0, -(1.0 + cos_w0), (1.0 + cos_w0) / 2.0]
    a = [1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha]
    return [x / a[0] for x in b], [x / a[0] for x in a]


def _rbj_highshelf(f0: float, gain_db: float, sr: float, slope: float = 1.0):
    """RBJ cookbook high-shelf biquad. Returns (b, a), a0 normalized to 1."""
    import math

    a_gain = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / 2.0 * math.sqrt((a_gain + 1.0 / a_gain) * (1.0 / slope - 1.0) + 2.0)
    sqrt_a = math.sqrt(a_gain)
    b = [
        a_gain * ((a_gain + 1) + (a_gain - 1) * cos_w0 + 2 * sqrt_a * alpha),
        -2 * a_gain * ((a_gain - 1) + (a_gain + 1) * cos_w0),
        a_gain * ((a_gain + 1) + (a_gain - 1) * cos_w0 - 2 * sqrt_a * alpha),
    ]
    a = [
        (a_gain + 1) - (a_gain - 1) * cos_w0 + 2 * sqrt_a * alpha,
        2 * ((a_gain - 1) - (a_gain + 1) * cos_w0),
        (a_gain + 1) - (a_gain - 1) * cos_w0 - 2 * sqrt_a * alpha,
    ]
    return [x / a[0] for x in b], [x / a[0] for x in a]


def _k_weight_power(freqs, sr: float):
    """Squared K-weighting magnitude |H(f)|^2 at each frequency in freqs."""
    import numpy as np

    freqs = np.asarray(freqs, dtype=np.float64)
    w = 2.0 * np.pi * freqs / sr
    z = np.exp(-1j * w)
    z2 = z * z
    power = np.ones_like(freqs)
    for b, a in (
        _rbj_highpass(_HP_F0, _HP_Q, sr),
        _rbj_highshelf(_SHELF_F0, _SHELF_GAIN_DB, sr),
    ):
        h = (b[0] + b[1] * z + b[2] * z2) / (a[0] + a[1] * z + a[2] * z2)
        power = power * (h.real**2 + h.imag**2)
    return power


def measure_lufs(audio, sample_rate: int) -> float | None:
    """Gated integrated loudness in LUFS, or None for silence.

    BS.1770-style: K-weighted 400 ms blocks with 75% overlap, -70 LUFS
    absolute gate, -10 LU relative gate. Pure numpy, works on any mono
    float signal in [-1, 1].
    """
    import numpy as np

    x = np.asarray(audio, dtype=np.float64).ravel()
    sr = float(sample_rate)
    if x.size == 0 or sr <= 0:
        return None
    block = max(1, int(round(_BLOCK_S * sr)))
    hop = max(1, int(round(_HOP_S * sr)))
    if x.size < block:
        x = np.pad(x, (0, block - x.size))
    starts = range(0, x.size - block + 1, hop)
    frames = np.stack([x[s : s + block] for s in starts])
    spectrum = np.fft.rfft(frames, axis=1)
    freqs = np.fft.rfftfreq(block, d=1.0 / sr)
    weights = _k_weight_power(freqs, sr)
    mag2 = spectrum.real**2 + spectrum.imag**2
    n = float(block)
    if block % 2 == 0:
        energy = (
            weights[0] * mag2[:, 0]
            + 2.0 * np.sum(weights[1:-1] * mag2[:, 1:-1], axis=1)
            + weights[-1] * mag2[:, -1]
        ) / (n * n)
    else:
        energy = (
            weights[0] * mag2[:, 0]
            + 2.0 * np.sum(weights[1:] * mag2[:, 1:], axis=1)
        ) / (n * n)
    with np.errstate(divide="ignore"):
        block_lufs = -0.691 + 10.0 * np.log10(energy)
    gated = block_lufs[block_lufs > _ABS_GATE_LUFS]
    if gated.size == 0:
        return None
    rel_threshold = float(np.mean(gated)) - _REL_GATE_LU
    kept_idx = np.nonzero(block_lufs > rel_threshold)[0]
    kept_idx = kept_idx[block_lufs[kept_idx] > _ABS_GATE_LUFS]
    if kept_idx.size == 0:
        return None
    with np.errstate(divide="ignore"):
        integrated = -0.691 + 10.0 * np.log10(float(np.mean(energy[kept_idx])))
    return float(integrated) if np.isfinite(integrated) else None


def normalize_loudness(
    audio,
    sample_rate: int,
    target_lufs: float = TARGET_LUFS,
    peak_dbfs: float = PEAK_DBFS,
):
    """Loudness-normalize a mono sentence: gain to target LUFS, limit peak.

    Returns float32 audio in [-peak, peak]. Silence (or anything with no
    measurable speech energy) passes through unchanged except for the
    peak limit, so silent clips can never blow up into noise.
    """
    import numpy as np

    x = np.asarray(audio, dtype=np.float64).ravel()
    if x.size == 0:
        return x.astype(np.float32)
    x = np.clip(x, -1.0, 1.0)
    lufs = measure_lufs(x, sample_rate)
    if lufs is not None and np.isfinite(lufs):
        y = x * 10.0 ** ((target_lufs - lufs) / 20.0)
    else:
        y = x
    peak = float(np.max(np.abs(y)))
    limit = 10.0 ** (peak_dbfs / 20.0)
    if peak > limit > 0.0:
        y = y * (limit / peak)
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Voice-style EQ + "orb" ambience (pure numpy, no extra dependency)
# ---------------------------------------------------------------------------
#
# ``apply_voice_style`` runs every synthesized sentence through the chain
# selected by VOICE_STYLE: an 80 Hz high-pass (removes handling/plosive
# rumble), a gentle +3 dB presence bell centred on 3 kHz (the 2-4 kHz band
# that carries consonant intelligibility), and -- only for the "orb"
# style -- a very subtle short reverb (180 ms decaying-noise tail at about
# -20 dB wet, 12 ms pre-delay) that adds a sense of space without smearing
# speech. Loudness normalization always runs last so the -16 LUFS / -1 dBFS
# guarantee holds for every style.

HP_CUTOFF_HZ = 80.0
PRESENCE_CENTER_HZ = 3000.0
PRESENCE_GAIN_DB = 3.0
REVERB_SECONDS = 0.18
REVERB_WET = 0.10
REVERB_PREDELAY_MS = 12.0


def _spectral_multiply(audio, sample_rate: int, gains):
    """Multiply the spectrum by per-bin gains (zero-phase, same length)."""
    import numpy as np

    x = np.asarray(audio, dtype=np.float64).ravel()
    if x.size == 0:
        return x.astype(np.float32)
    spectrum = np.fft.rfft(x)
    shaped = spectrum * np.asarray(gains, dtype=np.float64)
    return np.fft.irfft(shaped, n=x.size).astype(np.float64)


def highpass_80hz(audio, sample_rate: int):
    """Second-order Butterworth high-pass at 80 Hz (zero-phase)."""
    import numpy as np

    x = np.asarray(audio, dtype=np.float64).ravel()
    if x.size == 0:
        return x.astype(np.float32)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(freqs > 0, HP_CUTOFF_HZ / np.maximum(freqs, 1e-9), np.inf)
        gains = 1.0 / np.sqrt(1.0 + ratio**4)
    gains[0] = 0.0
    return _spectral_multiply(x, sample_rate, gains).astype(np.float32)


def presence_boost(audio, sample_rate: int):
    """Gentle +3 dB bell centred on 3 kHz (2-4 kHz presence band)."""
    import numpy as np

    x = np.asarray(audio, dtype=np.float64).ravel()
    if x.size == 0:
        return x.astype(np.float32)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate))
    with np.errstate(divide="ignore", invalid="ignore"):
        octaves = np.log2(np.maximum(freqs, 1e-9) / PRESENCE_CENTER_HZ)
        bell = np.exp(-0.5 * (octaves / 0.5) ** 2)
    bell[0] = 0.0
    gains = 10.0 ** ((PRESENCE_GAIN_DB / 20.0) * bell)
    return _spectral_multiply(x, sample_rate, gains).astype(np.float32)


def orb_reverb(audio, sample_rate: int):
    """Very subtle short reverb: decaying-noise tail, low wet level.

    Deterministic (fixed seed) so the same sentence always sounds the
    same. Output length matches the input (tail is folded into the clip).
    """
    import math

    import numpy as np

    x = np.asarray(audio, dtype=np.float64).ravel()
    sr = float(sample_rate)
    if x.size == 0 or sr <= 0:
        return x.astype(np.float32)
    rng = np.random.default_rng(7)
    ir_len = max(1, int(round(REVERB_SECONDS * sr)))
    time_s = np.arange(ir_len) / sr
    tail = rng.standard_normal(ir_len) * 10.0 ** (-60.0 * time_s / (20.0 * REVERB_SECONDS))
    energy = float(np.sum(tail**2))
    if energy > 0:
        tail = tail / math.sqrt(energy)
    pre = int(round(REVERB_PREDELAY_MS / 1000.0 * sr))
    impulse = np.concatenate([np.zeros(pre), tail])
    need = x.size + impulse.size - 1
    fast_len = 1 << max(1, (need - 1).bit_length())
    wet = np.fft.irfft(np.fft.rfft(x, fast_len) * np.fft.rfft(impulse, fast_len),
                       n=fast_len)[: x.size]
    return (x + REVERB_WET * wet).astype(np.float32)


def apply_voice_style(audio, sample_rate: int, style: str | None = None):
    """Run one sentence through the VOICE_STYLE chain (EQ -> reverb? -> loud).

    Loudness normalization always runs last, so every style lands near
    -16 LUFS with peak at or below -1 dBFS. ``style="off"`` skips all
    processing and returns the input unchanged.
    """
    import numpy as np

    loud, eq, reverb = _style_flags(style)
    x = np.asarray(audio, dtype=np.float64).ravel()
    if not loud:
        return x.astype(np.float32)
    if eq:
        x = highpass_80hz(x, int(sample_rate)).astype(np.float64)
        x = presence_boost(x, int(sample_rate)).astype(np.float64)
    if reverb:
        x = orb_reverb(x, int(sample_rate)).astype(np.float64)
    return normalize_loudness(x, int(sample_rate))


def _write_wav(path: str, pcm16, sample_rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())


def _synth_normalized(normalized: str) -> tuple[list, int]:
    """Synthesize one already-normalized sentence. Returns (waves, sr).

    VITS settings come from the environment on every call (TTS_NOISE_SCALE,
    TTS_NOISE_SCALE_DURATION, TTS_SPEAKING_RATE), so retuning needs no code
    change and variant switches in one process take effect immediately.
    """
    import torch

    import numpy as np

    model, tokenizer = _load()
    rate = speaking_rate()
    # transformers >= 4.36 reads these synthesis controls off the model
    # instance (only speaking_rate is also a forward kwarg), so set all
    # three attributes here rather than only at load time.
    model.noise_scale = noise_scale()
    model.noise_scale_duration = noise_scale_duration()
    model.speaking_rate = rate
    sample_rate = int(model.config.sampling_rate)
    waves: list = []
    with torch.no_grad():
        for chunk in _chunks(normalized):
            if not chunk.strip():
                continue
            inputs = tokenizer(chunk, return_tensors="pt")
            if inputs["input_ids"].numel() == 0:
                continue
            waveform = model(**inputs, speaking_rate=rate).waveform[0].cpu().numpy()
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
    # Voice-style post-processing for every sentence (VOICE_STYLE env):
    # EQ (+ optional subtle orb reverb), then loudness normalization to
    # ~-16 LUFS peak-limited at -1 dB, so answers are clearly audible on
    # a phone speaker.
    audio = apply_voice_style(np.concatenate(waves), sample_rate)
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
