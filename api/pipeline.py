"""Honest Orb voice pipeline: audio in, grounded spoken answer out.

stt.transcribe -> brain.answer -> tts.speak(answer["spoken"]).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.brain import answer as brain_answer
from api.stt import transcribe as stt_transcribe
from api.tts import speak as tts_speak


def run(audio_path: str) -> dict:
    """Run the full voice pipeline. Returns transcript, answer, WAV path."""
    timings: dict[str, float] = {}
    overall = time.perf_counter()

    started = time.perf_counter()
    stt = stt_transcribe(audio_path)
    timings["stt"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    ans = brain_answer(stt["text"])
    timings["brain"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    wav_path = tts_speak(ans["spoken"])
    timings["tts"] = round(time.perf_counter() - started, 2)

    timings["total"] = round(time.perf_counter() - overall, 2)
    return {
        "transcript": stt["text"],
        "language_hint": stt.get("language_hint"),
        "stt_seconds": stt.get("seconds"),
        "answer": ans,
        "audio_out": wav_path,
        "timings": timings,
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python api/pipeline.py <audio file>", file=sys.stderr)
        return 2
    result = run(sys.argv[1])
    ans = result["answer"]
    print(f"transcript : {result['transcript']}")
    print(f"status     : {ans.get('status')} (lang={ans.get('language')})")
    print(
        "citations  : "
        + ", ".join(c.get("id", "?") for c in ans.get("citations", []))
    )
    print(f"spoken     : {ans.get('spoken')}")
    print(f"audio_out  : {result['audio_out']}")
    print(f"timings    : {json.dumps(result['timings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
