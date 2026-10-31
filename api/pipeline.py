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
from api import tts as tts_mod


def run(audio_path: str) -> dict:
    """Run the full voice pipeline. Returns transcript, answer, WAV paths.

    ``audio_out`` is the whole spoken answer as one clip; ``sentence_outs``
    holds one WAV per sentence so the server can stream the first sentence
    as soon as it is ready. ``sentences`` parallels ``sentence_outs``.
    """
    import tempfile

    timings: dict[str, float] = {}
    overall = time.perf_counter()

    started = time.perf_counter()
    stt = stt_transcribe(audio_path)
    timings["stt"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    ans = brain_answer(stt["text"])
    timings["brain"] = round(time.perf_counter() - started, 2)

    started = time.perf_counter()
    first_s: dict[str, float] = {}
    tmpdir = tempfile.mkdtemp(prefix="orb_pipe_")
    try:
        sent_paths = tts_mod.speak_sentences(
            ans["spoken"],
            out_dir=tmpdir,
            on_sentence=lambda i, _p: first_s.setdefault(
                "t", round(time.perf_counter() - started, 2)
            ),
        )
        timings["tts_first"] = first_s.get(
            "t", round(time.perf_counter() - started, 2)
        )
        fd, wav_path = tempfile.mkstemp(suffix=".wav")
        import os

        os.close(fd)
        if sent_paths:
            tts_mod.concat_wav_files(sent_paths, wav_path)
        else:
            # Nothing synthesizable: fall back to a single silent clip.
            wav_path = tts_mod.speak(ans["spoken"], out_path=wav_path)
        timings["tts"] = round(time.perf_counter() - started, 2)
    finally:
        import shutil

        # Sentence files live on: move them out before wiping the dir.
        kept: list[str] = []
        for p in sent_paths:
            fd, dest = tempfile.mkstemp(suffix=".wav")
            import os

            os.close(fd)
            shutil.move(p, dest)
            kept.append(dest)
        shutil.rmtree(tmpdir, ignore_errors=True)
        sent_paths = kept

    timings["total"] = round(time.perf_counter() - overall, 2)
    return {
        "transcript": stt["text"],
        "language_hint": stt.get("language_hint"),
        "stt_seconds": stt.get("seconds"),
        "answer": ans,
        "audio_out": wav_path,
        "sentence_outs": sent_paths,
        "sentences": tts_mod.split_sentences(ans["spoken"]),
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
