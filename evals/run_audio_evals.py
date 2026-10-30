"""Run every file in data/audio/ through the voice pipeline.

Prints transcript + status + citations + timings per file and saves the
full outputs to evals/audio_results.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.pipeline import run

AUDIO_DIR = ROOT / "data" / "audio"
RESULTS_PATH = ROOT / "evals" / "audio_results.json"


def main() -> int:
    files = sorted(
        p for p in AUDIO_DIR.iterdir() if p.is_file() and p.suffix == ".wav"
    )
    if not files:
        print(f"no .wav files in {AUDIO_DIR}", file=sys.stderr)
        return 1
    print(f"Running pipeline on {len(files)} audio file(s)...\n")
    results = []
    for path in files:
        try:
            started = time.perf_counter()
            out = run(str(path))
            out["wall_s"] = round(time.perf_counter() - started, 2)
            out["error"] = None
        except Exception as exc:  # noqa: BLE001 - recorded per file
            out = {"error": f"{type(exc).__name__}: {exc}"}
        entry = {"file": path.name, **out}
        results.append(entry)
        ans = out.get("answer", {}) if isinstance(out, dict) else {}
        print(f"--- {path.name} ---")
        print(f"transcript : {out.get('transcript')}")
        print(f"status     : {ans.get('status')} (lang={ans.get('language')})")
        print(
            "citations  : "
            + ", ".join(c.get("id", "?") for c in ans.get("citations", []))
        )
        print(f"timings    : {json.dumps(out.get('timings'))}")
        if out.get("error"):
            print(f"error      : {out['error']}")
        print()

    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    ok = sum(1 for r in results if not r.get("error"))
    print(f"{ok}/{len(results)} file(s) ok -> {RESULTS_PATH}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
