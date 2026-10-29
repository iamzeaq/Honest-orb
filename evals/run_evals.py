"""Run the Honest Orb eval suite: 15 questions through api.brain.answer().

Checks per question: expected status matches, and (when expected_ids is
non-empty) at least one expected section id is cited. Prints pass/fail per
question plus total score and average latency, and saves full outputs to
evals/results.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.brain import VALID_IDS, answer

EVALS_DIR = ROOT / "evals"
QUESTIONS_PATH = EVALS_DIR / "questions.json"
RESULTS_PATH = EVALS_DIR / "results.json"

REQUIRED_KEYS = {
    "language",
    "status",
    "right",
    "say_this",
    "do_this",
    "citations",
    "spoken",
}


def run_one(item: dict) -> dict:
    started = time.perf_counter()
    try:
        output = answer(item["question"])
        error = None
    except Exception as exc:  # noqa: BLE001 - recorded per question
        output = None
        error = f"{type(exc).__name__}: {exc}"
    latency_s = round(time.perf_counter() - started, 2)

    notes: list[str] = []
    if error is not None:
        passed = False
        notes.append(error)
    else:
        assert output is not None
        missing = REQUIRED_KEYS - set(output)
        if missing:
            notes.append(f"missing keys: {sorted(missing)}")
        status_ok = output.get("status") == item["expected_status"]
        if not status_ok:
            notes.append(
                f"status {output.get('status')!r} != expected "
                f"{item['expected_status']!r}"
            )
        cited = [
            c.get("id")
            for c in output.get("citations", [])
            if isinstance(c, dict)
        ]
        unknown = [c for c in cited if c not in VALID_IDS]
        if unknown:
            notes.append(f"citations not in sections.json: {unknown}")
        expected = item.get("expected_ids", [])
        if expected:
            if not any(e in cited for e in expected):
                notes.append(
                    f"no expected id cited (cited={cited}, "
                    f"expected one of {expected})"
                )
        if output.get("language") not in ("en", "pcm"):
            notes.append(f"bad language: {output.get('language')!r}")
        elif output.get("language") != item.get("language"):
            notes.append(
                f"language {output.get('language')!r} != question "
                f"language {item.get('language')!r} (info only)"
            )
        passed = status_ok and not missing and not unknown and (
            not expected or any(e in cited for e in expected)
        )

    return {
        "id": item["id"],
        "language": item.get("language"),
        "question": item["question"],
        "expected_status": item["expected_status"],
        "expected_ids": item.get("expected_ids", []),
        "output": output,
        "latency_s": latency_s,
        "passed": passed,
        "notes": notes,
    }


def main() -> int:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    print(f"Running {len(questions)} evals...\n")
    results = []
    for item in questions:
        res = run_one(item)
        results.append(res)
        mark = "PASS" if res["passed"] else "FAIL"
        out = res["output"] or {}
        cited = [c.get("id") for c in (out.get("citations") or [])]
        print(
            f"[{mark}] {res['id']} ({res['language']}) "
            f"status={out.get('status')} cited={cited} "
            f"{res['latency_s']}s"
        )
        for note in res["notes"]:
            print(f"       - {note}")
        if out.get("spoken"):
            print(f"       spoken: {out['spoken'][:160]}")

    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    passed = sum(1 for r in results if r["passed"])
    latencies = [r["latency_s"] for r in results]
    avg = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    print(f"\nScore: {passed}/{len(results)} passed")
    print(f"Average latency: {avg}s")
    print(f"Full outputs saved to {RESULTS_PATH}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
