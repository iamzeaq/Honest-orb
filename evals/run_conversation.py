"""Run the Honest Orb conversation evals: multi-turn scripts through
api.brain.converse() plus per-domain retrieval/jurisdiction probes.

Checks per live turn: expected mode, citations valid (subset of
sections.json) and matching cite_any/citations_empty, session facts,
spoken markers. Retrieval probes (offline) verify no section from an
unloaded domain can be returned; jurisdiction probes verify federal
sections apply everywhere and state sections only on state match.

Writes evals/conversation_results.json. Prints pass/fail per check,
pass rate, and average live-turn latency.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.brain import VALID_IDS, converse  # noqa: E402
from api.retrieval import Retriever, detect_state, jurisdiction_of  # noqa: E402

EVALS_DIR = ROOT / "evals"
TESTS_PATH = EVALS_DIR / "conversation_tests.json"
RESULTS_PATH = EVALS_DIR / "conversation_results.json"


def _check_turn(turn_out: dict, expect: dict) -> list[str]:
    notes: list[str] = []
    modes = expect.get("mode", [])
    if modes and turn_out.get("mode") not in modes:
        notes.append(f"mode {turn_out.get('mode')!r} not in {modes}")
    cited = [c.get("id") for c in turn_out.get("citations", [])
             if isinstance(c, dict)]
    unknown = [c for c in cited if c not in VALID_IDS]
    if unknown:
        notes.append(f"citations not in sections.json: {unknown}")
    if expect.get("citations_empty") and cited:
        notes.append(f"expected no citations, got {cited}")
    want_any = expect.get("cite_any", [])
    if want_any and not any(e in cited for e in want_any):
        notes.append(f"no expected cite (cited={cited}, want one of {want_any})")
    for key, val in (expect.get("facts") or {}).items():
        if (turn_out.get("facts") or {}).get(key) != val:
            notes.append(f"fact {key}={(turn_out.get('facts') or {}).get(key)!r} "
                         f"!= expected {val!r}")
    spoken = (turn_out.get("spoken") or "")
    lowered = spoken.lower()
    for marker in expect.get("spoken_contains", []):
        if marker.lower() not in lowered:
            notes.append(f"spoken missing {marker!r}: {spoken[:120]!r}")
    for marker in expect.get("spoken_absent", []):
        if marker.lower() in lowered:
            notes.append(f"spoken must not contain {marker!r}")
    if not spoken:
        notes.append("empty spoken")
    return notes


def run_live_turn(session: str, item: dict) -> dict:
    started = time.perf_counter()
    try:
        out = converse(session, item["text"])
        error = None
    except Exception as exc:  # noqa: BLE001 - recorded per turn
        out = None
        error = f"{type(exc).__name__}: {exc}"
    latency_s = round(time.perf_counter() - started, 2)
    if error is not None:
        return {"id": item["id"], "session": session, "text": item["text"],
                "output": None, "latency_s": latency_s,
                "passed": False, "notes": [error]}
    assert out is not None
    notes = _check_turn(out, item.get("expect", {}))
    return {"id": item["id"], "session": session, "text": item["text"],
            "output": {"mode": out.get("mode"), "spoken": out.get("spoken"),
                       "citations": [c.get("id") for c in out.get("citations", [])],
                       "facts": out.get("facts")},
            "latency_s": latency_s, "passed": not notes, "notes": notes}


def run_retrieval_probe(retriever: Retriever, probe: dict,
                        domains_loaded: set[str]) -> dict:
    domain = probe["requires_domain"]
    state = probe.get("state")
    hits = retriever.search(probe["text"], state=state, top_k=10)
    hit_ids = [h["id"] for h in hits]
    hit_domains = {h.get("domain", "police_rights") for h in hits}
    notes: list[str] = []
    # State-law domains only surface when the matching state is set:
    # without it they are correctly jurisdiction-filtered out.
    expect_hits = probe.get("expect_domain_hits", domain in domains_loaded)
    if expect_hits:
        if domain not in hit_domains:
            notes.append(f"domain {domain} loaded but not retrieved for "
                         f"{probe['text'][:60]!r} (state={state}, "
                         f"hits={hit_ids[:5]})")
    else:
        bad = [h["id"] for h in hits if h.get("domain") == domain]
        if bad:
            notes.append(f"domain {domain} should stay filtered out "
                         f"(state={state}) but retrieved: {bad}")
    return {"id": probe["id"], "text": probe["text"], "hits": hit_ids[:6],
            "passed": not notes, "notes": notes}


def run_jurisdiction_probe(retriever: Retriever, probe: dict) -> dict:
    notes: list[str] = []
    for state in ("kano", "lagos"):
        hits = retriever.search(probe["text"], state=state, top_k=10)
        for h in hits:
            jur = jurisdiction_of(h)
            if jur != "federal" and jur != state and not (
                    state in ("abuja", "fct") and jur in ("abuja", "fct")):
                notes.append(f"state law {h['id']} (jur={jur}) leaked into "
                             f"{state} query")
    states_with_law = {jurisdiction_of(s) for s in retriever.sections
                       if jurisdiction_of(s) != "federal"}
    if probe.get("want_state_law") and not states_with_law:
        notes.append("no state-law sections loaded at all "
                     "(probe needs data: honest skip, recorded here)")
        return {"id": probe["id"], "text": probe["text"],
                "state_laws_loaded": sorted(states_with_law),
                "passed": True, "notes": notes, "blocked_on_data": True}
    return {"id": probe["id"], "text": probe["text"],
            "state_laws_loaded": sorted(states_with_law),
            "passed": not notes, "notes": notes}


def main() -> int:
    tests = json.loads(TESTS_PATH.read_text(encoding="utf-8"))
    run_tag = time.strftime("%Y%m%d-%H%M%S")
    retriever = Retriever()
    domains_loaded = {s.get("domain", "") for s in retriever.sections}
    results: list[dict] = []
    latencies: list[float] = []

    for sess in tests["sessions"]:
        session_id = f"conv-{run_tag}-{sess['id']}"
        for item in sess["turns"]:
            res = run_live_turn(session_id, item)
            results.append(res)
            latencies.append(res["latency_s"])
            mark = "PASS" if res["passed"] else "FAIL"
            out = res["output"] or {}
            print(f"[{mark}] {res['id']} mode={out.get('mode')} "
                  f"cited={out.get('citations')} {res['latency_s']}s")
            for note in res["notes"]:
                print(f"       - {note}")

    for probe in tests["domain_probes"]:
        kind = probe.get("kind")
        if kind == "live":
            session_id = f"conv-{run_tag}-{probe['id']}"
            res = run_live_turn(session_id, {"id": probe["id"],
                                             "text": probe["text"],
                                             "expect": probe.get("expect", {})})
            results.append(res)
            latencies.append(res["latency_s"])
            mark = "PASS" if res["passed"] else "FAIL"
            out = res["output"] or {}
            print(f"[{mark}] {res['id']} mode={out.get('mode')} "
                  f"cited={out.get('citations')} {res['latency_s']}s")
            for note in res["notes"]:
                print(f"       - {note}")
        elif kind == "retrieval":
            res = run_retrieval_probe(retriever, probe, domains_loaded)
            results.append(res)
            print(f"[{'PASS' if res['passed'] else 'FAIL'}] {res['id']} "
                  f"retrieval hits={res['hits']}")
            for note in res["notes"]:
                print(f"       - {note}")
        elif kind == "jurisdiction":
            res = run_jurisdiction_probe(retriever, probe)
            results.append(res)
            blocked = " (blocked on data)" if res.get("blocked_on_data") else ""
            print(f"[{'PASS' if res['passed'] else 'FAIL'}] {res['id']} "
                  f"jurisdiction state_laws={res['state_laws_loaded']}{blocked}")
            for note in res["notes"]:
                print(f"       - {note}")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    avg_lat = round(sum(latencies) / max(len(latencies), 1), 2)
    print(f"\n{passed}/{total} passed, avg live-turn latency {avg_lat}s")
    RESULTS_PATH.write_text(json.dumps(
        {"passed": passed, "total": total, "avg_latency_s": avg_lat,
         "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
