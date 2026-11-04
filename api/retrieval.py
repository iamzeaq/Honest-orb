"""Honest Orb retriever: keyword/BM25-lite search over data/sections.json.

No embeddings, no non-Meta models — pure-Python token scoring (IDF-weighted
term overlap across id/source/title/text/keywords). Federal sections always
apply; state-law sections apply only when the session state matches.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECTIONS_PATH = ROOT / "data" / "sections.json"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Nigerian states + FCT for jurisdiction detection (light aliases included).
STATES = {
    "abia", "adamawa", "akwa ibom", "anambra", "bauchi", "bayelsa", "benue",
    "borno", "cross river", "delta", "ebonyi", "edo", "ekiti", "enugu",
    "gombe", "imo", "jigawa", "kaduna", "kano", "katsina", "kebbi", "kogi",
    "kwara", "lagos", "nasarawa", "niger", "ogun", "ondo", "osun", "oyo",
    "plateau", "rivers", "sokoto", "taraba", "yobe", "zamfara", "fct",
    "abuja",
}
_STATE_ALIAS = {"federal capital territory": "fct"}


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def detect_state(text: str) -> str | None:
    """Return a lowercase state/fct name found in text, else None."""
    lowered = (text or "").lower()
    for alias, canon in _STATE_ALIAS.items():
        if alias in lowered:
            return canon
    for state in STATES:
        if re.search(r"\b" + re.escape(state) + r"\b", lowered):
            return state
    return None


def jurisdiction_of(section: dict) -> str:
    return str(section.get("jurisdiction", "federal") or "federal").lower()


def is_state_sensitive(text: str) -> bool:
    """True when the question turns on state law (tenancy/eviction/rent)."""
    lowered = (text or "").lower()
    return bool(re.search(
        r"\b(landlord|tenant|tenancy|evict|eviction|quit notice|notice to quit|"
        r"rent|house rent|shop rent|recover possession|my landlord)\b", lowered))


class Retriever:
    def __init__(self, sections: list[dict] | None = None):
        if sections is None:
            with open(SECTIONS_PATH, encoding="utf-8") as fh:
                sections = json.load(fh)
        self.sections = sections
        # Precompute IDF over section texts + keywords.
        df: dict[str, int] = {}
        self._doc_tokens: list[list[str]] = []
        for s in self.sections:
            toks = _tokens(
                f"{s.get('source','')} {s.get('title','')} {s.get('text','')} "
                + " ".join(s.get("keywords", []))
            )
            self._doc_tokens.append(toks)
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = max(len(self.sections), 1)
        self._idf = {t: math.log(1 + n / (1 + c)) for t, c in df.items()}

    def search(self, query: str, state: str | None = None,
               top_k: int = 10) -> list[dict]:
        """Return up to top_k sections, jurisdiction-filtered.

        State-law sections are excluded unless they match `state`.
        """
        qtoks = _tokens(query)
        scored: list[tuple[float, dict]] = []
        for s, dtoks in zip(self.sections, self._doc_tokens):
            jur = jurisdiction_of(s)
            if jur != "federal":
                if state is None or state != jur:
                    # Abuja/FCT alias: match "abuja" queries to fct sections.
                    if not (state in ("abuja", "fct") and jur in ("abuja", "fct")):
                        continue
            dset = set(dtoks)
            score = 0.0
            for t in set(qtoks):
                if t in dset:
                    score += self._idf.get(t, 0.0)
            # Small boost for keyword-field hits (curated relevance).
            kw = set(_tokens(" ".join(s.get("keywords", []))))
            overlap = kw & set(qtoks)
            score += 1.5 * sum(self._idf.get(t, 0.0) for t in overlap)
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda p: -p[0])
        return [s for _, s in scored[: max(top_k, 1)]]

    def state_laws_available(self, state: str | None) -> bool:
        """Is there any state-specific law loaded for this state?"""
        if not state:
            return False
        norm = {"abuja": "fct"}.get(state, state)
        return any(
            {"abuja": "fct"}.get(jurisdiction_of(s), jurisdiction_of(s)) == norm
            for s in self.sections
        )
