"""Honest Orb brain: text question in, grounded answer out.

Uses the 35 sections in data/sections.json as a fixed system-prompt prefix
(so prompt caching works) plus the user's question, and returns a structured
answer via the Meta API (OpenAI-compatible SDK).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
SECTIONS_PATH = ROOT / "data" / "sections.json"
ENV_PATH = ROOT / ".env"

MODEL = "muse-spark-1.2-contributor"
BASE_URL = "https://api.meta.ai/v1"
MAX_OUTPUT_TOKENS = 600
MAX_OUTPUT_TURNS_TOKENS = 800

_client = None


def _load_env_file() -> None:
    """Load KEY=VALUE lines from the repo .env without extra dependencies."""
    try:
        text = ENV_PATH.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _load_env_file()
        api_key = os.environ.get("MODEL_API_KEY", "")
        if not api_key:
            raise RuntimeError("MODEL_API_KEY is not set (expected in .env)")
        _client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return _client


def _load_sections() -> list[dict]:
    with open(SECTIONS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


SECTIONS: list[dict] = _load_sections()
VALID_IDS: set[str] = {s["id"] for s in SECTIONS}


def _citation_label(section: dict) -> str:
    return f"{section['source']}, s.{section['section']}"


def _build_system_prompt() -> str:
    ids = ", ".join(s["id"] for s in SECTIONS)
    parts = [
        "You are Honest Orb, a calm assistant that tells everyday Nigerians "
        "their legal rights during police encounters, in English or natural "
        "Nigerian Pidgin.",
        "",
        "Answer in the SAME language as the user's question: English for "
        "English questions, natural Nigerian Pidgin for Pidgin questions "
        "(fluent Pidgin a Nigerian would speak, never broken English). "
        "Pidgin markers include dey, dem, una, wan, wetin, abeg, abi, na, "
        "suppose be, carry go, don, fit, and get used for have: a question "
        "containing these is Pidgin even if it is short.",
        "",
        "Rules:",
        "- Answer ONLY from the LEGAL SECTIONS below. Do not use outside "
        "knowledge. Every answered or emergency response must cite at least "
        "one section id from VALID IDS, using labels like "
        '"Police Act 2020, s.35" or "Constitution 1999, s.35". '
        "The citations list must never be empty for those statuses.",
        "- If the sections do not clearly cover the question: status "
        '"not_sure", and say (in the user\'s language) to contact the Legal '
        "Aid Council of Nigeria or a lawyer.",
        "- If the user mentions violence, injury or immediate danger: status "
        '"emergency", tell them to call 112 first, then give the right that '
        "applies.",
        "- Off-topic questions (not about rights during police encounters): "
        "status \"off_topic\", and politely say you only help with rights "
        "during police encounters.",
        "- Calm, plain words. This is legal information, not legal advice. "
        "Keep every text field short: right, say_this and do_this are each "
        "one sentence of at most 20 words; spoken is at most 3 short "
        "sentences.",
        "- Precision: when a time limit, number or condition appears in a "
        "cited section, state it exactly as the section says. Example: "
        "detention before court is 1 day where a court of competent "
        "jurisdiction is within 40 km, otherwise 2 days or a longer period "
        "the court considers reasonable (Constitution s.35(4)-(5)). Never "
        "simplify to 24 hours if the law says otherwise.",
        "- No overclaiming: only state what the cited section text actually "
        "says. If the law does not mention something specifically (for "
        "example phones), say what the general right is (for example "
        "privacy of communications, s.37) without inventing detail.",
        "- Safety first: always advise staying calm and polite. Never advise "
        "refusing, arguing or resisting physically. Pattern: state your "
        "right politely, ask for the officer's name or number and station, "
        "and challenge it later through a lawyer or the Legal Aid Council.",
        "",
        "For the spoken field: 2-3 short sentences combining right, say_this "
        "and do_this, written for text-to-speech. No symbols such as #, *, "
        "or /. Never read bare section numbers aloud; refer to laws by name "
        "(for example \"the Police Act\" or \"the Constitution\"). In an "
        "emergency response the spoken field must begin with the instruction "
        "to call 112.",
        "",
        f"VALID IDS: {ids}",
        "",
        "LEGAL SECTIONS:",
    ]
    for s in SECTIONS:
        keywords = ", ".join(s.get("keywords", []))
        parts.append(
            f"[{s['id']}] {_citation_label(s)} - {s['title']}\n"
            f"{s['text']}\n"
            f"Keywords: {keywords}"
        )
    return "\n\n".join(parts)


# Fixed prefix: identical on every call so prompt caching applies.
SYSTEM_PROMPT: str = _build_system_prompt()


def _answer_schema():
    from pydantic import BaseModel

    class Citation(BaseModel):
        id: str
        label: str

    class BrainAnswer(BaseModel):
        language: Literal["en", "pcm"]
        status: Literal["answered", "not_sure", "emergency", "off_topic"]
        right: str
        say_this: str
        do_this: str
        citations: list[Citation]
        spoken: str

    return BrainAnswer


CONVERSE_PREFIX = """You are Honest Orb — "your legal companion. Talk to it like a friend; it answers like the law."

Scope: a LEGAL COMPANION for Nigerians (rights, legal issues, civic rights/duties in Nigeria).
Conversational and friendly: greetings, small talk and empathy are welcome. For unrelated topics
(sports, recipes, general trivia) use mode "redirect": one warm friendly line, then gently steer
back to how you can help with rights, legal or civic matters. You are NOT a police-only chatbot.

Per-turn JSON modes:
- "chat": warm natural small talk or friendly acknowledgement. Never force law into chat.
- "clarify": ask the ONE most important missing question (e.g. which state, when it happened).
  Spoken is just that question plus a warm half-line. No citations needed.
- "legal": the turn is a legal/civic issue AND the RETRIEVED SECTIONS below clearly cover it.
  Ground every claim in cited sections only. Cite the section ids you used.
- "emergency": user mentions violence, injury, torture, confession under beating, or immediate
  danger. Spoken MUST begin with calling 112 first, then the right that applies.
- "not_covered": a legal/civic issue the retrieved sections do NOT clearly cover (including a
  state-law issue for a state with no loaded law). Say honestly you don't have that law loaded
  yet and point to the Legal Aid Council of Nigeria or a lawyer. No citations, never invent law.
- "redirect": unrelated topic. One warm friendly line + steer back to rights/legal/civic help.

Rules (carry-over, never break):
- Precision: state time limits/numbers/conditions EXACTLY as the cited section says. Example:
  detention before court is 1 day where a court of competent jurisdiction is within 40 km,
  otherwise 2 days or a longer period the court considers reasonable (Constitution s.35(4)-(5)).
  Never simplify to "24 hours" if the law says otherwise.
- No overclaiming: only state what the cited text actually says. If the law does not mention
  something specifically (for example phones), say the general right (for example privacy of
  communications, s.37) without inventing detail. A section that mentions a topic only in
  passing does NOT cover a general question about that topic: e.g. a suspect's right to a
  lawyer at the station does not cover "how do I get a free lawyer for my court case"; freedom
  of expression does not cover "how do I request records from a ministry". Such turns are
  not_covered.
- facts.state is a lowercase Nigerian state name (or "fct") when the user has named one,
  otherwise the EMPTY string. Never write "unknown" or "none".
- Safety first: always advise staying calm and polite. Never advise refusing, arguing or
  resisting physically. Pattern: state the right politely, ask for the officer's name and
  station, challenge it later through a lawyer or the Legal Aid Council.
- Resolve references ("can he do that?", "what about my case?") from the conversation summary
  and recent turns. Handle topic switches cleanly; track the current topic in facts.
- Understand Pidgin input, but ALWAYS reply in clear Nigerian English (calm, warm).
- Spoken style: Nigerian English, calm, warm, 1-3 short sentences, written for text-to-speech.
  No symbols (#, *, /). Never read bare section numbers aloud; refer to laws by name
  (e.g. "the Tenancy Law", "the Constitution", "the Anti-Torture Act").
- legal/emergency details: right, what_to_say, what_to_do — each one sentence, ≤20 words.
- Keep reasoning minimal."""


def _converse_schema():
    from pydantic import BaseModel

    class ConverseCitation(BaseModel):
        id: str
        label: str

    class ConverseDetails(BaseModel):
        right: str = ""
        what_to_say: str = ""
        what_to_do: str = ""

    class ConverseFacts(BaseModel):
        state: str = ""
        topic: str = ""

    class ConverseAnswer(BaseModel):
        mode: str = "chat"
        spoken: str = ""
        details: ConverseDetails = ConverseDetails()
        citations: list[ConverseCitation] = []
        facts: ConverseFacts = ConverseFacts()
        summary: str = ""

    return ConverseAnswer


# Server-side session memory: session_id -> {turns, summary, facts}.
# (In-process; a server restart clears it and the browser simply continues
# with a fresh summary — session_id persists in the browser.)
_SESSIONS: dict[str, dict] = {}
_SESSION_LOCK = None


def _session_lock():
    global _SESSION_LOCK
    if _SESSION_LOCK is None:
        import threading

        _SESSION_LOCK = threading.Lock()
    return _SESSION_LOCK


def _clean_state(value: str | None) -> str:
    """Normalize a state fact: lowercase name/"fct", else empty.

    The model sometimes writes "unknown"/"none" — those mean no state and
    must never persist, or later turns misread jurisdiction.
    """
    v = (value or "").strip().lower()
    if v in ("", "unknown", "none", "null", "n/a"):
        return ""
    return v


def get_session(session_id: str) -> dict:
    with _session_lock():
        sess = _SESSIONS.get(session_id)
        if sess is None:
            sess = {"turns": [], "summary": "", "facts": {"state": "", "topic": ""}}
            _SESSIONS[session_id] = sess
        return sess


_GREETING_RE = None


def _greeting_re():
    global _GREETING_RE
    if _GREETING_RE is None:
        import re

        _GREETING_RE = re.compile(
            r"^(hi+|hello+|hey+|he+y|good\s?(morning|afternoon|evening)|"
            r"how\s?far|wetin\s?dey|how\s?are\s?you|who\s?are\s?you|"
            r"good\s?day|good\s?day\s?o|abeg\s?greet)\s*[!.?]*$",
            re.IGNORECASE,
        )
    return _GREETING_RE


_EMERGENCY_RE = None


def _emergency_re():
    global _EMERGENCY_RE
    if _EMERGENCY_RE is None:
        import re

        _EMERGENCY_RE = re.compile(
            r"\b(beat|beaten|beating|tortur|kill|killed|killing|shoot|shot|"
            r"stab|blood|confess|force\s+me|threaten.*(kill|shoot|die)|"
            r"die|dying|danger|injur|wound|assault|rape|kidnap|"
            r"cell.*(beat|torture|confess)|police.*(beat|shot|kill))\b",
            re.IGNORECASE,
        )
    return _EMERGENCY_RE


def _converse_fast_path(text: str, sess: dict, retriever) -> dict | None:
    """Deterministic turns that need no model call (fast + robust).

    Returns an answer dict, or None when the turn needs the model.
    """
    from api.retrieval import detect_state, is_state_sensitive

    stripped = text.strip()
    if _greeting_re().match(stripped):
        spoken = ("Hello, my friend. I am Honest Orb, your legal companion. "
                  "Tell me what happened, and I will tell you what the law says.")
        return _record_turn(sess, "chat", spoken, {}, [], text, "")

    state = _clean_state(sess["facts"].get("state")) or detect_state(text) or ""
    if state:
        with _session_lock():
            sess["facts"]["state"] = state
    if _emergency_re().search(text):
        return None  # model path with emergency hint
    if is_state_sensitive(text) and not state:
        with _session_lock():
            sess["facts"]["topic"] = "tenancy"
        spoken = ("I can help with that. Which state is the house in — "
                  "for example Lagos, Abuja, or Kano? Tenancy law differs by state.")
        return _record_turn(sess, "clarify", spoken, {}, [], text, "tenancy")
    if (state and len(text.split()) <= 3 and detect_state(text)
            and (sess["facts"].get("topic") or "") == "tenancy"
            and not retriever.state_laws_available(state)):
        # The turn only names the state ("Lagos") answering our clarify.
        with _session_lock():
            sess["facts"]["topic"] = "tenancy"
        label = "Abuja (FCT)" if state == "fct" else state.capitalize()
        spoken = (f"Thanks — {label}. Honestly, I don't have the {label} tenancy law "
                  "loaded yet, so I can't tell you the exact notice your landlord must give. "
                  "Please contact the Legal Aid Council of Nigeria or a lawyer "
                  "before you move out.")
        details = {
            "right": "Tenancy notice periods depend on your state's law.",
            "what_to_say": "Ask your landlord to put the notice in writing.",
            "what_to_do": "Take any notice you received to the Legal Aid Council or a lawyer quickly.",
        }
        return _record_turn(sess, "not_covered", spoken, details, [], text, "tenancy")
    if is_state_sensitive(text) and state and not retriever.state_laws_available(state):
        label = "Abuja (FCT)" if state == "fct" else state.capitalize()
        spoken = (f"Honestly, I don't have the {label} tenancy law loaded yet, "
                  "so I can't tell you the exact notice your landlord must give. "
                  "Please contact the Legal Aid Council of Nigeria or a lawyer "
                  "before you move out.")
        details = {
            "right": "Tenancy notice periods depend on your state's law.",
            "what_to_say": "Ask your landlord to put the notice in writing.",
            "what_to_do": "Take any notice you received to the Legal Aid Council or a lawyer quickly.",
        }
        with _session_lock():
            sess["facts"]["topic"] = "tenancy"
        return _record_turn(sess, "not_covered", spoken, details, [], text, "tenancy")
    return None


def _record_turn(sess: dict, mode: str, spoken: str, details: dict,
                 citations: list[dict], user_text: str, topic: str) -> dict:
    with _session_lock():
        sess["turns"].append({"user": user_text, "mode": mode, "spoken": spoken})
        sess["turns"] = sess["turns"][-6:]
        if topic:
            sess["facts"]["topic"] = topic
        facts = dict(sess["facts"])
    return {
        "mode": mode,
        "language": "en",
        "spoken": spoken,
        "details": {
            "right": details.get("right", ""),
            "what_to_say": details.get("what_to_say", ""),
            "what_to_do": details.get("what_to_do", ""),
        },
        "citations": citations,
        "facts": facts,
        "summary": sess.get("summary", ""),
    }


def converse(session_id: str, text: str) -> dict:
    """One conversational turn with server-side memory.

    Returns {mode, language, spoken, details, citations, facts, summary}.
    Citation ids are validated against sections.json; unknown ids dropped.
    Exactly one Muse Spark call per turn (fast paths use none).
    """
    from api.retrieval import Retriever, detect_state

    sess = get_session(session_id or "default")
    retriever = Retriever(SECTIONS)
    fast = _converse_fast_path(text, sess, retriever)
    if fast is not None:
        return fast

    state = _clean_state(sess["facts"].get("state")) or detect_state(text) or ""
    context_turns = sess["turns"][-4:]
    context = " ".join(t.get("user", "") for t in context_turns)
    # The current turn dominates: repeat it so follow-up context
    # ("Lagos", "can he do that?") cannot drown out the actual question.
    query = f"{text} {text} {text} {context}"
    if _emergency_re().search(text):
        query += " torture beating beaten confess confession injury dignity humane treatment"
    hits = retriever.search(query, state=state or None, top_k=10)

    lines = [
        f"Session facts: state={state or 'unknown'}, "
        f"topic={sess['facts'].get('topic') or 'none'}.",
    ]
    if sess.get("summary"):
        lines.append(f"Conversation so far: {sess['summary']}")
    for t in context_turns:
        lines.append(f"User: {t['user']} | Orb ({t['mode']}): {t['spoken']}")
    lines.append(f"New user turn: {text}")
    if _emergency_re().search(text):
        lines.append("Triage hint: possible violence/danger — consider emergency mode.")
    lines.append("")
    lines.append("RETRIEVED SECTIONS (only cite these; if none clearly cover the turn, "
                 "use clarify, not_covered, chat or redirect):")
    if hits:
        for s in hits:
            lines.append(
                f"[{s['id']}] {s['source']}, s.{s['section']} - {s.get('title','')}\n"
                f"{s.get('text','')[:1500]}"
            )
    else:
        lines.append("(no sections retrieved)")
    lines.append("")
    lines.append("Reply in JSON per the schema. spoken: 1-3 short sentences, Nigerian English.")

    client = _get_client()
    # One repair retry: a truncated over-long reply fails schema parsing.
    result: dict | None = None
    last_error: Exception | None = None
    for _ in range(2):
        try:
            response = _call_with_retry(
                client,
                model=MODEL,
                reasoning={"effort": "minimal"},
                max_output_tokens=MAX_OUTPUT_TURNS_TOKENS,
                input=[
                    {"role": "system", "content": CONVERSE_PREFIX},
                    {"role": "user", "content": "\n".join(lines)},
                ],
                text_format=_converse_schema(),
            )
            parsed = response.output_parsed
            if parsed is None:
                raise RuntimeError("Model returned no parsed output")
            result = parsed.model_dump()
            break
        except Exception as exc:  # noqa: BLE001 - retry once, then surface
            last_error = exc
    if result is None:
        assert last_error is not None
        raise last_error
    valid = [c for c in result.get("citations", []) if c.get("id") in VALID_IDS]
    if result.get("mode") in ("legal", "emergency") and not valid:
        # Grounded modes must cite: fall back to an honest limitation.
        result["mode"] = "not_covered"
        result["spoken"] = ("Honestly, I can't verify that in the laws I have loaded. "
                            "Please contact the Legal Aid Council of Nigeria or a lawyer.")
        result["details"] = {"right": "", "what_to_say": "", "what_to_do": ""}
    result["citations"] = valid
    result["language"] = "en"

    detected = _clean_state(
        detect_state(text) or result.get("facts", {}).get("state", "") or state
    )
    topic = result.get("facts", {}).get("topic", "") or sess["facts"].get("topic", "")
    with _session_lock():
        if detected:
            sess["facts"]["state"] = detected
        if topic:
            sess["facts"]["topic"] = topic
        if result.get("summary"):
            sess["summary"] = result["summary"][:500]
        sess["turns"].append({"user": text, "mode": result["mode"],
                              "spoken": result.get("spoken", "")})
        sess["turns"] = sess["turns"][-6:]
        result["facts"] = dict(sess["facts"])
        result["summary"] = sess.get("summary", "")
    return result


def _call_with_retry(client, **kwargs):
    from openai import APIConnectionError, APIStatusError, APITimeoutError

    last_error: Exception | None = None
    for attempt in range(3):  # initial attempt + up to 2 retries
        try:
            return client.responses.parse(**kwargs)
        except (APIConnectionError, APITimeoutError) as exc:
            last_error = exc
        except APIStatusError as exc:
            if exc.status_code is not None and exc.status_code >= 500:
                last_error = exc
            else:
                raise
        time.sleep(1.0 * (attempt + 1))
    assert last_error is not None
    raise last_error


def answer(question: str) -> dict:
    """Answer a question grounded in data/sections.json.

    Returns a dict with keys: language, status, right, say_this, do_this,
    citations ([{id, label}]), spoken. Citation ids are validated against
    sections.json; unknown ids are dropped.
    """
    client = _get_client()
    result: dict | None = None
    for _ in range(2):
        # One repair retry if the model returns no citations: the spec
        # requires at least one existing section id on every answer.
        response = _call_with_retry(
            client,
            model=MODEL,
            reasoning={"effort": "minimal"},
            max_output_tokens=MAX_OUTPUT_TOKENS,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            text_format=_answer_schema(),
        )
        parsed = response.output_parsed
        if parsed is None:
            raise RuntimeError("Model returned no parsed output")
        result = parsed.model_dump()
        result["citations"] = [
            c
            for c in result.get("citations", [])
            if c.get("id") in VALID_IDS
        ]
        if result.get("status") in ("answered", "emergency") and not result[
            "citations"
        ]:
            continue
        return result
    assert result is not None
    return result
