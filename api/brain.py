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
