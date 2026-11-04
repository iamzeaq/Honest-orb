# Honest Orb assessment: from police-only bot to legal companion

Date: 2026-11-04. Codebase as found: 35 sections in [data/sections.json](/workspaces/Honest-orb/data/sections.json)
(14 Constitution 1999 Ch. IV + 21 Police Act 2020), single-turn [api/brain.py](/workspaces/Honest-orb/api/brain.py),
[api/server.py](/workspaces/Honest-orb/api/server.py) + [web/](/workspaces/Honest-orb/web) voice UI, 18 evals in
[evals/questions.json](/workspaces/Honest-orb/evals/questions.json).

## 1. What currently restricts it to police topics

- **System prompt scope.** `SYSTEM_PROMPT` says "rights during police encounters" and forces
  `off_topic` for anything else (e.g. q14 jollof-rice test). Small talk ("Hello") has no
  friendly path — it would be classified `off_topic` or `not_sure`.
- **Data scope.** Only Constitution Ch. IV (ss.33–46) + Police Act 2020 arrest/search/bail/detention
  sections are extracted. No tenancy, labour, FOI, land, consumer, anti-corruption, legal-aid
  or anti-torture text exists, so any non-police legal question can only return `not_sure`.
- **Stateless brain.** `answer(question)` takes one string, no session/memory, so follow-ups
  ("Can he actually do that?"), topic switches and state-jurisdiction tracking are impossible.
- **Evals lock it in.** All 18 evals are police-encounter Q&A; q13 (roadblock bribe complaint)
  expects `not_sure` and q14 (jollof) expects `off_topic` — both correct only for a police bot.

## 2. What changes (per spec step)

1. **Law data** — extend [api/build_sections.py](/workspaces/Honest-orb/api/build_sections.py) with a generic
   extractor + per-law config (id prefix, source, jurisdiction, domain, section list, keywords,
   OCR fixes). **BLOCKER (verified 2026-11-04, `ls data/raw/`): only `Police-Act-2020.pdf` and
   `constitution_1999.pdf` are present.** All ten new PDFs (acja_2015, lagos_tenancy_law_2011,
   foi_act_2011, land_use_act, labour_act, fccpa_2018, corrupt_practices_act_2000,
   legal_aid_act_2011, anti_torture_act_2017, fct_recovery_of_premises_act) are absent, so no
   new sections can be extracted tonight. The builder is written to ingest them the moment they
   land in `data/raw/` (missing file = logged skip, existing 35 sections kept byte-identical).
2. **Retrieval** — new `api/retrieval.py`: pure-Python BM25 over sections.json (no embeddings,
   no non-Meta models), returns ~8–12 sections; federal laws always apply, state laws only on
   state match; unknown state + state-sensitive query → ask.
3. **Conversation brain** — new `converse()` in brain.py: server-side session memory
   (session_id, recent turns, running summary, facts incl. state/topic), one Muse Spark call per
   turn returning JSON `{mode, spoken, details, citations, facts}` with modes
   chat/clarify/legal/emergency/not_covered/redirect. Legacy `answer()` kept as a thin wrapper
   mapping legal→answered / not_covered→not_sure / redirect→off_topic so the existing 18 evals
   still pass unchanged. Carry-overs kept: exact s.35(4)–(5) 1-day/40km/2-day wording, no
   overclaiming, safety-first script, citation ids validated against sections.json.
4. **Voice-first UI** — keep the orb design; voice stays default output; no auto card (caption
   line only + "Show details"); continuous listen-after-speak; tap orb to end; "Type instead";
   session_id in localStorage; "your legal companion" subtitle.
5. **Speed** — streaming endpoints already exist (`/api/ask_text_stream`, NDJSON per-sentence
   audio); keep TTS model loaded once, keep_warm=1 cost reported at deploy.
6. **Voice** — unchanged: `facebook/mms-tts-eng`, `VOICE_STYLE=orb`, 0.4/0.5/0.9 stay as deployed.
7. **Tests** — new `evals/conversation_tests.json` + runner covering the spec's script incl.
   multi-turn memory (Lagos tenancy), new-domain probes (honestly `not_covered` until PDFs land).
8. **Deploy** — `modal deploy`, live smoke test, commit (never secrets).

## 3. What is reused untouched

- TTS/STT modules and voice config; orb canvas design; FastAPI app shape and `/audio/`,
  `/api/section/` endpoints; Constitution + Police Act extraction code and the 35 sections;
  the 18 existing evals and their runner contract; per-sentence streaming protocol.
