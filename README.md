# Honest Orb

**Your legal companion. Talk to it like a friend — it answers like the law.**

Honest Orb is a voice-first AI companion that helps everyday Nigerians understand their rights. You tap the orb and speak in your own words — English or Pidgin — and it answers out loud in clear Nigerian English, grounded in real Nigerian law, with the exact sections one tap away.

- **Live demo:** https://iamzeaq--honest-orb-fastapi-app.us-east.modal.run
- **Demo video:** _add link_

> Honest Orb gives legal **information**, not legal advice. In an emergency, call **112**.

---

## The problem

Millions of Nigerians meet the law at its sharpest edges — a police checkpoint, a landlord's eviction threat, an unpaid salary — without knowing their rights. Many can't read legal text, and in a tense moment nobody has time to. Lawyers are expensive and far away.

## What Honest Orb does

- **Voice-first.** Tap the orb and talk. It talks back. It listens again automatically, so it feels like a phone call. Text is optional ("Show details").
- **Conversational.** Say hello, explain your situation in your own words, ask follow-ups ("can he actually do that?"), switch topics — it keeps the context.
- **Grounded in real law.** Legal answers come only from the verified laws loaded below, with citations you can open and read.
- **Honest about its limits.** If a law isn't loaded (for example, tenancy law for a state we don't cover yet), it says so and points you to the Legal Aid Council or a lawyer. It never invents law.
- **Safety first.** It always advises staying calm and polite, never resisting, and challenging abuses later through a lawyer or Legal Aid. If someone is in danger, it says "call 112" first.
- **State-aware.** Federal laws apply everywhere; for state laws (like tenancy) it asks which state you're in.

## How it works

```
 🎤 Voice ──► Muse Voice Transcribe ──► text
                                          │
                     conversation memory + BM25 retrieval over verified laws
                                          │
                                          ▼
                                   Muse Spark (brain)
                     mode: chat · clarify · legal · emergency · not_covered
                     spoken reply + citations (validated against the law base)
                                          │
                                          ▼
 🔊 Voice ◄── Meta MMS-TTS (sentence-by-sentence streaming) ◄── spoken reply
```

- **Retrieval** is plain keyword (BM25) search — no extra AI models.
- **Citations** are checked against the law base; any citation that doesn't exist is dropped.
- **Speech normalisation** turns "s.35", numbers and "112" into natural spoken words.

## Meta models used (and where they run)

Every model Honest Orb calls at runtime is a Meta model.

| Role | Meta model | Where it runs |
|---|---|---|
| Brain — conversation, legal reasoning, structured answers | **Muse Spark** | Meta Model API |
| Ears — speech to text (English + Pidgin input) | **Muse Voice Transcribe** (`muse-voice-transcribe-1.0`) | Meta Model API |
| Mouth — text to speech | **MMS-TTS** (`facebook/mms-tts-eng`, open weights) | Our server on Modal |

The app was built with **Muse Code**.

## R&D: giving Honest Orb a Nigerian voice

Off-the-shelf Meta MMS voices are clear but don't sound Nigerian, so we spent a large part of the hackathon fine-tuning Meta's open-weight **MMS-TTS** on Nigerian speech. Both fine-tuned models are Meta models (MMS, fine-tuned) and are kept private.

**Experiment 1 — the founder's own voice, Nigerian Pidgin (base `facebook/mms-tts-pcm`)**
- Wrote a 180-line recording script in Pidgin and English, using the words Honest Orb says most (police, warrant, bail, Legal Aid, "no wahala", "abeg").
- Recorded it on a phone in 9 batches. Silence-splitting over-segmented the audio (mid-sentence pauses, restarts), so we built an **alignment pipeline**: every audio piece is transcribed with **Muse Voice Transcribe**, matched to its script line by text similarity, merged or dropped, then **verified** (duration vs. word count, re-transcription, silence checks). Result: 137 verified clips, 8.8 minutes (`api/voice_prep.py`, `api/verify_voice.py`).
- Fine-tuned with the `finetune-hf-vits` recipe on a free Colab T4 (patched for current library versions).
- Finding: a −2-semitone pitch shift we added to disguise the speaker introduced metallic artefacts the model learned, and 9 minutes of phone audio wasn't enough for a natural voice.

**Experiment 2 — a Nigerian English speaker (base `facebook/mms-tts-eng`)**
- Audited the open **SLR70 Nigerian English** dataset: 1,314 recordings, 12 male speakers, 138.9 minutes, 48 kHz studio quality. Picked the single best speaker (27.1 minutes, consistent loudness, transcripts verified at ~90% with Muse Voice Transcribe) — see `data/slr70_audit.md`.
- Fine-tuned for 120 epochs on a free Kaggle T4.
- Finding: clearer and more Nigerian than Experiment 1, but not yet as intelligible as the stock voice.

**How we chose the live voice — measured, not guessed.** We generated the same sentences with each candidate voice, transcribed them back with Muse Voice Transcribe, and scored word accuracy:

| Variant | Voice | Settings | Clarity |
|---|---|---|---|
| A | `mms-tts-eng` | defaults + loudness | 84.8% |
| B | `mms-tts-eng` | steadier/slower (0.4 / 0.5 / 0.9) + EQ | 75.8% |
| **C (live)** | **`mms-tts-eng`** | **steadier/slower + EQ + "orb" resonance** | **93.9%** |
| D | `mms-tts-pcm` | steadier/slower + EQ | 81.8% |

Clarity matters most for a legal assistant, so variant C ships today. The voice is a single setting (`TTS_MODEL_ID`), so a fine-tuned Nigerian voice can replace it as soon as it beats C on clarity. Next: more clean studio-quality Nigerian speech, no pitch disguise, and longer training.

## Legal knowledge base

| Law | Jurisdiction | Covers |
|---|---|---|
| Constitution of the Federal Republic of Nigeria 1999 — Chapter IV | Federal | Fundamental rights |
| Nigeria Police Act 2020 | Federal | Arrest, search, bail, detention, use of force |
| Administration of Criminal Justice Act 2015 | Federal | Rights of suspects, arrest, bail, detention |
| Anti-Torture Act 2017 | Federal | Prohibition of torture, complaints, remedies |
| Legal Aid Act 2011 | Federal | Who qualifies for free legal help and how |
| Labour Act (Cap L1, LFN 2004) | Federal | Wages, contracts, notice, termination |
| Land Use Act (Cap L5, LFN 2004) | Federal | Land rights and occupancy |
| Freedom of Information Act 2011 | Federal | Access to public records |
| Federal Competition and Consumer Protection Act 2018 | Federal | Consumer rights |
| Corrupt Practices and Other Related Offences Act 2000 | Federal | Bribery and corruption by officials |
| Lagos State Tenancy Law 2011 | Lagos State | Landlord–tenant rights, notices, recovery of premises |

Sources are official gazettes and published Laws of the Federation editions. The Constitution text is the original 1999 edition; only Chapter IV (unaffected by later alterations relevant here) is used.

## Accuracy and testing

- **99 verified law sections** across 11 laws (`data/sections.json`). Scanned gazettes were OCR'd at build time and hand-checked; anything that couldn't be verified (for example, a garbled FCT Recovery of Premises Act scan) was left out rather than guessed.
- **Conversation tests: 43/43 pass.** These cover greetings, follow-up questions, memory across turns, topic switches, emergencies, off-topic questions and every legal domain.
- **Legal accuracy evals: 18/18 pass.** These check that answers cite the right sections and state time limits exactly as the law does (`evals/`).
- Average brain response time is 3.7 seconds.

## Run it yourself

Requirements: Python 3.11+, `ffmpeg`, a Meta Model API key.

```bash
git clone https://github.com/iamzeaq/Honest-orb.git
cd Honest-orb
python -m venv .venv && . .venv/bin/activate
pip install -r api/requirements.txt
cp .env.example .env        # add MODEL_API_KEY (and HF_TOKEN if using a private voice)
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000 and tap the orb.

- Rebuild the law base: `python api/build_sections.py`
- Run the evals: `python evals/run_evals.py`
- Deploy to Modal: see `scripts/deploy_modal.md` (`modal deploy modal_app.py`)

## Repository layout

```
api/        server, brain, retrieval, speech-to-text, text-to-speech, law extraction
web/        the orb interface (HTML/CSS/JS, PWA)
data/       law sources (data/raw) and the extracted law base (data/sections.json)
evals/      evaluation questions, runners and results
train/      MMS-TTS fine-tuning notebook (R&D)
docs/       design notes
```

## Roadmap

- A natural Nigerian voice from our MMS fine-tuning work, and Pidgin replies
- Yoruba, Hausa and Igbo
- Tenancy law for more states (Abuja, Kano, Rivers, Enugu and beyond)
- More laws: Electoral Act 2022, Violence Against Persons (Prohibition) Act, Child Rights Act, Nigeria Data Protection Act
- "Know Your Options" — short step-by-step action plans
- Other countries: same engine, a different constitution and language

## Credits and licences

- **Meta MMS-TTS** — CC-BY-NC 4.0 (Meta). Used non-commercially.
- **SLR70 Nigerian English speech dataset** — Copyright 2018, 2019 Google, Inc., CC BY-SA 4.0, https://www.openslr.org/70/. Used only to fine-tune a private R&D voice; the dataset and weights are not redistributed.
- **Nigerian laws** — public legislation of the Federal Republic of Nigeria and Lagos State.
- Code: MIT licence.

---

*Honest Orb provides general legal information, not legal advice. For your specific situation, contact the Legal Aid Council of Nigeria or a lawyer. In an emergency, call 112.*
