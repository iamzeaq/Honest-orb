# Clarity + speed results

Date: 2026-10-31. Brain (`api/brain.py`) and `evals/` unchanged; all gains
are TTS-side. Brain answers below are the recorded eval outputs for
q02/q05/q16 (live brain API unreachable from this sandbox; replayed through
the real HTTP endpoints + real TTS model, so brain latency cancels out).

## Time-to-first-audio and total (seconds, client-measured)

Before = original code (`/api/ask_text`, one clip). After = sentence queue
(`audio_urls`, `timings.tts_first`) + `/api/ask_text_stream` NDJSON.

| q | question | before first / total | after JSON first / total | after stream: 1st audio ready |
|---|---|---|---|---|
| q02 | arrested without reason ("24 hours") | 52.44 / 52.45 | 20.35 / 20.35 (tts_first 13.5) | tts_s 3.1 |
| q05 | beaten at checkpoint ("112") | 7.52 / 7.52 | 7.04 / 7.05 (tts_first 1.9) | tts_s 1.8 |
| q16 | detention time ("40 kilometres") | 13.63 / 13.63 | 10.11 / 10.12 (tts_first 1.9) | tts_s 1.6 |

Notes: q02-before includes first-ever-synthesis warmup in that process.
`tts_first`/`tts_s` are server-measured (immune to client buffering).
Line-buffered stream probe (q05, fresh process): meta +0.05s, sentence
audios +9.3s/+13.0s/+14.6s — playback starts at the first line while later
sentences still render. Recorded brain latencies: 2.32s / 2.84s / 1.67s.

Samples: `data/audio_out/clarity_before_{1_q02,2_q05,3_q16}.wav` (original
synthesis) and `clarity_after_*` (full clip + per-sentence `_s0/_s1/_s2`).

## What changed

- `api/tts.py`: new `normalize_for_speech` (numbers→words, `112`→`one one
  two`, `s.35`/`ss.49–51`→`section(s)…`, symbols stripped, abbreviations
  expanded; missing vocab `q`→`k`, no digit ever reaches the tokenizer);
  new `split_sentences`, `speak_sentence`, `speak_sentences`, plus
  `concat_wav_files`. `speak()` normalizes then concatenates.
- `api/server.py`: `/api/ask_text` returns `audio_urls` + `sentences` +
  `timings.tts_first` (`audio_url` still the full clip); new
  `/api/ask_text_stream` NDJSON endpoint (meta, then one audio line per
  sentence); `/api/ask` returns the queue too.
- `api/pipeline.py`: `run()` synthesizes once, returns `audio_out` (full),
  `sentence_outs`, `sentences`, `timings.tts_first`.
- `web/app.js`: typed questions use the stream with JSON fallback; answers
  play sentence-by-sentence in order; orb caption and pulse follow the
  currently playing sentence.
- `api/test_tts_clarity.py`: new offline tests (normalize, split, queue
  shape). Full suite: 27 tests OK.
