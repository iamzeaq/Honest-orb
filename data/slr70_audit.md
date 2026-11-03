# SLR70 Nigerian-English (male) dataset feasibility audit

Date: 2026-11-02. Scope: MALE archive only (`en_ng_male.zip`, ~454 MB).
No training was run; the `iamzeaq/honest-orb-mms-pcm` checkpoint, training
config and the A–E clarity experiment were not touched.
Raw data lives in `data/slr70/` (gitignored, never commit it).

## Verdict: YES — worth fine-tuning tonight

Speaker `07508` gives 27 clean minutes from 300 clips — the largest,
most consistent single male voice in the archive — and the whole run fits
in ~30 min on a Colab T4.

## Source

- Page: https://www.openslr.org/70/
- Files: `line_index_male.tsv` (1314 rows: `<ngm_SPKR_UTT>\t<transcript>`),
  `en_ng_male.zip` (1316 files: 1314 WAV + LICENSE + duplicate),
  `LICENSE` (CC BY-SA 4.0, full text in `data/slr70/LICENSE`).
- License: Creative Commons Attribution-ShareAlike 4.0 International.

## Corpus inventory (all 1314 clips)

- TSV ↔ WAV match is exact: 1314 entries, 1314 files, 0 missing either way.
- Format is uniform: every file 48 000 Hz, mono, 16-bit PCM.
- Corrupt: 0. Silent (RMS < −60 dBFS): 0. Clipped: 0.
- Total: 138.9 min. Content is read speech (assistant prompts, trivia,
  news-style sentences) — generic domain, not Pidgin/legal.

## Per-speaker breakdown (speaker ID = middle field of file name)

| speaker | clips | total min | avg clip s |
|---|---|---|---|
| 07508 | 300 | 27.1 | 5.42 |
| 09334 | 126 | 19.8 | 9.41 |
| 14310 | 133 | 13.2 | 5.97 |
| 02436 | 87 | 12.1 | 8.38 |
| 07049 | 122 | 10.7 | 5.24 |
| 09697 | 102 | 10.5 | 6.16 |
| 03397 | 83 | 9.5 | 6.90 |
| 03349 | 78 | 9.3 | 7.16 |
| 02484 | 95 | 9.2 | 5.84 |
| 12484 | 108 | 8.3 | 4.60 |
| 08784 | 50 | 5.9 | 7.09 |
| 01523 | 30 | 3.3 | 6.56 |

## Top-3 quality check (10 random clips each, seed 7, via `api/stt.py`)

Word overlap = 1 − WER vs the TSV transcript. SNR = loudest-vs-quietest
decile energy ratio (saturates high on studio recordings with digital
silence between phrases — read as "studio-clean", not lab SNR). Loudness
via `api.tts.measure_lufs`.

| speaker | mean overlap | min | SNR | LUFS mean ± std |
|---|---|---|---|---|
| 07508 | 0.90 | 0.71 | ~75 dB | −24.2 ± 2.4 |
| 09334 | 0.89 | 0.69 | ~68 dB | −22.4 ± 4.5 |
| 14310 | 0.86 | 0.50 | ~72 dB | −19.1 ± 0.7 |

Error inspection: most misses are STT artefacts (duplicated words like
"cut cut", Nigerian proper names), not transcript errors — the TSV
transcripts are trustworthy. 09334 has loud outliers (−13.7 and −31.2
LUFS clips: "merry Christmas day" vs a quiet take). 14310 is the most
level-consistent but has the lowest overlap (a 0.50 outlier) and only
13.2 min. **07508 wins: most minutes, best mean overlap, no loudness
outliers.**

## Best single male speaker: 07508

- 300 usable recordings, 27.1 min total, avg 5.4 s (range 2.0–11.5 s —
  ideal TTS length, no clip exceeds 15 s so nothing needs splitting).
- 2–29 words per clip (mean 11.9). Transcripts contain digits (158 across
  corpus), the letter `q` (73×), and punctuation including `é`/`ẹ`/en-dash
  (Yoruba words) — all handled lossily-but-safely by
  `normalize_for_speech` (digits→words, `q`→`k`, the rest stripped; note
  `ẹ`/`é` collapse, acceptable for an English-voice fine-tune).

## Sample-rate compatibility

Source is 48 kHz mono 16-bit; our pipeline (and MMS-TTS) wants 16 kHz
mono 16-bit. Straight `ffmpeg -ar 16000 -ac 1` downsample — no
compatibility risk.

## Preprocessing for our MMS-TTS fine-tuning pipeline

1. Select `ngm_07508_*.wav` (300 files). No silence-splitting needed —
   the archive is already one utterance per file (unlike our batch
   recordings in `api/voice_prep.py`).
2. Resample to 16 kHz mono 16-bit WAV with ffmpeg.
3. Loudness-normalize clips to a consistent level (match the prior
   disguised-dataset recipe).
4. Run every TSV transcript through `api.tts.normalize_for_speech` so no
   digits/`q`/unsupported symbols reach the tokenizer.
5. Train/test split ~90/10 with a fixed seed: 270 train / 30 test.
6. Emit `wavs/*.wav` + `metadata.csv` with `file|text` rows using the
   normalized text (same layout as `data/voice/dataset/`).
7. Keep the existing recipe otherwise: batch 16, 80 epochs, fp16,
   eval/checkpoint every 64 steps (8 epochs), keep last 3, early-stop on
   climbing eval loss.

## Estimated Colab T4 training time

270 train clips ÷ batch 16 = 17 steps/epoch × 80 epochs = 1360 steps.
At our measured ~1.3 steps/s ≈ 17–18 min pure training, plus ~10 min
setup and cheap evals on 30 clips → **~30 min total**, comfortably inside
a free-tier evening session. (The notebook's older 20–45 min note assumed
2–4 s/step; the T4 measurement since came in at ~1.3 steps/s.)

## Dataset build (2026-11-03, no training run)

- Selected only speaker `07508` (300 clips). Each clip: ffmpeg to 16 kHz
  mono 16-bit, leading/trailing silence trimmed (energy VAD) with 0.15 s
  padding, loudness-normalized to −20 LUFS (peak −1 dBFS).
- Measured post-build loudness: mean −20.2 LUFS, std 0.6, range
  −23.5…−20.0 (peak-limited clips sit at −20.0).
- All 300 TSV transcripts through `api.tts.normalize_for_speech`
  (verified: no digits/`q`/unsupported chars remain). Deterministic
  shuffle, seed 42: first 270 rows = train, last 30 = test.
- `metadata.csv` (`file|text`) + `wavs/` (original `ngm_07508_*` names)
  zipped at top level as `voice_dataset.zip` in the repo root
  (gitignored, overwrites the previous zip).

## Licensing (CC BY-SA 4.0)

- Attribution (add to README): "Nigerian English speech data from
  OpenSLR SLR70 (https://www.openslr.org/70/), © its contributors,
  used under Creative Commons Attribution-ShareAlike 4.0 International
  (https://creativecommons.org/licenses/by-sa/4.0/)."
- ShareAlike implication: if we *distribute* the fine-tuned weights (or
  the dataset), that distribution must carry the same CC BY-SA 4.0 terms
  with attribution. Private use — training on Colab and keeping the
  checkpoint in our private HF repo, which is the current plan — does
  not trigger the share-alike distribution clause. Do not publish the
  weights publicly without re-licensing them CC BY-SA 4.0.
