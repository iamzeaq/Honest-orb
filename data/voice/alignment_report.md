# Voice alignment report

Built with `python api/voice_prep.py`: batches split on ~0.5 s silence,
pieces transcribed with Muse Voice Transcribe (`api/stt.py`), then aligned
to the full 180-line script by normalized word overlap (keep threshold
>= 0.6) with one forward-moving pointer across all batches. Kept clips are
trimmed, padded with ~0.15 s silence, 16 kHz mono WAV. `metadata.csv` uses
the SCRIPT text, not the transcript.

## Per-batch coverage

| batch | lines covered | clips kept |
|---|---|---|
| batch_01.m4a | 001–020 | 20 |
| batch_02.m4a | 021–040 | 19 |
| batch_03.m4a | 041–057 | 16 |
| batch_04.m4a | 058–080 | 23 |
| batch_05.m4a | 081–100 | 17 |
| batch_06.m4a | 093–120 | 13 |
| batch_07.m4a | 121–141 | 12 |
| batch_08.m4a | 142–169 | 22 |
| batch_09.m4a | 173–180 | 6 |

Notes:

- batch_06 re-reads lines 093–100 (spot-checked transcripts score 0.77–1.0
  against those lines); the better of the two readings was kept.
- Lines 170–172 fall in the gap between batch_08 (ends 169) and batch_09
  (starts 173): never read.

## Totals

- Total clips kept: 144 / 180 (36 low-scoring pieces dropped and logged)
- Total speech: 4.6 min across 9 batches

## Never matched (36)

022, 045, 085, 086, 092, 101, 102, 103, 104, 106, 107, 110, 111, 113, 114,
115, 126, 130, 131, 132, 133, 134, 135, 137, 138, 162, 163, 165, 166, 167,
168, 170, 171, 172, 178, 179

These lines were skipped, mumbled, or read too noisily to reach the 0.6
threshold. Re-record them; nothing was fabricated for them.
