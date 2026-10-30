# Honest Orb Voice Recording Guide

You are recording the voice of Honest Orb: a calm, warm Lagos big-brother
voice. Read like you are explaining things to a friend who is nervous at a
police checkpoint, not like you are reading news.

## Setup (same every time)

- Record in a quiet room with doors and windows closed. Same room, same
  chair, same time of day for all 9 batches.
- Hold the phone about 20cm from your mouth (a stretched hand span).
  Never move it closer or further mid-batch.
- Put the phone on Do Not Disturb. One test clip first: play it back and
  check there is no fan, traffic, or echo.

## How to read

- Read naturally, like talking to a friend. Keep the same warm pace in
  Pidgin and in English.
- Read only the sentence, never the line number (001, 002, ...).
- Keep Pidgin in Pidgin and English in English. Do not translate lines.
- Pause 2 full seconds of silence between lines. This silence is how the
  computer cuts your recording into clips, so do not skip it.
- If you cough, stumble, or a bike passes, pause 2 seconds and re-read
  that one line. Do not stop the recording.

## Batches

- Record in batches of 20 lines: lines 001-020 go in `batch_01.m4a`,
  lines 021-040 in `batch_02.m4a`, up to lines 161-180 in `batch_09.m4a`.
- Use any voice-recorder app that saves `.m4a`. One file per batch.
- After each batch, count your lines: the file must hold exactly 20
  sentences with silence between them.

## Checklist before you send a batch

- [ ] File is named `batch_0N.m4a` (01 to 09).
- [ ] It holds the right 20 lines, in script order.
- [ ] 2-second silence between every line.
- [ ] No background noise, no clipping (no harsh crackling on loud words).

Put finished files in `data/voice/raw/`, then run:

```sh
source .venv/bin/activate
python api/voice_prep.py
```

The script converts, splits, and pairs every clip, and tells you if any
batch has more or fewer than 20 clips.
