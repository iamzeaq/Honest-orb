# Deploy Honest Orb to Modal (Route 2)

HF Spaces needs PRO for Gradio/Docker hosting, so this route serves the
same FastAPI app (`api/server.py`) from Modal instead. The operator runs
every command below from the repo root on this machine — the assistant
does not run `modal setup` or `modal deploy` itself.

## 0. Install the Modal client

```bash
pip install modal
```

## 1. Log in to Modal (interactive — run it yourself)

```bash
python -m modal setup
```

This opens a browser (or prints a token page). Log in and paste the token
back. Verify with:

```bash
modal profile list
```

## 2. Create the `honest-orb` secret

One Modal secret holds the two credentials plus the five voice settings
(clarity variant C — the bake-off winner: C 93.9% > A 84.8% > D 81.8% >
B 75.8%). Replace the `<...>` placeholders with the values from `.env`
— never commit them anywhere:

```bash
modal secret create honest-orb \
  MODEL_API_KEY=<paste-MODEL_API_KEY-from-.env> \
  HF_TOKEN=<paste-HF_TOKEN-from-.env> \
  TTS_MODEL_ID=facebook/mms-tts-eng \
  VOICE_STYLE=orb \
  TTS_NOISE_SCALE=0.4 \
  TTS_NOISE_SCALE_DURATION=0.5 \
  TTS_SPEAKING_RATE=0.9
```

Confirm with `modal secret list` (it shows names only, never values).

## 3. Deploy (persistent endpoint)

```bash
modal deploy modal_app.py
```

The output prints the public URL, shaped like
`https://<workspace>--honest-orb-fastapi-app.modal.run`.

## 4. Test it

First boot downloads the TTS voice model (~1 GB into the
`honest-orb-models` volume) and warms it up, so allow several minutes.
After that:

```bash
URL=https://<workspace>--honest-orb-fastapi-app.modal.run
curl $URL/health
curl -X POST $URL/api/ask_text \
  -H 'Content-Type: application/json' \
  -d '{"text":"Can police search my phone when they stop me on the road?"}'
```

`/health` answers `{"status":"ok"}`; the ask endpoint returns the brain
answer plus TTS audio URLs (same contract as the local server).

## Notes

- Cold starts after the first are fast: the model cache lives on the
  `honest-orb-models` volume, not in the image; no weights are baked in.
- No container is kept warm by default (`keep_warm` is commented out in
  `modal_app.py`). For demo hours, uncomment `keep_warm=1` and redeploy —
  it costs while the container idles, so turn it back off afterwards.
- CPU with ~4 GB RAM, `ffmpeg` in the image, 1 h function timeout to
  cover the first-boot download.
