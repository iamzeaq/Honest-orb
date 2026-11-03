---
title: Honest Orb
emoji: 🔮
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Honest Orb

Honest Orb is a voice-first AI that tells everyday Nigerians their legal rights out loud, in English or Pidgin. Tap the orb, ask in your own words, and hear a clear answer grounded in the Nigerian Constitution.

This Space runs the Docker image from the repo root (`Dockerfile`, port 7860). The TTS voice model downloads at container startup — no weights are baked into the image.

## Secrets (Settings → Variables and secrets)

| Name | Value |
| --- | --- |
| `MODEL_API_KEY` | Meta API key that serves the brain (`muse-spark-1.2-contributor`) |
| `HF_TOKEN` | Hugging Face token with access to the private TTS model (fine-grained, read-only is enough) |
| `TTS_MODEL_ID` | Private voice model id, e.g. `iamzeaq/honest-orb-mms-pcm` |

Information, not legal advice · Emergency 112.
