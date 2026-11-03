# DEPLOY.md — put Honest Orb on Hugging Face Spaces (Docker, free CPU)

Do this once, in this order. Nothing here deploys from this checkout
automatically — you click each step yourself. Estimated time: ~20 minutes
plus the first-boot model download.

You need: a Hugging Face account, your Meta `MODEL_API_KEY`, a Hugging
Face token (`HF_TOKEN`) with access to the private TTS model, and the
model id (`TTS_MODEL_ID`, e.g. `iamzeaq/honest-orb-mms-pcm`).

## 1. Create the Space

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and click **Create new Space**.
2. **Space name:** `honest-orb` (under your own username, e.g. `iamzeaq/honest-orb`).
3. **Select the Space SDK:** choose **Docker**.
4. **Space hardware:** leave the default **CPU basic (free)**.
5. Choose **Blank** template.
6. Click **Create Space**.

## 2. Add the 3 secrets

1. On your new Space page, open the **Settings** tab.
2. Scroll to **Variables and secrets** and click **New secret** three times:

   | Name | Value to paste |
   | --- | --- |
   | `MODEL_API_KEY` | Your Meta API key (the same value as local `.env`) |
   | `HF_TOKEN` | Your Hugging Face token (fine-grained, read-only, with access to the private TTS model) |
   | `TTS_MODEL_ID` | `iamzeaq/honest-orb-mms-pcm` (or your own private model id) |

3. Keep them as **Secret** (not Public). The Space restarts when you save.

How they are used: the container reads all three from the environment at
runtime. The TTS model downloads at startup (not at image build), so the
token is only ever needed inside the running Space — never baked into git
or the image.

## 3. Push this repo to the Space

Run these on your own machine, from the repo root, replacing `YOURNAME`
with your Hugging Face username:

```bash
git remote add space https://huggingface.co/spaces/YOURNAME/honest-orb
git push space main
```

(If you already added that remote, just `git push space main`. Pushing a
later fix is the same command again.)

## 4. Finish the Space README

The Space needs the Docker header to know what to run:

1. On the Space page, open the **Files** tab and click **README.md** → **Edit**.
2. Delete its contents and paste in the whole of `deploy/space_README.md` from this repo.
3. Click **Commit**.

The Space now builds the `Dockerfile` at the repo root and serves it on
port `7860` (`sdk: docker`, `app_port: 7860`).

## 5. Check it works

1. Wait for the build badge on the Space page to turn green (**Running**).
   The first boot takes several minutes: it downloads the TTS model and
   warms it up, which is slow on free CPU.
2. Open the Space URL and tap the orb, or from your machine run:

```bash
curl https://YOURNAME-honest-orb.hf.space/health
```

which should answer `{"status":"ok"}`.

If the app loads but answers fail, re-check step 2 (a missing or
misspelled secret is the usual cause) and the Space **Logs** tab.
