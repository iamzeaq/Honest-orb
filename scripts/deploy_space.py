"""Deploy Honest Orb to a Hugging Face Space (Docker, free CPU).

One command from the repo root::

    python scripts/deploy_space.py [--dry-run] [--space iamzeaq/honest-orb]

Reads MODEL_API_KEY and HF_TOKEN from ``.env`` (never prints them).
Does everything through the ``huggingface_hub`` Python API:

1. create (or reuse) the Space: repo_type="space", public, default
   free-CPU hardware; --sdk docker (default) or gradio. The gradio SDK
   wraps the same FastAPI app: app.py runs uvicorn api.server:app on
   :7860, requirements.txt pins CPU torch, packages.txt adds ffmpeg;
2. set secrets (MODEL_API_KEY, HF_TOKEN) and variables (TTS_MODEL_ID,
   VOICE_STYLE, TTS_NOISE_SCALE, TTS_NOISE_SCALE_DURATION,
   TTS_SPEAKING_RATE) for clarity variant C;
3. stage the SDK payload plus api/, web/, data/sections.json and the
   Space README, then upload_folder;
4. poll the runtime stage until RUNNING (build errors are fetched from
   the Space logs);
5. test GET /health and POST /api/ask_text, printing the public URL,
   the voice variant, and client-measured timings.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Clarity bake-off winner (api/clear_voice_eval.py, 3 sentences via api/stt):
# C avg 93.9% > A 84.8% > D 81.8% > B 75.8%.
VARIANT = "C"
SPACE_VARS = {
    "TTS_MODEL_ID": "facebook/mms-tts-eng",
    "VOICE_STYLE": "orb",
    "TTS_NOISE_SCALE": "0.4",
    "TTS_NOISE_SCALE_DURATION": "0.5",
    "TTS_SPEAKING_RATE": "0.9",
}
SECRET_NAMES = ("MODEL_API_KEY", "HF_TOKEN")
TEST_QUESTION = "Can police search my phone when they stop me on the road?"


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines (quotes stripped). Never prints values."""
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            env[key] = value
    return env


GRADIO_APP_PY = '''"""Honest Orb on a Gradio-SDK Space: serve the FastAPI app on :7860."""
import uvicorn

from api.server import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
'''

GRADIO_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.6.0+cpu
fastapi
uvicorn
python-multipart
transformers
numpy
imageio-ffmpeg
pymupdf
openai
requests
"""

GRADIO_PACKAGES = "ffmpeg\n"

GRADIO_README = """---
title: Honest Orb
emoji: 🔮
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
python_version: "3.11"
pinned: false
---

# Honest Orb

Honest Orb is a voice-first AI that tells everyday Nigerians their legal rights out loud, in English or Pidgin. Tap the orb, ask in your own words, and hear a clear answer grounded in the Nigerian Constitution.

This Space runs the FastAPI app (`api/server.py`) on port 7860; no Gradio UI is used. The TTS voice model downloads at container startup — no weights are baked into the image.

## Secrets (Settings → Variables and secrets)

| Name | Value |
| --- | --- |
| `MODEL_API_KEY` | Meta API key that serves the brain (`muse-spark-1.2-contributor`) |
| `HF_TOKEN` | Hugging Face token with access to the private TTS model (fine-grained, read-only is enough) |
| `TTS_MODEL_ID` | Private voice model id, e.g. `iamzeaq/honest-orb-mms-pcm` |

Information, not legal advice · Emergency 112.
"""


def stage_app(sdk: str = "docker") -> Path:
    """Copy exactly the Space payload into a temp dir; return its path."""
    stage = Path(tempfile.mkdtemp(prefix="honest_orb_space_"))
    if sdk == "docker":
        shutil.copy2(ROOT / "Dockerfile", stage / "Dockerfile")
        shutil.copy2(ROOT / ".dockerignore", stage / ".dockerignore")
        shutil.copy2(ROOT / "deploy" / "space_README.md", stage / "README.md")
        header = (stage / "README.md").read_text(encoding="utf-8")
        assert "sdk: docker" in header and "app_port: 7860" in header, \
            "Space README lost its Docker YAML header"
    elif sdk == "gradio":
        (stage / "app.py").write_text(GRADIO_APP_PY, encoding="utf-8")
        (stage / "requirements.txt").write_text(GRADIO_REQUIREMENTS,
                                                encoding="utf-8")
        (stage / "packages.txt").write_text(GRADIO_PACKAGES, encoding="utf-8")
        (stage / "README.md").write_text(GRADIO_README, encoding="utf-8")
        header = GRADIO_README
        assert "sdk: gradio" in header and "app_file: app.py" in header, \
            "Space README lost its Gradio YAML header"
    else:
        raise ValueError(f"unknown SDK: {sdk}")
    shutil.copytree(ROOT / "api", stage / "api",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(ROOT / "web", stage / "web",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (stage / "data").mkdir()
    shutil.copy2(ROOT / "data" / "sections.json",
                 stage / "data" / "sections.json")

    banned = (".env", ".venv", "raw", "voice", "slr70", "audio",
              "audio_out", "evals", "train")
    for rel in [p.relative_to(stage).as_posix()
                for p in list(stage.rglob("*"))]:
        top = rel.split("/")[0]
        assert not rel.endswith(".zip") and top not in banned, \
            f"staging leak: {rel}"
    return stage


def http_json(method: str, url: str, payload: bytes | None = None,
              timeout_s: float = 300) -> tuple[int, str, float]:
    req = urllib.request.Request(url, data=payload, method=method,
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.status, resp.read().decode("utf-8"), \
                time.perf_counter() - started
    except Exception as exc:  # urllib raises HTTPError (has .read) on 4xx/5xx
        body = ""
        try:
            body = exc.read().decode("utf-8")  # type: ignore[union-attr]
        except Exception:
            pass
        return getattr(exc, "code", -1) or -1, body or str(exc), \
            time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--space", default="iamzeaq/honest-orb")
    args.add_argument("--sdk", default="docker", choices=("docker", "gradio"))
    args.add_argument("--dry-run", action="store_true",
                      help="stage + validate only; no HF calls")
    args.add_argument("--poll-timeout-s", type=float, default=1800)
    args.add_argument("--poll-interval-s", type=float, default=20)
    opts = args.parse_args(argv)

    env = load_dotenv(ROOT / ".env")
    missing = [k for k in SECRET_NAMES if not env.get(k)]
    if missing:
        if opts.dry_run:
            print(f"dry-run note: {', '.join(missing)} missing from .env; "
                  f"a live run would stop here.")
        else:
            print(f"ERROR: {', '.join(missing)} missing from .env; "
                  f"add them (see .env.example) and re-run.", file=sys.stderr)
            return 1

    stage = stage_app(opts.sdk)
    staged = sorted(p.relative_to(stage).as_posix()
                    for p in stage.rglob("*") if p.is_file())
    print(f"staged {len(staged)} files, e.g. {', '.join(staged[:4])} ...")
    print(f"voice variant {VARIANT}: " +
          ", ".join(f"{k}={v}" for k, v in SPACE_VARS.items()))

    if opts.dry_run:
        print("dry-run: Space untouched.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=env["HF_TOKEN"])
    me = api.whoami()
    print(f"authenticated as {me.get('name')}")

    try:
        api.create_repo(repo_id=opts.space, repo_type="space",
                        space_sdk=opts.sdk, private=False, exist_ok=True)
    except Exception as exc:
        if "402" in str(exc) or "Payment Required" in str(exc):
            print(f"ERROR: Hugging Face returned 402 for SDK {opts.sdk!r}: "
                  "that Space kind needs a PRO subscription even on "
                  "cpu-basic. See https://huggingface.co/pro, then re-run.",
                  file=sys.stderr)
            return 4
        raise
    print(f"space ready: {opts.space} (public, {opts.sdk})")

    for name in SECRET_NAMES:
        api.add_space_secret(opts.space, name, env[name])
        print(f"secret set: {name}")
    for key, value in SPACE_VARS.items():
        api.add_space_variable(opts.space, key, value)
        print(f"variable set: {key}={value}")

    api.upload_folder(folder_path=str(stage), repo_id=opts.space,
                      repo_type="space")
    print(f"uploaded {len(staged)} files to {opts.space}")
    shutil.rmtree(stage, ignore_errors=True)

    deadline = time.time() + opts.poll_timeout_s
    last_stage = ""
    while time.time() < deadline:
        runtime = api.get_space_runtime(opts.space)
        if runtime.stage != last_stage:
            print(f"stage: {runtime.stage} (hardware: {runtime.hardware})")
            last_stage = runtime.stage
        if runtime.stage == "RUNNING":
            break
        if runtime.stage in ("BUILD_ERROR", "RUNTIME_ERROR", "STOPPED"):
            try:
                logs = api.get_space_logs(opts.space) or ""
            except Exception as exc:
                logs = f"<logs unavailable: {exc}>"
            print("BUILD FAILED. Log tail:\n" + "\n".join(
                logs.splitlines()[-40:]), file=sys.stderr)
            return 2
        time.sleep(opts.poll_interval_s)
    else:
        print("ERROR: timed out waiting for RUNNING", file=sys.stderr)
        return 2

    # repo_id "iamzeaq/honest-orb" -> host "iamzeaq-honest-orb.hf.space"
    base = "https://" + opts.space.replace("/", "-") + ".hf.space"

    code, body, dt = http_json("GET", base + "/health")
    print(f"GET /health -> {code} in {dt:.1f}s: {body[:120]}")
    if code != 200:
        return 3
    import json as _json
    code, body, dt = http_json(
        "POST", base + "/api/ask_text",
        _json.dumps({"text": TEST_QUESTION}).encode(), timeout_s=300)
    print(f"POST /api/ask_text -> {code} in {dt:.1f}s")
    try:
        ans = _json.loads(body).get("answer", {})
        print(f"  status={ans.get('status')} "
              f"spoken={str(ans.get('spoken'))[:100]}")
    except Exception:
        print(f"  body={body[:200]}")
    print(f"public URL: {base} (variant {VARIANT})")
    return 0 if code == 200 else 3


if __name__ == "__main__":
    sys.exit(main())
