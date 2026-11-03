"""Honest Orb on Modal (Route 2): serve ``api.server:app`` as a web endpoint.

No Gradio/HF Space involved. Run from the repo root after
``python -m modal setup`` and creating the ``honest-orb`` secret
(see ``scripts/deploy_modal.md``)::

    modal deploy modal_app.py

What this builds:

- Debian slim (Python 3.11) + ``ffmpeg`` (apt) + CPU-only torch from the
  PyTorch CPU index + the app's ``api/requirements.txt``-equivalent deps.
- Only the serving payload is baked into the image (``api/``, ``web/``,
  ``data/sections.json``) — never ``.env`` or other secrets.
- The TTS voice model (~1 GB) downloads on first boot into a persisted
  Modal Volume (``honest-orb-models``), so later cold starts skip it.
- CPU with ~4 GB RAM, generous timeout for the first-boot download.
  No containers are kept warm by default (see ``keep_warm`` note below).
"""

import modal

REMOTE_APP_DIR = "/app"
MODEL_MOUNT = "/models"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch==2.6.0+cpu",
                 index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("fastapi", "uvicorn", "python-multipart", "transformers",
                 "numpy", "imageio-ffmpeg", "pymupdf", "openai", "requests")
    # Serving payload baked into the image (copy=True): only api/, web/
    # and data/sections.json — never .env or other secrets. Paths are
    # relative to the repo root (run modal deploy from there).
    .add_local_dir("api", remote_path=f"{REMOTE_APP_DIR}/api", copy=True)
    .add_local_dir("web", remote_path=f"{REMOTE_APP_DIR}/web", copy=True)
    .add_local_file("data/sections.json",
                    remote_path=f"{REMOTE_APP_DIR}/data/sections.json",
                    copy=True)
)

model_volume = modal.Volume.from_name("honest-orb-models",
                                      create_if_missing=True)

app = modal.App("honest-orb")


@app.function(
    image=image,
    volumes={MODEL_MOUNT: model_volume},
    secrets=[modal.Secret.from_name("honest-orb")],
    cpu=2.0,
    memory=4096,
    timeout=3600,
    # keep_warm=1,  # optional: uncomment during demo hours so answers start
    # fast; costs accrue while the container idles, so keep it off otherwise.
)
@modal.asgi_app()
def fastapi_app():
    """Return the repo's FastAPI app (imports run inside the container)."""
    import os
    import sys

    sys.path.insert(0, REMOTE_APP_DIR)
    # Persist the Hugging Face model cache on the Volume so the TTS voice
    # model downloads once and survives cold starts. Must be set before
    # importing api.server (it resolves the cache dir at import time).
    os.environ.setdefault("HF_HUB_CACHE", f"{MODEL_MOUNT}/hub")
    from api.server import app as _app

    return _app
