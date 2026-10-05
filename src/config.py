"""Loads config.yaml once and exposes it as CFG. Every module imports from here
instead of re-reading the file.

Also loads `.env` (for HF_TOKEN) and points the HuggingFace cache at the
project folder. Both must happen before huggingface_hub / pyannote /
speechbrain / faster_whisper are imported anywhere, which is why this module
is imported first by everything else.
"""
import os
import warnings
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path):
    """Minimal .env reader (KEY=VALUE lines). Real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

# Models total ~1-2 GB. Keep them next to the project instead of the default
# C:\Users\<you>\.cache — on a small C: drive that download otherwise fails.
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# We only ever save charts to a file (fig.savefig in eval/*.py), never show
# one interactively — force a non-interactive backend. Without this, a
# venv/subprocess launched from inside Jupyter/Colab/VS Code inherits an
# MPLBACKEND env var pointing at that host's *own* inline-plotting backend
# (e.g. "module://matplotlib_inline.backend_inline"), which this project's
# matplotlib install doesn't have and crashes on import. Override, don't
# setdefault: the whole point is this env var is already set to something
# that breaks here.
os.environ["MPLBACKEND"] = "Agg"

# Library deprecation chatter that isn't actionable for us (DeepFilterNet's
# old torchaudio import path, SpeechBrain's torch.load call).
warnings.filterwarnings("ignore", message=r".*torchaudio\.backend\.common.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=r".*weights_only=False.*")

with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def path(key: str) -> Path:
    """Resolve a paths.* entry to an absolute Path, creating its parent dir."""
    p = ROOT / CFG["paths"][key]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def resolve_device(value: str | None) -> str:
    """"auto" (or None) -> "cuda" when a GPU is usable, else "cpu"; anything else as given."""
    if value not in (None, "auto"):
        return value
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    return token if token and not token.startswith("hf_xxx") else None
