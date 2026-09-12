"""Loads config.yaml once and exposes it as a dict. Every module imports CFG
from here instead of re-reading the file."""
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent

# Speaker/ASR models are pulled from the HuggingFace Hub on first use and can
# total several hundred MB. Point the cache at this project's own drive
# instead of the default C:\Users\<you>\.cache\huggingface — on a low-space
# C: drive that download will otherwise fail outright. Must be set before
# huggingface_hub/speechbrain/faster_whisper are imported anywhere, which is
# why this lives at the top of config.py (imported first by every module).
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

_CFG_PATH = ROOT / "config.yaml"

with open(_CFG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def path(key: str) -> Path:
    """Resolve a paths.* entry from config.yaml to an absolute Path,
    creating parent directories as needed."""
    p = ROOT / CFG["paths"][key]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
