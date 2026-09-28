"""Helpers shared by the eval scripts: noise mixing, signal metrics, I/O.

The metric functions are pure numpy (unit-tested in tests/test_metrics.py).
"""
import json
from pathlib import Path

import numpy as np

from src.config import CFG, ROOT

EVAL_DIR = ROOT / CFG["paths"]["eval_dir"]
RESULTS_DIR = ROOT / CFG["paths"]["results_dir"]
NOISE_DIR = EVAL_DIR / "noise"
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".mp4", ".mkv", ".mov", ".webm"}


def list_audio(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in AUDIO_EXTS)


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add `noise` to `clean`, scaled so the result has exactly `snr_db` SNR.
    The noise is looped/cropped to the clean clip's length."""
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))
    noise = noise[:len(clean)]
    p_clean = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(p_clean / (p_noise * 10 ** (snr_db / 10)))
    mix = clean + scale * noise
    peak = np.max(np.abs(mix))
    return (mix / peak * 0.99 if peak > 0.99 else mix).astype(np.float32)


def _align(ref: np.ndarray, est: np.ndarray):
    n = min(len(ref), len(est))
    return ref[:n].astype(np.float64), est[:n].astype(np.float64)


def snr_db(ref: np.ndarray, est: np.ndarray) -> float:
    """Plain SNR of `est` against the clean reference `ref`."""
    ref, est = _align(ref, est)
    return float(10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - est) ** 2) + 1e-12)))


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant SDR (Le Roux et al., 2019): like SNR, but ignores a
    global gain difference, so a denoiser that also changes loudness isn't
    unfairly penalised."""
    ref, est = _align(ref, est)
    ref = ref - ref.mean()
    est = est - est.mean()
    target = (np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)) * ref
    return float(10 * np.log10(np.sum(target ** 2) / (np.sum((est - target) ** 2) + 1e-12)))


def load_noises(sr: int) -> list[np.ndarray]:
    from src.audio_io import load_audio
    files = list_audio(NOISE_DIR)
    if not files:
        raise SystemExit(f"No noise files in {NOISE_DIR}. Add a few .wav recordings of "
                         f"background noise (cafe, traffic, fan...) — see data/README.md.")
    return [load_audio(f, sr) for f in files]


def save_results(name: str, payload) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    return out


def print_table(rows: list[dict], cols: list[str]):
    if not rows:
        raise SystemExit("No results to report — check the eval data paths (see data/README.md).")
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return "" if v is None else str(v)
