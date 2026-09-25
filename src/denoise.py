"""Stage 1 — denoise: noisy audio in, cleaned audio out.

Backends (config.yaml `denoise.backend`):

deepfilternet  DeepFilterNet3 (Schröter et al., 2022). Works in the STFT
               domain at 48 kHz in two steps: (1) a network predicts a gain
               per ERB band — a coarse "how much of this band is speech"
               envelope — and (2) a second network predicts a short complex
               "deep filter" per low-frequency bin that reconstructs the
               periodic harmonic structure of voiced speech. Small (~2M
               params), runs faster than real time on CPU.
spectral       Classical spectral gating (noisereduce): estimate the noise
               spectrum per frequency, then attenuate time-frequency bins that
               don't rise above it. No learning — the non-DL baseline.
none           Pass-through. Used for the "without denoising" ablation.
"""
import numpy as np

from .audio_io import resample
from .config import CFG

BACKENDS = ("deepfilternet", "spectral", "none")

_df = None  # (model, df_state) — loaded once


def _get_deepfilternet():
    global _df
    if _df is None:
        from df.enhance import init_df
        print("Loading DeepFilterNet3 (first run downloads ~10 MB)...")
        model, df_state, _ = init_df(log_level="WARNING", log_file=None)
        _df = (model, df_state)
    return _df


def denoise(wav: np.ndarray, sr: int, backend: str | None = None) -> tuple[np.ndarray, int]:
    """Returns (denoised_wav, its_sample_rate). DeepFilterNet always returns 48 kHz."""
    backend = backend or CFG["denoise"]["backend"]

    if backend == "none":
        return wav, sr

    if backend == "spectral":
        import noisereduce as nr
        out = nr.reduce_noise(y=wav, sr=sr, stationary=False,
                              prop_decrease=CFG["denoise"]["spectral_prop_decrease"])
        return out.astype(np.float32), sr

    if backend == "deepfilternet":
        import torch
        from df.enhance import enhance
        model, df_state = _get_deepfilternet()
        df_sr = df_state.sr()
        x = torch.from_numpy(resample(wav, sr, df_sr)).unsqueeze(0)
        with torch.no_grad():
            y = enhance(model, df_state, x, atten_lim_db=CFG["denoise"]["atten_lim_db"])
        return y.squeeze(0).cpu().numpy().astype(np.float32), df_sr

    raise ValueError(f"Unknown denoise backend {backend!r}; expected one of {BACKENDS}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from .audio_io import load_audio, save_wav

    p = argparse.ArgumentParser(description="Denoise a single file.")
    p.add_argument("input")
    p.add_argument("--backend", choices=BACKENDS, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    raw = load_audio(args.input, CFG["audio"]["output_sr"])
    clean, sr = denoise(raw, CFG["audio"]["output_sr"], args.backend)
    out = Path(args.out or Path(args.input).with_name(Path(args.input).stem + "_denoised.wav"))
    save_wav(out, clean, sr)
    print(f"Saved {out}")
