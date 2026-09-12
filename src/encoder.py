"""Speaker embedding via SpeechBrain's ECAPA-TDNN (pretrained on VoxCeleb).

Produces a 192-d L2-normalised embedding per utterance. Normalising here
means cosine similarity downstream is just a dot product — never skip it.
"""
import numpy as np
import torch

from .config import CFG, ROOT

_enc = None


def _get_encoder():
    global _enc
    if _enc is None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        savedir = ROOT / CFG["speaker"]["model_dir"]
        savedir.mkdir(parents=True, exist_ok=True)
        print("Loading ECAPA-TDNN speaker encoder (first run downloads ~80MB)...")
        _enc = EncoderClassifier.from_hparams(
            source=CFG["speaker"]["model_source"],
            savedir=str(savedir),
            # Windows blocks symlink creation without admin/dev-mode privileges;
            # copying the cached files instead avoids WinError 1314.
            local_strategy=LocalStrategy.COPY,
        )
    return _enc


def embed(wav16k: np.ndarray) -> np.ndarray:
    """wav16k: mono float32 array at 16kHz. Returns L2-normalised (192,) vector."""
    enc = _get_encoder()
    t = torch.from_numpy(wav16k.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        e = enc.encode_batch(t).squeeze().cpu().numpy()
    norm = np.linalg.norm(e)
    if norm < 1e-8:
        return e  # degenerate silent input — caller should reject upstream
    return e / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Both should already be unit-norm; this is just a dot product then."""
    return float(np.dot(a, b))


if __name__ == "__main__":
    from .audio_io import record_fixed
    from .vad import trim_silence, net_speech_seconds

    print("Say something for 4 seconds to test the encoder...")
    clip = record_fixed(4.0)
    clip = trim_silence(clip)
    print(f"Net speech: {net_speech_seconds(clip):.2f}s")
    e = embed(clip)
    print(f"Embedding shape: {e.shape}, norm: {np.linalg.norm(e):.4f}")
