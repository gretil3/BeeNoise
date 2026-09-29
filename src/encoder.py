"""Speaker embeddings via SpeechBrain's ECAPA-TDNN (pretrained on VoxCeleb).

ECAPA-TDNN turns a variable-length utterance into a fixed 192-d vector that
captures *who* is speaking, not *what* is said: 1-D dilated convolutions over
filterbank features, squeeze-excitation blocks that reweight channels, and
attentive statistics pooling (weighted mean + std over time). Trained with
AAM-softmax on ~7k VoxCeleb speakers, so same-speaker vectors land close
together in cosine space.

Embeddings are L2-normalised here, so cosine similarity downstream is a plain
dot product — never skip the normalisation.
"""
import numpy as np

from .config import CFG, ROOT

EMB_DIM = 192
_enc = None


def _get_encoder():
    global _enc
    if _enc is None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        savedir = ROOT / CFG["speaker"]["model_dir"]
        savedir.mkdir(parents=True, exist_ok=True)
        print("Loading ECAPA-TDNN speaker encoder (first run downloads ~80 MB)...")
        _enc = EncoderClassifier.from_hparams(
            source=CFG["speaker"]["model_source"],
            savedir=str(savedir),
            # Windows blocks symlinks without admin/Developer Mode (WinError 1314).
            local_strategy=LocalStrategy.COPY,
            # The default "custom.py" doesn't exist in this model repo, so every load
            # asked the Hub again and huggingface_hub re-touched an empty
            # .cache/.../.no_exist/custom.py marker -- which `gradio app.py`'s hot
            # reloader sees as a changed .py file, restarting the app mid-run. Pointing
            # at a file savedir always has keeps the load fully local.
            pymodule_file="hyperparams.yaml",
        )
    return _enc


def _normalise(e: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(e, axis=-1, keepdims=True)
    return e / np.maximum(norm, 1e-8)


def embed(wav16k: np.ndarray) -> np.ndarray:
    """Mono float32 at 16 kHz -> unit-norm (192,) vector."""
    import torch
    t = torch.from_numpy(wav16k.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        e = _get_encoder().encode_batch(t).squeeze().cpu().numpy()
    return _normalise(e)


def embed_many(chunks: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    """Embed equal-length chunks in batches -> (N, 192). Faster than a loop."""
    import torch
    if not chunks:
        return np.zeros((0, EMB_DIM), dtype=np.float32)
    out = []
    for i in range(0, len(chunks), batch_size):
        batch = torch.from_numpy(np.stack(chunks[i:i + batch_size]).astype(np.float32))
        with torch.no_grad():
            out.append(_get_encoder().encode_batch(batch).squeeze(1).cpu().numpy())
    return _normalise(np.concatenate(out))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Both unit-norm already, so this is just a dot product."""
    return float(np.dot(a, b))
