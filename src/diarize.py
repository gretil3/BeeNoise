"""Stage 2 — diarization: "who spoke when", with anonymous labels.

Output is a list of Segment(start, end, "SPEAKER_00" ...). Labels only mean
"same voice as other segments with this label" — naming them is stage 3.

Backends (config.yaml `diarize.backend`):

pyannote       pyannote/speaker-diarization-3.1. (1) A segmentation network
               (PyanNet: SincNet + LSTM) scans 10 s windows and outputs
               frame-level activity for up to 3 local speakers, overlap
               included (powerset encoding). (2) Each local speaker gets a
               WeSpeaker-ResNet34 embedding. (3) Agglomerative clustering links
               local speakers across windows into global speakers.
ecapa_cluster  Our own baseline, same idea minus the neural segmenter: VAD ->
               sliding 1.5 s windows -> ECAPA embedding per window ->
               agglomerative clustering on cosine distance. Cannot handle
               overlapping speech (one label per window). No HF token needed.
"""
import numpy as np

from .config import CFG, hf_token, resolve_device
from .segments import Segment, merge_adjacent

BACKENDS = ("pyannote", "ecapa_cluster")

_pipeline = None


def _get_pyannote():
    global _pipeline
    if _pipeline is None:
        import torch
        from pyannote.audio import Pipeline

        token = hf_token()
        if token is None:
            raise RuntimeError(
                "pyannote needs a HuggingFace token. Copy .env.example to .env, "
                "put your HF_TOKEN in it, and accept the terms on the model pages "
                "listed there. Or run with --diarizer ecapa_cluster.")
        name = CFG["diarize"]["pyannote_model"]
        print(f"Loading {name} (first run downloads ~30 MB)...")
        try:
            _pipeline = Pipeline.from_pretrained(name, use_auth_token=token)
        except TypeError:  # pyannote >= 4 renamed the argument
            _pipeline = Pipeline.from_pretrained(name, token=token)
        if _pipeline is None:
            raise RuntimeError(
                f"Could not load {name}. Make sure you accepted the user conditions "
                "on huggingface.co for both pyannote/speaker-diarization-3.1 and "
                "pyannote/segmentation-3.0 with the account that owns HF_TOKEN.")
        _pipeline.to(torch.device(resolve_device(CFG["diarize"]["device"])))
    return _pipeline


def _diarize_pyannote(wav16: np.ndarray, sr: int, **speaker_kwargs) -> list[Segment]:
    import torch
    pipeline = _get_pyannote()
    # Pass the waveform in memory — avoids pyannote's own file decoding, which
    # is the usual source of Windows audio-backend errors.
    out = pipeline({"waveform": torch.from_numpy(wav16).unsqueeze(0), "sample_rate": sr},
                   **speaker_kwargs)
    annotation = getattr(out, "speaker_diarization", out)  # pyannote 4 wraps the result
    return [Segment(float(turn.start), float(turn.end), str(label))
            for turn, _, label in annotation.itertracks(yield_label=True)]


def _diarize_ecapa_cluster(wav16: np.ndarray, sr: int, num_speakers=None,
                           min_speakers=None, max_speakers=None) -> list[Segment]:
    from scipy.cluster.hierarchy import fcluster, linkage

    from .encoder import embed_many
    from .vad import FRAME_MS, speech_mask

    win, hop = CFG["diarize"]["window_sec"], CFG["diarize"]["hop_sec"]
    mask = speech_mask(wav16, sr)
    frames_per_sec = 1000 / FRAME_MS

    starts, chunks = [], []
    t = 0.0
    while t + win <= len(wav16) / sr:
        f0, f1 = int(t * frames_per_sec), int((t + win) * frames_per_sec)
        if mask[f0:f1].mean() >= 0.5:  # mostly speech -> worth embedding
            starts.append(t)
            chunks.append(wav16[int(t * sr):int(t * sr) + int(win * sr)])
        t += hop
    if not chunks:
        return []
    if len(chunks) == 1:
        return [Segment(starts[0], starts[0] + win, "SPEAKER_00")]

    emb = embed_many(chunks)
    Z = linkage(emb, method="average", metric="cosine")
    if num_speakers:
        labels = fcluster(Z, t=num_speakers, criterion="maxclust")
    else:
        labels = fcluster(Z, t=CFG["diarize"]["cluster_threshold"], criterion="distance")
        n = len(set(labels))
        if max_speakers and n > max_speakers:
            labels = fcluster(Z, t=max_speakers, criterion="maxclust")
        elif min_speakers and n < min_speakers:
            labels = fcluster(Z, t=min_speakers, criterion="maxclust")

    # Each window "owns" the hop-sized slice around its centre.
    segs = []
    for s, lab in zip(starts, labels, strict=True):
        centre = s + win / 2
        segs.append(Segment(max(0.0, centre - hop / 2), centre + hop / 2, f"SPEAKER_{lab - 1:02d}"))
    return merge_adjacent(segs, max_gap=hop)


def diarize(wav16: np.ndarray, sr: int, backend: str | None = None,
            num_speakers: int | None = None, min_speakers: int | None = None,
            max_speakers: int | None = None) -> list[Segment]:
    backend = backend or CFG["diarize"]["backend"]
    kwargs = {
        "num_speakers": num_speakers or CFG["diarize"]["num_speakers"],
        "min_speakers": min_speakers or CFG["diarize"]["min_speakers"],
        "max_speakers": max_speakers or CFG["diarize"]["max_speakers"],
    }
    kwargs = {k: v for k, v in kwargs.items() if v}
    if backend == "pyannote":
        segs = _diarize_pyannote(wav16, sr, **kwargs)
    elif backend == "ecapa_cluster":
        segs = _diarize_ecapa_cluster(wav16, sr, **kwargs)
    else:
        raise ValueError(f"Unknown diarize backend {backend!r}; expected one of {BACKENDS}")
    return sorted(segs, key=lambda s: s.start)
