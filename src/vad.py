"""Voice activity detection with webrtcvad (GMM-based, very fast).

Used to strip silence from enrollment recordings and to find speech regions
for the ecapa_cluster diarizer. webrtcvad needs 16-bit PCM at 8/16/32/48 kHz
in frames of exactly 10/20/30 ms.
"""
import numpy as np

from .config import CFG

FRAME_MS = 30


def speech_mask(wav: np.ndarray, sr: int, aggressiveness: int | None = None) -> np.ndarray:
    """One bool per 30 ms frame: True = speech."""
    import webrtcvad
    level = CFG["enroll"]["vad_aggressiveness"] if aggressiveness is None else aggressiveness
    vad = webrtcvad.Vad(level)
    n = int(sr * FRAME_MS / 1000)
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    return np.array([vad.is_speech(pcm[i:i + n].tobytes(), sr)
                     for i in range(0, len(pcm) - n + 1, n)], dtype=bool)


def speech_only(wav: np.ndarray, sr: int) -> np.ndarray:
    """Concatenate just the voiced frames."""
    n = int(sr * FRAME_MS / 1000)
    mask = speech_mask(wav, sr)
    if not mask.any():
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([wav[i * n:(i + 1) * n] for i in np.flatnonzero(mask)])


def speech_seconds(wav: np.ndarray, sr: int) -> float:
    return float(speech_mask(wav, sr).sum()) * FRAME_MS / 1000.0
