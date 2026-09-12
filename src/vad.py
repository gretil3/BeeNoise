"""Voice activity detection using webrtcvad.

webrtcvad requires 16-bit PCM mono audio at 8/16/32/48 kHz, in frames of
exactly 10/20/30 ms. We standardize on 16 kHz / 30 ms frames throughout.
"""
import numpy as np
import webrtcvad

from .config import CFG

SR = CFG["audio"]["sample_rate"]
FRAME_MS = CFG["audio"]["frame_ms"]
FRAME_SAMPLES = int(SR * FRAME_MS / 1000)

_vad = webrtcvad.Vad(CFG["audio"]["vad_aggressiveness"])


def _float_to_pcm16(frame: np.ndarray) -> bytes:
    clipped = np.clip(frame, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def is_speech(frame: np.ndarray) -> bool:
    """frame must be exactly FRAME_SAMPLES long, float32 in [-1, 1]."""
    if len(frame) != FRAME_SAMPLES:
        # pad/truncate defensively — happens on the last frame of a stream
        if len(frame) < FRAME_SAMPLES:
            frame = np.pad(frame, (0, FRAME_SAMPLES - len(frame)))
        else:
            frame = frame[:FRAME_SAMPLES]
    return _vad.is_speech(_float_to_pcm16(frame), SR)


def trim_silence(audio: np.ndarray) -> np.ndarray:
    """Trim leading/trailing non-speech frames from a full utterance.
    Used to clean up enrollment/eval clips before embedding."""
    n_frames = len(audio) // FRAME_SAMPLES
    if n_frames == 0:
        return audio

    flags = [is_speech(audio[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES])
             for i in range(n_frames)]

    if not any(flags):
        return audio  # nothing flagged as speech — return as-is, let caller decide

    first = flags.index(True)
    last = len(flags) - 1 - flags[::-1].index(True)
    start = first * FRAME_SAMPLES
    end = (last + 1) * FRAME_SAMPLES
    return audio[start:end]


def net_speech_seconds(audio: np.ndarray) -> float:
    """How many seconds of the clip webrtcvad flags as actual speech —
    used to reject too-short utterances before trusting a speaker ID."""
    n_frames = len(audio) // FRAME_SAMPLES
    if n_frames == 0:
        return 0.0
    speech_frames = sum(
        is_speech(audio[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES])
        for i in range(n_frames)
    )
    return speech_frames * FRAME_MS / 1000.0
