"""Audio capture and playback.

Canonical format for the whole project: 16 kHz, mono, float32 in [-1, 1].
Convert at the boundary (this module), never in the middle of the pipeline.
"""
import numpy as np
import sounddevice as sd
import soundfile as sf

from .config import CFG

SR = CFG["audio"]["sample_rate"]
CHANNELS = CFG["audio"]["channels"]


def record_fixed(seconds: float) -> np.ndarray:
    """Record a fixed-length clip. Used for enrollment prompts."""
    n = int(seconds * SR)
    print(f"  recording {seconds:.1f}s ...")
    audio = sd.rec(n, samplerate=SR, channels=CHANNELS, dtype="float32")
    sd.wait()
    return audio.reshape(-1)


def record_until_silence(max_seconds: float = 15.0) -> np.ndarray:
    """Stream in frame_ms chunks; stop after silence_ms of trailing silence.
    Keeps a pre-roll buffer so the first phoneme isn't clipped.
    Requires VAD (see vad.py) — imported lazily to avoid a circular import.
    """
    from .vad import is_speech, FRAME_SAMPLES

    frame_ms = CFG["audio"]["frame_ms"]
    silence_ms = CFG["audio"]["silence_ms"]
    preroll_ms = CFG["audio"]["preroll_ms"]
    min_utt_sec = CFG["audio"]["min_utt_sec"]

    preroll_frames = max(1, preroll_ms // frame_ms)
    silence_frames_needed = max(1, silence_ms // frame_ms)
    max_frames = int(max_seconds * 1000 / frame_ms)

    ring = []              # pre-roll ring buffer (frames, always kept)
    buf = []                # frames belonging to the current utterance
    triggered = False
    trailing_silence = 0
    frames_seen = 0

    print("  listening... (speak now)")
    stream = sd.InputStream(samplerate=SR, channels=CHANNELS, dtype="float32",
                             blocksize=FRAME_SAMPLES)
    with stream:
        while frames_seen < max_frames:
            frame, _ = stream.read(FRAME_SAMPLES)
            frame = frame.reshape(-1)
            frames_seen += 1

            speech = is_speech(frame)

            if not triggered:
                ring.append(frame)
                if len(ring) > preroll_frames:
                    ring.pop(0)
                if speech:
                    triggered = True
                    buf.extend(ring)
                    buf.append(frame)
                    trailing_silence = 0
            else:
                buf.append(frame)
                if speech:
                    trailing_silence = 0
                else:
                    trailing_silence += 1
                    if trailing_silence >= silence_frames_needed:
                        break

    if not buf:
        return np.zeros(0, dtype=np.float32)

    audio = np.concatenate(buf)
    if len(audio) / SR < min_utt_sec:
        return np.zeros(0, dtype=np.float32)
    return audio


def play(audio: np.ndarray, sr: int = SR):
    sd.play(audio, samplerate=sr)
    sd.wait()


def save_wav(path, audio: np.ndarray, sr: int = SR):
    sf.write(str(path), audio, sr)


def load_wav(path) -> np.ndarray:
    """Load a WAV and resample/convert to canonical 16k mono float32."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample
        n_target = int(len(audio) * SR / sr)
        audio = resample(audio, n_target).astype(np.float32)
    return audio


if __name__ == "__main__":
    # Smoke test: record 3s, play it back, save to disk.
    print("Recording 3 seconds — say something...")
    clip = record_fixed(3.0)
    print(f"Captured {len(clip)} samples ({len(clip)/SR:.2f}s). Playing back...")
    play(clip)
    out = CFG["paths"]["enroll_dir"]
    from .config import ROOT
    out_path = ROOT / out / "smoketest.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_wav(out_path, clip)
    print(f"Saved to {out_path}")
