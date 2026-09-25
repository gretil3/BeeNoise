"""Audio/video loading, resampling, saving and mic recording.

Canonical in-memory format everywhere: mono float32 numpy array in [-1, 1]
plus an explicit sample rate. Convert at the boundary (this module), never in
the middle of the pipeline.

Decoding goes through ffmpeg so any input works (mp4, mkv, mov, mp3, m4a,
wav...). If ffmpeg isn't on PATH we use the binary bundled with the
`imageio-ffmpeg` pip package, so nobody on the team has to install it by hand.
"""
import re
import shutil
import subprocess
from math import gcd

import numpy as np


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_audio(path, sr: int) -> np.ndarray:
    """Decode any audio/video file to mono float32 at `sr`."""
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
           "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}:\n"
                           f"{proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def has_video(path) -> bool:
    """True if the file has a real video stream (ignores mp3 cover art)."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                          capture_output=True)
    info = proc.stderr.decode(errors="replace")
    return any("attached pic" not in line
               for line in re.findall(r"Stream #.*Video:.*", info))


def resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav
    from scipy.signal import resample_poly
    g = gcd(sr_in, sr_out)
    return resample_poly(wav, sr_out // g, sr_in // g).astype(np.float32)


def save_wav(path, wav: np.ndarray, sr: int):
    import soundfile as sf
    sf.write(str(path), np.clip(wav, -1.0, 1.0), sr, subtype="PCM_16")


def cut(wav: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    return wav[max(0, int(start * sr)):max(0, int(end * sr))]


def record_until_enter(sr: int, max_seconds: float) -> np.ndarray:
    """Record from the default mic until the user presses Enter (or max_seconds)."""
    import threading

    import sounddevice as sd

    chunks: list[np.ndarray] = []
    done = threading.Event()

    def callback(indata, frames, time_info, status):
        chunks.append(indata[:, 0].copy())
        if sum(len(c) for c in chunks) >= max_seconds * sr:
            done.set()

    threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=callback):
        print(f"  recording... press Enter to stop (auto-stops at {max_seconds:.0f}s)")
        done.wait()
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
