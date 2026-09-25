"""Setup check for a new teammate: python -m src.check_env

Checks Python version, every library, ffmpeg, the HF token, and (with
--models) actually loads each model once, which also pre-downloads them.
"""
import argparse
import importlib
import sys

OK, BAD, WARN = "  [ok]  ", "  [FAIL]", "  [warn]"


def check_imports() -> bool:
    good = True
    v = sys.version_info
    if (v.major, v.minor) not in ((3, 10), (3, 11)):
        print(f"{BAD} Python {v.major}.{v.minor} — use 3.10 or 3.11 (DeepFilterNet has no "
              f"Windows wheels for 3.12+)")
        good = False
    else:
        print(f"{OK} Python {v.major}.{v.minor}")

    for mod, why in [("numpy", "core"), ("scipy", "core"), ("yaml", "config"),
                     ("soundfile", "wav I/O"), ("sounddevice", "mic recording"),
                     ("webrtcvad", "VAD"), ("torch", "DL runtime"), ("torchaudio", "DL runtime"),
                     ("df.enhance", "stage 1 DeepFilterNet"), ("noisereduce", "stage 1 baseline"),
                     ("pyannote.audio", "stage 2 diarization"),
                     ("speechbrain", "stage 3 ECAPA"), ("faster_whisper", "stage 4 Whisper"),
                     ("jiwer", "WER eval"), ("pystoi", "STOI eval"),
                     ("pyannote.metrics", "DER eval"), ("matplotlib", "plots"),
                     ("gradio", "demo UI")]:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            print(f"{OK} {mod:<18} {ver:<12} ({why})")
        except Exception as e:  # noqa: BLE001 — report anything, keep going
            print(f"{BAD} {mod:<18} ({why}): {type(e).__name__}: {e}")
            good = False

    try:
        import numpy
        if int(numpy.__version__.split(".")[0]) >= 2:
            print(f"{BAD} numpy {numpy.__version__} — DeepFilterNet needs numpy<2")
            good = False
    except ImportError:
        pass

    from .audio_io import ffmpeg_exe
    try:
        print(f"{OK} ffmpeg: {ffmpeg_exe()}")
    except Exception as e:  # noqa: BLE001
        print(f"{BAD} ffmpeg not found ({e}) — pip install imageio-ffmpeg")
        good = False

    from .config import hf_token
    if hf_token():
        print(f"{OK} HF_TOKEN is set")
    else:
        print(f"{WARN} HF_TOKEN not set — pyannote diarization won't load. See .env.example "
              f"(or use --diarizer ecapa_cluster)")
    return good


def check_models() -> bool:
    import numpy as np

    from . import denoise, diarize, encoder, stt  # all lazy — nothing loads yet
    good = True
    sr = 16000
    noise = (np.random.default_rng(0).standard_normal(sr * 3) * 0.05).astype(np.float32)
    steps = [
        ("DeepFilterNet", lambda: denoise.denoise(noise, sr, "deepfilternet")),
        ("ECAPA-TDNN", lambda: encoder.embed(noise)),
        ("Whisper", lambda: stt.transcribe_text(noise, "en")),
        ("pyannote", lambda: diarize._get_pyannote()),
    ]
    for name, fn in steps:
        try:
            fn()
            print(f"{OK} {name} loads and runs")
        except Exception as e:  # noqa: BLE001
            print(f"{BAD} {name}: {type(e).__name__}: {e}")
            good = False
    return good


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", action="store_true",
                   help="also load every model (downloads ~1-2 GB on first run)")
    args = p.parse_args()
    print("Libraries:")
    ok = check_imports()
    if args.models:
        print("\nModels:")
        ok = check_models() and ok
    print("\nAll good." if ok else "\nSome checks failed — see README 'Setup'.")
    sys.exit(0 if ok else 1)
