"""Speech-to-text via faster-whisper (CTranslate2-backed, fast on CPU)."""
import numpy as np

from .config import CFG

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print(f"Loading faster-whisper '{CFG['stt']['model_size']}' "
              f"(first run downloads the model)...")
        _model = WhisperModel(
            CFG["stt"]["model_size"],
            device=CFG["stt"]["device"],
            compute_type=CFG["stt"]["compute_type"],
        )
    return _model


def transcribe(wav16k: np.ndarray):
    """Returns (text, avg_logprob, no_speech_prob). Caller should gate on
    confidence before trusting the text — see config.yaml stt thresholds."""
    model = _get_model()
    segments, info = model.transcribe(
        wav16k,
        language=CFG["stt"]["language"],
        beam_size=CFG["stt"]["beam_size"],
        vad_filter=True,
    )
    segments = list(segments)
    if not segments:
        return "", -9.9, 1.0

    text = " ".join(s.text.strip() for s in segments).strip()
    avg_logprob = float(np.mean([s.avg_logprob for s in segments]))
    no_speech_prob = float(np.mean([getattr(s, "no_speech_prob", 0.0) for s in segments]))
    return text, avg_logprob, no_speech_prob


def transcribe_or_none(wav16k: np.ndarray):
    """Applies the confidence gate from config.yaml. Returns None if the
    transcription is too unreliable to hand to the LLM."""
    text, avg_logprob, no_speech_prob = transcribe(wav16k)
    if not text:
        return None
    if avg_logprob < CFG["stt"]["min_avg_logprob"]:
        return None
    if no_speech_prob > CFG["stt"]["max_no_speech_prob"]:
        return None
    return text


if __name__ == "__main__":
    from .audio_io import record_fixed
    from .vad import trim_silence

    print("Say a sentence for 4 seconds...")
    clip = record_fixed(4.0)
    clip = trim_silence(clip)
    text, logprob, nsp = transcribe(clip)
    print(f"Text: {text!r}")
    print(f"avg_logprob={logprob:.3f}  no_speech_prob={nsp:.3f}")
