"""Stage 4 — transcription with Whisper (via faster-whisper).

Whisper is an encoder-decoder Transformer trained on 680k hours of weakly
supervised audio: log-Mel spectrogram (30 s windows) -> encoder -> a decoder
that autoregressively emits text tokens plus timestamp tokens.
faster-whisper runs the same weights on CTranslate2 (int8 on CPU), ~4x faster
than the reference implementation. `word_timestamps=True` aligns each word
using the decoder's cross-attention — that's what lets us attach a speaker to
every word in stage 5.
"""
import numpy as np

from .audio_io import cut
from .config import CFG
from .segments import Segment, Word, merge_adjacent

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        size = CFG["stt"]["model_size"]
        print(f"Loading Whisper '{size}' (first run downloads the model)...")
        _model = WhisperModel(size, device=CFG["stt"]["device"],
                              compute_type=CFG["stt"]["compute_type"])
    return _model


def transcribe_words(wav16: np.ndarray, language: str | None = None,
                     offset: float = 0.0, speaker: str | None = None) -> tuple[list[Word], str]:
    """Returns (words with absolute timestamps, detected language)."""
    segments, info = _get_model().transcribe(
        wav16,
        language=language or CFG["stt"]["language"],
        beam_size=CFG["stt"]["beam_size"],
        vad_filter=True,
        word_timestamps=True,
    )
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append(Word(w.start + offset, w.end + offset, w.word, speaker, w.probability))
    return words, info.language


def transcribe_text(wav16: np.ndarray, language: str | None = None) -> str:
    words, _ = transcribe_words(wav16, language)
    return "".join(w.text for w in words).strip()


def transcribe_turns(wav16: np.ndarray, sr: int, turns: list[Segment],
                     language: str | None = None) -> tuple[list[Word], str]:
    """`segment` strategy: transcribe each speaker turn on its own. Words come
    back already labelled with the turn's speaker."""
    pad = 0.2
    words, langs = [], []
    for t in merge_adjacent(turns, max_gap=0.5):
        start = max(0.0, t.start - pad)
        clip = cut(wav16, sr, start, t.end + pad)
        if len(clip) < int(0.3 * sr):
            continue
        w, lang = transcribe_words(clip, language, offset=start, speaker=t.speaker)
        words.extend(w)
        langs.append(lang)
    lang = max(set(langs), key=langs.count) if langs else (language or "")
    return words, lang
