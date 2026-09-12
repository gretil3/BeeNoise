"""ASR evaluation: WER across Whisper model sizes.

Expected layout:
    data/eval/wer/transcripts.json   -> {"clip1.wav": "the reference sentence", ...}
    data/eval/wer/*.wav              -> the audio files referenced above

Run from the project root:
    python -m eval.wer
    python -m eval.wer --models tiny base small
"""
import argparse
import json
import time
from pathlib import Path

import jiwer

from src.audio_io import load_wav
from src.config import ROOT
from src.vad import trim_silence

WER_DIR = ROOT / "data" / "eval" / "wer"
RESULTS_DIR = ROOT / "eval" / "results"

TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def evaluate_model(model_size: str, transcripts: dict, device="cpu", compute_type="int8"):
    from faster_whisper import WhisperModel
    print(f"\nLoading Whisper '{model_size}'...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    refs, hyps, latencies = [], [], []
    for fname, ref_text in transcripts.items():
        wav_path = WER_DIR / fname
        if not wav_path.exists():
            print(f"  (missing {fname}, skipping)")
            continue
        audio = load_wav(wav_path)
        audio = trim_silence(audio)

        t0 = time.time()
        segments, _ = model.transcribe(audio, language="en", beam_size=5, vad_filter=True)
        hyp_text = " ".join(s.text.strip() for s in segments).strip()
        latencies.append(time.time() - t0)

        refs.append(ref_text)
        hyps.append(hyp_text)

    if not refs:
        return None

    wer = jiwer.wer(refs, hyps, truth_transform=TRANSFORM, hypothesis_transform=TRANSFORM)
    return {
        "model": model_size,
        "wer": wer,
        "n_clips": len(refs),
        "avg_latency_sec": sum(latencies) / len(latencies),
        "examples": list(zip(refs[:3], hyps[:3])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["tiny", "base", "small"])
    args = parser.parse_args()

    transcript_path = WER_DIR / "transcripts.json"
    if not transcript_path.exists():
        print(f"No transcripts found at {transcript_path}.")
        print("Create data/eval/wer/transcripts.json mapping wav filename -> "
              "reference text, plus the wav files themselves, before running this.")
        return

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcripts = json.load(f)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for model_size in args.models:
        r = evaluate_model(model_size, transcripts)
        if r:
            print(f"  {model_size}: WER={r['wer']*100:.2f}%  "
                  f"avg_latency={r['avg_latency_sec']:.2f}s  n={r['n_clips']}")
            results.append(r)

    with open(RESULTS_DIR / "wer_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved WER results to {RESULTS_DIR / 'wer_summary.json'}")


if __name__ == "__main__":
    main()
