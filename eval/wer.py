"""Metric 4 — transcription accuracy (WER), per noise level, with vs without
denoising. This is the key "does denoising even help?" result.

    WER = (substitutions + deletions + insertions) / words in reference

For every clean clip we build noisy versions at several SNRs (same noise
files as eval/denoise_quality.py), then transcribe each one with and without
denoising. Plot: WER vs input SNR, one line per denoiser.

Needs:
    data/eval/wer/transcripts.json   {"clip1.wav": "the reference text", ...}
    data/eval/wer/*.wav              the clips (clean, single speaker)
    data/eval/noise/*.wav            background noise

Run:
    python -m eval.wer
    python -m eval.wer --snrs 0 5 10 --backends none deepfilternet spectral --language en
"""
import argparse
import json

import jiwer

from src.audio_io import load_audio, resample
from src.config import CFG
from src.denoise import denoise
from src.stt import transcribe_text

from .common import EVAL_DIR, RESULTS_DIR, load_noises, mix_at_snr, print_table, save_results

WER_DIR = EVAL_DIR / "wer"
SR = CFG["audio"]["analysis_sr"]

_NORMALISE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def wer(refs: list[str], hyps: list[str]) -> float:
    return float(jiwer.wer(refs, hyps, reference_transform=_NORMALISE,
                           hypothesis_transform=_NORMALISE))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snrs", nargs="+", type=float, default=[0, 5, 10, 20],
                   help="noise levels; the clean condition is always included")
    p.add_argument("--backends", nargs="+", default=["none", CFG["denoise"]["backend"]])
    p.add_argument("--language", default=CFG["stt"]["language"])
    args = p.parse_args()

    tpath = WER_DIR / "transcripts.json"
    if not tpath.exists():
        raise SystemExit(f"Missing {tpath} — see data/README.md.")
    transcripts = json.loads(tpath.read_text(encoding="utf-8"))
    noises = load_noises(SR)

    conditions = ["clean"] + args.snrs
    refs = {(c, b): [] for c in conditions for b in args.backends}
    hyps = {(c, b): [] for c in conditions for b in args.backends}
    examples = []
    for ci, (fname, ref_text) in enumerate(transcripts.items()):
        path = WER_DIR / fname
        if not path.exists():
            print(f"  (missing {fname}, skipping)")
            continue
        clean = load_audio(path, SR)
        for cond in conditions:
            audio = clean if cond == "clean" else mix_at_snr(clean, noises[ci % len(noises)], cond)
            for backend in args.backends:
                out, out_sr = denoise(audio, SR, backend)
                hyp = transcribe_text(resample(out, out_sr, SR), args.language)
                refs[(cond, backend)].append(ref_text)
                hyps[(cond, backend)].append(hyp)
                if ci < 2:
                    examples.append({"clip": fname, "snr": cond, "denoise": backend,
                                     "ref": ref_text, "hyp": hyp})
        print(f"  {fname} done")

    rows = [{"snr": c, "denoise": b, "WER": wer(refs[(c, b)], hyps[(c, b)]),
             "n_clips": len(refs[(c, b)])}
            for c in conditions for b in args.backends if refs[(c, b)]]
    print()
    print_table(rows, ["snr", "denoise", "WER", "n_clips"])
    save_results("wer", {"model": CFG["stt"]["model_size"], "rows": rows, "examples": examples})

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [str(int(c)) if c != "clean" else "clean" for c in conditions]
    for b in args.backends:
        ax.plot(xs, [next(r["WER"] for r in rows if r["snr"] == c and r["denoise"] == b) * 100
                     for c in conditions], "o-", label=f"denoise={b}")
    ax.set(xlabel="input SNR (dB)", ylabel="WER (%)",
           title=f"Does denoising help Whisper-{CFG['stt']['model_size']}?")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "wer_vs_snr.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'wer_vs_snr.png'}")


if __name__ == "__main__":
    main()
