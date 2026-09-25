"""Metric 1 — denoising quality.

Takes clean speech clips, mixes in recorded noise at several SNRs, denoises
the mixture with each backend, and measures how much closer to the clean
original it got:

    SNR / SI-SDR improvement (dB)   after - before, higher is better
    STOI                            intelligibility 0..1, before vs after
    PESQ (optional)                 perceptual quality 1..4.5, if `pesq` is installed

Needs:
    data/eval/wer/*.wav     clean read speech (the same clips the WER eval uses)
    data/eval/noise/*.wav   background noise recordings

Run:
    python -m eval.denoise_quality
    python -m eval.denoise_quality --snrs 0 5 10 --backends deepfilternet spectral
"""
import argparse

import numpy as np

from src.audio_io import load_audio, resample
from src.denoise import denoise

from .common import (
    EVAL_DIR,
    RESULTS_DIR,
    list_audio,
    load_noises,
    mix_at_snr,
    print_table,
    save_results,
    si_sdr,
    snr_db,
)

SR = 16000  # STOI/PESQ are defined at 16 kHz (wideband)


def _stoi(ref, est):
    from pystoi import stoi
    n = min(len(ref), len(est))
    return float(stoi(ref[:n], est[:n], SR, extended=False))


def _pesq(ref, est):
    try:
        from pesq import pesq
    except ImportError:
        return None
    n = min(len(ref), len(est))
    try:
        return float(pesq(SR, ref[:n], est[:n], "wb"))
    except Exception:  # noqa: BLE001 — pesq raises on silent/very short input
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-dir", default=str(EVAL_DIR / "wer"))
    p.add_argument("--snrs", nargs="+", type=float, default=[0, 5, 10, 20])
    p.add_argument("--backends", nargs="+", default=["deepfilternet", "spectral"])
    p.add_argument("--save-mixtures", action="store_true",
                   help="also write noisy + denoised wavs to eval/results/denoise_audio/ to listen to")
    args = p.parse_args()

    clips = list_audio(args.clean_dir)
    if not clips:
        raise SystemExit(f"No clean clips in {args.clean_dir} — see data/README.md.")
    noises = load_noises(SR)

    rows = []
    for ci, clip in enumerate(clips):
        clean = load_audio(clip, SR)
        noise = noises[ci % len(noises)]
        for snr in args.snrs:
            noisy = mix_at_snr(clean, noise, snr)
            base = {"clip": clip.name, "snr_in": snr, "snr_noisy": snr_db(clean, noisy),
                    "sisdr_noisy": si_sdr(clean, noisy), "stoi_noisy": _stoi(clean, noisy),
                    "pesq_noisy": _pesq(clean, noisy)}
            for backend in args.backends:
                out, out_sr = denoise(noisy, SR, backend)
                out = resample(out, out_sr, SR)
                row = {**base, "backend": backend,
                       "snr_out": snr_db(clean, out), "sisdr_out": si_sdr(clean, out),
                       "stoi_out": _stoi(clean, out), "pesq_out": _pesq(clean, out)}
                row["snr_gain"] = row["snr_out"] - row["snr_noisy"]
                row["sisdr_gain"] = row["sisdr_out"] - row["sisdr_noisy"]
                rows.append(row)
                if args.save_mixtures:
                    from src.audio_io import save_wav
                    d = RESULTS_DIR / "denoise_audio"
                    d.mkdir(parents=True, exist_ok=True)
                    save_wav(d / f"{clip.stem}_snr{snr:g}_noisy.wav", noisy, SR)
                    save_wav(d / f"{clip.stem}_snr{snr:g}_{backend}.wav", out, SR)
            print(f"  {clip.name} @ {snr:g} dB done")

    summary = []
    for backend in args.backends:
        for snr in args.snrs:
            rs = [r for r in rows if r["backend"] == backend and r["snr_in"] == snr]
            agg = {"backend": backend, "snr_in": snr}
            for k in ("snr_gain", "sisdr_gain", "stoi_noisy", "stoi_out", "pesq_noisy", "pesq_out"):
                vals = [r[k] for r in rs if r[k] is not None]
                agg[k] = float(np.mean(vals)) if vals else None
            summary.append(agg)

    print()
    print_table(summary, ["backend", "snr_in", "snr_gain", "sisdr_gain", "stoi_noisy",
                          "stoi_out", "pesq_noisy", "pesq_out"])
    save_results("denoise_quality", {"summary": summary, "rows": rows})

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for backend in args.backends:
        s = [r for r in summary if r["backend"] == backend]
        axes[0].plot([r["snr_in"] for r in s], [r["sisdr_gain"] for r in s], "o-", label=backend)
        axes[1].plot([r["snr_in"] for r in s], [r["stoi_out"] for r in s], "o-", label=backend)
    noisy = [r for r in summary if r["backend"] == args.backends[0]]
    axes[1].plot([r["snr_in"] for r in noisy], [r["stoi_noisy"] for r in noisy], "k--",
                 label="noisy (no denoise)")
    axes[0].set(xlabel="input SNR (dB)", ylabel="SI-SDR improvement (dB)", title="Denoising gain")
    axes[1].set(xlabel="input SNR (dB)", ylabel="STOI", title="Intelligibility")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "denoise_quality.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'denoise_quality.png'}")


if __name__ == "__main__":
    main()
