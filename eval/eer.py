"""Speaker verification evaluation: EER, DET-style curve, score histogram.

Expected layout for held-out data:
    data/eval/<enrolled_user_name>/*.wav   # genuine held-out utterances
    data/eval/_impostors/*.wav              # utterances from NON-enrolled speakers

Run from the project root:
    python -m eval.eer
    python -m eval.eer --min_utts 1 3 5 10      (ablation over enrollment count)
    python -m eval.eer --durations 1 2 4 8      (ablation over utterance duration)

Outputs figures + a JSON summary into eval/results/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import profiles
from src.audio_io import load_wav, SR
from src.config import ROOT
from src.encoder import embed, cosine
from src.vad import trim_silence

RESULTS_DIR = ROOT / "eval" / "results"


def _collect_eval_clips():
    """Returns {user_name: [wav_path, ...]}, plus a separate list for impostors."""
    eval_dir = ROOT / "data" / "eval"
    genuine = {}
    for d in eval_dir.iterdir():
        if not d.is_dir() or d.name == "_impostors":
            continue
        wavs = sorted(d.glob("*.wav"))
        if wavs:
            genuine[d.name] = wavs
    impostor_dir = eval_dir / "_impostors"
    impostors = sorted(impostor_dir.glob("*.wav")) if impostor_dir.exists() else []
    return genuine, impostors


def compute_scores(genuine_clips, impostor_clips, users, trim_to_seconds=None):
    """Returns (target_scores, nontarget_scores) as numpy arrays."""
    centroids = {u.name: u.centroid for u in users}
    target_scores, nontarget_scores = [], []

    for user_name, wavs in genuine_clips.items():
        if user_name not in centroids:
            continue
        for wav_path in wavs:
            audio = load_wav(wav_path)
            audio = trim_silence(audio)
            if trim_to_seconds:
                n = int(trim_to_seconds * SR)
                audio = audio[:n]
            if len(audio) < SR * 0.5:
                continue
            e = embed(audio)
            for name, c in centroids.items():
                s = cosine(e, c)
                if name == user_name:
                    target_scores.append(s)
                else:
                    nontarget_scores.append(s)

    for wav_path in impostor_clips:
        audio = load_wav(wav_path)
        audio = trim_silence(audio)
        if trim_to_seconds:
            n = int(trim_to_seconds * SR)
            audio = audio[:n]
        if len(audio) < SR * 0.5:
            continue
        e = embed(audio)
        for name, c in centroids.items():
            nontarget_scores.append(cosine(e, c))

    return np.array(target_scores), np.array(nontarget_scores)


def compute_eer(target_scores: np.ndarray, nontarget_scores: np.ndarray):
    """Sweeps thresholds, returns (eer, tau_at_eer, thresholds, far, frr)."""
    thresholds = np.linspace(-1.0, 1.0, 401)
    far = np.array([(nontarget_scores >= t).mean() if len(nontarget_scores) else np.nan
                     for t in thresholds])
    frr = np.array([(target_scores < t).mean() if len(target_scores) else np.nan
                     for t in thresholds])

    diff = np.abs(far - frr)
    idx = int(np.nanargmin(diff))
    eer = float((far[idx] + frr[idx]) / 2)
    tau = float(thresholds[idx])
    return eer, tau, thresholds, far, frr


def plot_det(thresholds, far, frr, eer, tau, out_path):
    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, far * 100, label="FAR (false accept)")
    plt.plot(thresholds, frr * 100, label="FRR (false reject)")
    plt.axvline(tau, color="gray", linestyle="--", label=f"EER threshold = {tau:.3f}")
    plt.scatter([tau], [eer * 100], color="red", zorder=5, label=f"EER = {eer*100:.2f}%")
    plt.xlabel("Cosine similarity threshold (tau)")
    plt.ylabel("Error rate (%)")
    plt.title("Speaker Verification: FAR / FRR vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_histogram(target_scores, nontarget_scores, tau, out_path):
    plt.figure(figsize=(6, 5))
    plt.hist(target_scores, bins=40, alpha=0.6, label="target (same speaker)", density=True)
    plt.hist(nontarget_scores, bins=40, alpha=0.6, label="non-target (different speaker)", density=True)
    plt.axvline(tau, color="red", linestyle="--", label=f"chosen tau = {tau:.3f}")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title("Score Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--durations", nargs="*", type=float, default=None,
                         help="Ablation: truncate each eval clip to these durations (sec)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    users = profiles.get_all_users()
    if not users:
        print("No enrolled users found. Run `python -m src.enroll --name <name>` first.")
        return

    genuine, impostors = _collect_eval_clips()
    if not genuine:
        print(f"No held-out eval clips found under data/eval/<user>/*.wav.")
        print("Record ~20 held-out utterances per enrolled user (and some in "
              "data/eval/_impostors/ from non-enrolled speakers) before running this.")
        return

    print(f"Users: {[u.name for u in users]}")
    print(f"Genuine eval clips: { {k: len(v) for k, v in genuine.items()} }")
    print(f"Impostor clips: {len(impostors)}")

    if args.durations:
        summary = []
        for dur in args.durations:
            t, n = compute_scores(genuine, impostors, users, trim_to_seconds=dur)
            eer, tau, *_ = compute_eer(t, n)
            print(f"  duration={dur:.1f}s -> EER={eer*100:.2f}%  tau={tau:.3f}  "
                  f"(n_target={len(t)}, n_nontarget={len(n)})")
            summary.append({"duration_sec": dur, "eer": eer, "tau": tau})
        with open(RESULTS_DIR / "eer_vs_duration.json", "w") as f:
            json.dump(summary, f, indent=2)
        durs = [s["duration_sec"] for s in summary]
        eers = [s["eer"] * 100 for s in summary]
        plt.figure(figsize=(6, 4))
        plt.plot(durs, eers, marker="o")
        plt.xlabel("Utterance duration (s)")
        plt.ylabel("EER (%)")
        plt.title("EER vs. Utterance Duration")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "eer_vs_duration.png", dpi=150)
        print(f"\nSaved ablation results to {RESULTS_DIR}")
        return

    target_scores, nontarget_scores = compute_scores(genuine, impostors, users)
    eer, tau, thresholds, far, frr = compute_eer(target_scores, nontarget_scores)

    print(f"\n=== RESULTS ===")
    print(f"EER = {eer*100:.2f}%  at tau = {tau:.3f}")
    print(f"n_target={len(target_scores)}  n_nontarget={len(nontarget_scores)}")
    print(f"\nUpdate config.yaml -> speaker.tau to {tau:.3f} to use this threshold.")

    plot_det(thresholds, far, frr, eer, tau, RESULTS_DIR / "det_curve.png")
    plot_histogram(target_scores, nontarget_scores, tau, RESULTS_DIR / "score_histogram.png")

    with open(RESULTS_DIR / "eer_summary.json", "w") as f:
        json.dump({
            "eer": eer, "tau": tau,
            "n_target": len(target_scores), "n_nontarget": len(nontarget_scores),
            "users": [u.name for u in users],
        }, f, indent=2)

    print(f"Figures + summary saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
