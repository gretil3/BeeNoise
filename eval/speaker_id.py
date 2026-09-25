"""Metric 3 — speaker ID accuracy + false accepts.

Uses the same labelled recordings as eval/der.py. In the reference labels,
use the enrolled name for enrolled people (exactly as enrolled, spaces -> _)
and anything else (e.g. `guest1`) for people who are NOT enrolled.

    accuracy      % of enrolled-speaker segments given the right name
    false_accept  % of unenrolled-speaker segments given an enrolled name
                  (the dangerous error — a stranger labelled as Alex)

Two modes:
    --mode oracle     (default) score each *reference* segment on its own. This
                      isolates stage 3 from diarization mistakes and gives a
                      clean tau sweep -> recommended speaker.tau.
    --mode pipeline   run stages 2+3 for real and score each predicted segment
                      against the reference speaker it overlaps most.

Run (with vs without denoising — blueprint limitation #3):
    python -m eval.speaker_id
    python -m eval.speaker_id --mode pipeline
"""
import argparse

import numpy as np

from src import profiles as profiles_db
from src.audio_io import cut, load_audio, resample
from src.config import CFG
from src.denoise import denoise
from src.segments import overlap, read_reference

from .common import RESULTS_DIR, list_audio, print_table, save_results
from .der import DIA_DIR, find_reference


def _score(pairs, enrolled: set[str]):
    """pairs: [(true_name, predicted_name_or_None)] -> (accuracy, false_accept, n_enr, n_unenr)."""
    enr = [(t, p) for t, p in pairs if t in enrolled]
    unenr = [(t, p) for t, p in pairs if t not in enrolled]
    acc = float(np.mean([p == t for t, p in enr])) if enr else None
    fa = float(np.mean([p in enrolled for _, p in unenr])) if unenr else None
    return acc, fa, len(enr), len(unenr)


def main():
    from src.speaker_id import identify_clip, identify_clusters

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["oracle", "pipeline"], default="oracle")
    p.add_argument("--backends", nargs="+", default=["none", CFG["denoise"]["backend"]])
    args = p.parse_args()

    profiles = profiles_db.get_all()
    if not profiles:
        raise SystemExit("Nobody is enrolled. Run python -m src.enroll first.")
    enrolled = {pr.name.replace(" ", "_") for pr in profiles}
    for pr in profiles:
        pr.name = pr.name.replace(" ", "_")  # match RTTM label convention
    files = [f for f in list_audio(DIA_DIR) if find_reference(f)]
    if not files:
        raise SystemExit(f"No recording+reference pairs in {DIA_DIR} — see data/README.md.")

    sr_out, sr = CFG["audio"]["output_sr"], CFG["audio"]["analysis_sr"]
    min_sec = CFG["speaker"]["min_segment_sec"]
    summary, sweeps = [], {}
    for backend in args.backends:
        pairs, scored = [], []  # scored: (true_name, best_name, best_score) for the tau sweep
        for f in files:
            ref = read_reference(find_reference(f))
            clean, csr = denoise(load_audio(f, sr_out), sr_out, backend)
            wav16 = resample(clean, csr, sr)

            if args.mode == "oracle":
                for s in ref:
                    if s.duration < min_sec:
                        continue
                    pred, best, score = identify_clip(cut(wav16, sr, s.start, s.end), profiles)
                    pairs.append((s.speaker, pred))
                    scored.append((s.speaker, best, score))
            else:
                from src.diarize import diarize
                hyp = diarize(wav16, sr)
                matches = identify_clusters(wav16, sr, hyp, profiles)
                for h in hyp:
                    if h.duration < min_sec:
                        continue
                    true = max(ref, key=lambda r: overlap(h.start, h.end, r.start, r.end))
                    if overlap(h.start, h.end, true.start, true.end) == 0:
                        continue  # false-alarm segment: a DER problem, not an ID one
                    m = matches[h.speaker]
                    pairs.append((true.speaker, m.name if m.enrolled else None))

        acc, fa, n_enr, n_unenr = _score(pairs, enrolled)
        summary.append({"denoise": backend, "mode": args.mode, "accuracy": acc,
                        "false_accept": fa, "enrolled_segs": n_enr, "unenrolled_segs": n_unenr,
                        "tau": CFG["speaker"]["tau"]})
        sweeps[backend] = scored

    print()
    print_table(summary, ["denoise", "mode", "accuracy", "false_accept", "enrolled_segs",
                          "unenrolled_segs", "tau"])

    payload = {"summary": summary}
    if args.mode == "oracle":
        import matplotlib.pyplot as plt
        taus = np.round(np.arange(0.0, 0.81, 0.02), 2)
        fig, ax = plt.subplots(figsize=(6, 4))
        payload["sweep"] = {}
        for backend, scored in sweeps.items():
            curve = []
            for tau in taus:
                pr = [(t, b if s >= tau else None) for t, b, s in scored]
                acc, fa, _, _ = _score(pr, enrolled)
                curve.append({"tau": float(tau), "accuracy": acc, "false_accept": fa})
            payload["sweep"][backend] = curve
            ax.plot(taus, [c["accuracy"] or 0 for c in curve], "-", label=f"accuracy ({backend})")
            if any(c["false_accept"] is not None for c in curve):
                ax.plot(taus, [c["false_accept"] or 0 for c in curve], "--",
                        label=f"false accept ({backend})")
                # Recommend the tau that maximises accuracy - false_accept.
                best = max(curve, key=lambda c: (c["accuracy"] or 0) - (c["false_accept"] or 0))
                print(f"Recommended speaker.tau for denoise={backend}: {best['tau']:.2f} "
                      f"(accuracy={best['accuracy']:.3f}, false_accept={best['false_accept']:.3f})")
        ax.axvline(CFG["speaker"]["tau"], color="gray", lw=0.8, label="current tau")
        ax.set(xlabel="tau (cosine threshold)", ylabel="rate", title="Speaker ID vs threshold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(RESULTS_DIR / "speaker_id_tau_sweep.png", dpi=150)
        print(f"Saved {RESULTS_DIR / 'speaker_id_tau_sweep.png'}")
    save_results(f"speaker_id_{args.mode}", payload)


if __name__ == "__main__":
    main()
