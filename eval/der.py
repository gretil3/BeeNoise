"""Metric 2 — Diarization Error Rate (DER).

    DER = (missed speech + false alarm + speaker confusion) / total reference speech

computed with pyannote.metrics, which finds the best one-to-one mapping
between predicted labels (SPEAKER_00...) and reference names first, so label
names don't have to match. A 0.25 s forgiveness collar around each reference
boundary is standard (hand-labelled boundaries are never exact).

Needs, in data/eval/diarization/:
    meeting1.mp4 (or .wav ...)   a held-out multi-speaker recording
    meeting1.rttm                its hand-labelled ground truth
                                 (or meeting1.txt exported from Audacity labels)

Run (ablation: with vs without denoising, both diarizers):
    python -m eval.der
    python -m eval.der --backends none deepfilternet --diarizers pyannote ecapa_cluster
"""
import argparse

from src.audio_io import load_audio, resample
from src.config import CFG
from src.denoise import denoise
from src.diarize import diarize
from src.segments import Segment, read_reference

from .common import EVAL_DIR, list_audio, print_table, save_results

DIA_DIR = EVAL_DIR / "diarization"


def to_annotation(segments: list[Segment]):
    from pyannote.core import Annotation
    from pyannote.core import Segment as PSeg
    ann = Annotation()
    for i, s in enumerate(segments):
        ann[PSeg(s.start, s.end), i] = s.speaker
    return ann


def find_reference(audio_path):
    for ext in (".rttm", ".txt"):
        ref = audio_path.with_suffix(ext)
        if ref.exists():
            return ref
    return None


def main():
    from pyannote.metrics.diarization import DiarizationErrorRate

    p = argparse.ArgumentParser()
    p.add_argument("--backends", nargs="+", default=["none", CFG["denoise"]["backend"]],
                   help="denoisers to compare (none = no denoising)")
    p.add_argument("--diarizers", nargs="+", default=[CFG["diarize"]["backend"]])
    p.add_argument("--collar", type=float, default=0.25)
    p.add_argument("--oracle-num-speakers", action="store_true",
                   help="tell the diarizer the true number of speakers")
    args = p.parse_args()

    files = [f for f in list_audio(DIA_DIR) if find_reference(f)]
    if not files:
        raise SystemExit(f"No recording+reference pairs in {DIA_DIR} — see data/README.md.")

    sr_out, sr = CFG["audio"]["output_sr"], CFG["audio"]["analysis_sr"]
    rows = []
    for f in files:
        ref = read_reference(find_reference(f))
        n_ref = len({s.speaker for s in ref})
        raw = load_audio(f, sr_out)
        for backend in args.backends:
            clean, csr = denoise(raw, sr_out, backend)
            wav16 = resample(clean, csr, sr)
            for diarizer in args.diarizers:
                hyp = diarize(wav16, sr, diarizer,
                              num_speakers=n_ref if args.oracle_num_speakers else None)
                metric = DiarizationErrorRate(collar=args.collar, skip_overlap=False)
                comp = metric(to_annotation(ref), to_annotation(hyp), detailed=True)
                total = comp["total"] or 1.0
                rows.append({
                    "file": f.name, "denoise": backend, "diarizer": diarizer,
                    "DER": comp["diarization error rate"],
                    "missed": comp["missed detection"] / total,
                    "false_alarm": comp["false alarm"] / total,
                    "confusion": comp["confusion"] / total,
                    "ref_speakers": n_ref, "hyp_speakers": len({s.speaker for s in hyp}),
                })
                print(f"  {f.name} | denoise={backend} | {diarizer}: DER={rows[-1]['DER']:.3f}")

    print()
    print_table(rows, ["file", "denoise", "diarizer", "DER", "missed", "false_alarm",
                       "confusion", "ref_speakers", "hyp_speakers"])
    save_results("der", {"collar": args.collar, "rows": rows})


if __name__ == "__main__":
    main()
