"""Helper for Phase 8 data collection: record held-out clips for the EER
evaluation without touching enrollment data.

Usage:
    python -m src.collect_eval_clips --name Alex --n 20           # genuine clips
    python -m src.collect_eval_clips --name _impostors --n 10     # impostor clips

Saves to data/eval/<name>/clip_XX.wav — exactly the layout eval/eer.py expects.
"""
import argparse

from .audio_io import record_fixed, save_wav
from .config import CFG, ROOT
from .vad import trim_silence, net_speech_seconds

# Different sentences than enroll.py's PROMPTS, so eval genuinely holds out
# unseen content rather than re-testing on enrollment phrases.
EVAL_PROMPTS = [
    "Cats and dogs each hate the other.",
    "The pipe began to rust while new.",
    "The lazy cow lay in the cool grass.",
    "Open the crate but don't break the glass.",
    "Add the sum to the product of these three.",
    "Thieves who rob friends deserve jail.",
    "The ripe taste of cheese improves with age.",
    "Act on these orders with great speed.",
    "The hog crawled under the high fence.",
    "Move the vat over the hot fire.",
    "The store was jammed before the sale could start.",
    "The soft cushion broke the man's fall.",
    "The salt breeze came across the sea.",
    "The girl at the booth sold fifty bonds.",
    "The small pup gnawed a hole in the sock.",
    "The fish twisted and turned on the bent hook.",
    "Press the pants and sew a button on the vest.",
    "The swan dive was far short of perfect.",
    "The beauty of the view stunned the young boy.",
    "Two blue fish swam in the tank.",
]


def collect(name: str, n: int, seconds: float):
    out_dir = ROOT / CFG["paths"]["eval_dir"] / name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob("clip_*.wav"))
    start_idx = len(existing)

    print(f"Collecting {n} held-out clips for '{name}' into {out_dir}")
    print("(Use different phrasing/pacing than enrollment if possible.)\n")

    kept = 0
    i = start_idx
    while kept < n:
        prompt = EVAL_PROMPTS[i % len(EVAL_PROMPTS)]
        input(f"[{kept+1}/{n}] Say: \"{prompt}\"  (Enter to record)")
        clip = record_fixed(seconds)
        clip = trim_silence(clip)
        net_speech = net_speech_seconds(clip)
        if net_speech < 1.0:
            print(f"  -> too little speech ({net_speech:.2f}s), retrying")
            continue
        save_wav(out_dir / f"clip_{i:02d}.wav", clip)
        print(f"  -> saved ({net_speech:.2f}s net speech)")
        kept += 1
        i += 1

    print(f"\nDone. {kept} clips saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True,
                         help="Enrolled user name for genuine clips, or "
                              "'_impostors' for non-enrolled speaker clips")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()
    collect(args.name, args.n, args.seconds)
