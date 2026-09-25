"""Convert an Audacity label export to RTTM.

How to hand-label a recording in Audacity:
  1. Open the recording, add a label track (Edit > Labels > Add Label at Selection).
  2. Select each stretch of speech and label it with the speaker's name
     (use the exact enrolled name for enrolled people, e.g. `Alex`; anything
     else, e.g. `guest1`, for people who aren't enrolled).
  3. File > Export > Export Labels... -> meeting1.txt

Then:
    python -m eval.labels_to_rttm data/eval/diarization/meeting1.txt

(eval/der.py and eval/speaker_id.py also read the .txt directly — this is
just for when you want a standard .rttm file.)
"""
import argparse
from pathlib import Path

from src.segments import read_audacity_labels, write_rttm

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("labels", help="Audacity label export (.txt)")
    p.add_argument("--out", help="output .rttm (default: same name, .rttm)")
    args = p.parse_args()

    src = Path(args.labels)
    segs = read_audacity_labels(src)
    out = Path(args.out) if args.out else src.with_suffix(".rttm")
    write_rttm(segs, out, uri=src.stem)
    print(f"{len(segs)} segments, speakers {sorted({s.speaker for s in segs})} -> {out}")
