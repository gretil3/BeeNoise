"""Enrollment CLI: python -m src.enroll --name Alex

Records several varied phrases, embeds each, rejects outliers (coughs,
false starts), and stores the centroid as the user's voiceprint.
"""
import argparse

import numpy as np

from . import profiles
from .audio_io import record_fixed, save_wav
from .config import CFG, ROOT
from .encoder import embed, cosine
from .vad import trim_silence, net_speech_seconds

# Harvard-sentence-style prompts: short, phonetically varied, nothing repeated.
PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Please call Stella and ask her to bring these things.",
    "A large size in stockings is hard to sell.",
    "The rainbow is a division of white light into many beautiful colors.",
    "We need a small plastic snake and a big toy frog for the kids.",
    "It's easy to tell the depth of a well by its echo.",
    "Four hooded figures left quickly before the storm.",
    "Smoky fires lack flame and heat but produce a lot of smoke.",
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
]


def enroll(name: str, student_id: str | None = None):
    n = CFG["enroll"]["n_utterances"]
    dur = CFG["enroll"]["utt_seconds"]
    reject_cos = CFG["enroll"]["reject_cosine"]

    out_dir = ROOT / CFG["paths"]["enroll_dir"] / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Enrolling '{name}' — {n} phrases, {dur:.0f}s each.")
    print("Speak naturally, at a normal pace. Press Enter to start each one.\n")

    embeddings = []
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        input(f"[{i+1}/{n}] Say: \"{prompt}\"  (Enter to record)")
        clip = record_fixed(dur)
        clip = trim_silence(clip)

        net_speech = net_speech_seconds(clip)
        if net_speech < 1.0:
            print(f"  -> too little speech detected ({net_speech:.2f}s), retrying this one")
            continue

        e = embed(clip)
        running_mean = np.mean(embeddings, axis=0) if embeddings else None
        if running_mean is not None:
            sim = cosine(e / np.linalg.norm(e), running_mean / np.linalg.norm(running_mean))
            if sim < reject_cos:
                print(f"  -> outlier detected (cosine {sim:.2f} vs running mean), discarding")
                continue

        save_wav(out_dir / f"utt_{i:02d}.wav", clip)
        embeddings.append(e)
        print(f"  -> kept ({net_speech:.2f}s net speech)")

    if len(embeddings) < max(3, n // 2):
        print(f"\nOnly {len(embeddings)} good utterances captured — enrollment aborted. "
              f"Try again in a quieter environment.")
        return

    embeddings = np.stack(embeddings)
    centroid = embeddings.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    # Intra-speaker spread: mean cosine of each utterance to the centroid.
    spread = float(np.mean([cosine(e / np.linalg.norm(e), centroid) for e in embeddings]))

    existing = profiles.get_user_by_name(name)
    if existing:
        profiles.update_user_embedding(existing.id, centroid, len(embeddings), spread)
        print(f"\nUpdated existing profile for '{name}' "
              f"({len(embeddings)} utterances, spread={spread:.3f}).")
    else:
        profiles.add_user(name, centroid, len(embeddings), spread, student_id=student_id)
        print(f"\nEnrolled '{name}' with {len(embeddings)} utterances "
              f"(intra-speaker spread={spread:.3f}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Student name to enroll")
    parser.add_argument("--student-id", default=None, help="Optional student ID / NIM")
    args = parser.parse_args()
    enroll(args.name, args.student_id)
