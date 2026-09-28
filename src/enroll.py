"""Enrollment: python -m src.enroll --name Alex

The speaker reads a paragraph aloud (~30-60 s of varied speech). We keep only
the voiced frames, cut them into 3 s chunks, embed each chunk with ECAPA-TDNN,
drop outlier chunks (coughs, a door slam), and store the mean of the rest as
the speaker's voiceprint centroid.

Text-independent on purpose: the paragraph is just a way to get varied
phonemes. Later recordings can say anything and still match.

Already have a recording? Skip the mic:
    python -m src.enroll --name Alex --file alex_reading.m4a
"""
import argparse
import re

import numpy as np

from . import profiles
from .audio_io import load_audio, record_until_enter, save_wav
from .config import CFG, ROOT
from .vad import speech_only

PARAGRAPHS = {
    "en": (
        "Every morning, the old baker opens his shop before the sun comes up. "
        "He mixes flour, water, yeast and a pinch of salt, then kneads the dough "
        "until it feels smooth and soft. While the bread rises, he sweeps the "
        "floor, checks the oven, and writes the day's prices on a small chalk "
        "board. Children on their way to school often stop to watch through the "
        "window, pointing at the cakes and the golden rolls. By seven o'clock "
        "the street smells of warm bread, and a quiet queue of neighbours has "
        "formed outside the door. Nobody seems to mind the wait. They talk about "
        "the weather, the traffic, and whether the rain will finally stop this week."
    ),
    "id": (
        "Setiap pagi, seorang tukang roti tua membuka tokonya sebelum matahari "
        "terbit. Ia mencampur tepung, air, ragi, dan sejumput garam, lalu "
        "menguleni adonan sampai terasa halus dan lembut. Sambil menunggu roti "
        "mengembang, ia menyapu lantai, memeriksa oven, dan menulis harga hari "
        "itu di papan kapur kecil. Anak-anak yang berangkat ke sekolah sering "
        "berhenti untuk melihat lewat jendela, menunjuk kue dan roti yang "
        "keemasan. Pada pukul tujuh, jalanan sudah harum oleh roti hangat, dan "
        "antrean tetangga mulai terbentuk di depan pintu. Tidak ada yang keberatan "
        "menunggu. Mereka mengobrol tentang cuaca, kemacetan, dan apakah hujan "
        "akhirnya akan berhenti minggu ini."
    ),
}


def voiceprint(speech16: np.ndarray, sr: int) -> tuple[np.ndarray, int, float]:
    """Voiced audio -> (centroid, n_chunks_kept, spread)."""
    from .encoder import embed_many

    n = int(CFG["enroll"]["chunk_sec"] * sr)
    chunks = [speech16[i:i + n] for i in range(0, len(speech16) - n + 1, n)]
    if len(chunks) < 3:
        raise ValueError(f"Only {len(chunks)} chunks of speech — need at least 3.")
    embs = embed_many(chunks)

    # First pass: median is robust to a few bad chunks. Second pass: drop
    # chunks far from it and average the rest.
    first = np.median(embs, axis=0)
    first /= np.linalg.norm(first)
    keep = embs[embs @ first >= CFG["enroll"]["reject_cosine"]]
    if len(keep) < 3:
        raise ValueError("Too many inconsistent chunks — re-record somewhere quieter.")
    centroid = keep.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    spread = float(np.mean(keep @ centroid))
    return centroid, len(keep), spread


def enroll_audio(name: str, wav16: np.ndarray, sr: int, denoise: bool = False) -> profiles.Profile:
    """Shared by the CLI and the Gradio demo."""
    # `name` becomes a folder under enroll_dir, and on a public deployment
    # anyone can type it — reject anything that could escape that folder.
    if not re.fullmatch(r"[\w .-]{1,64}", name) or name.strip(". ") != name:
        raise ValueError("Name may only contain letters, digits, spaces, '.', '-' and '_', "
                         "and can't start or end with a dot or space.")
    if denoise:
        from .audio_io import resample
        from .denoise import denoise as run_denoise
        clean, clean_sr = run_denoise(wav16, sr)
        wav16 = resample(clean, clean_sr, sr)

    speech = speech_only(wav16, sr)
    speech_sec = len(speech) / sr
    if speech_sec < CFG["enroll"]["min_speech_sec"]:
        raise ValueError(f"Only {speech_sec:.1f}s of speech detected; need at least "
                         f"{CFG['enroll']['min_speech_sec']}s. Read the whole paragraph.")
    if speech_sec < CFG["enroll"]["recommended_speech_sec"]:
        print(f"  warning: {speech_sec:.1f}s of speech is on the short side — "
              f"{CFG['enroll']['recommended_speech_sec']}s+ gives a more reliable voiceprint.")

    out_dir = ROOT / CFG["paths"]["enroll_dir"] / name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_wav(out_dir / "enrollment.wav", wav16, sr)

    centroid, n_chunks, spread = voiceprint(speech, sr)
    return profiles.upsert(name, centroid, n_chunks, spread, speech_sec)


def main():
    p = argparse.ArgumentParser(description="Enroll a speaker's voiceprint.")
    p.add_argument("--name", required=True)
    p.add_argument("--file", nargs="*", help="existing recording(s) instead of the mic")
    p.add_argument("--lang", choices=sorted(PARAGRAPHS), default="en",
                   help="language of the paragraph to read")
    p.add_argument("--denoise", action="store_true",
                   help="denoise the enrollment audio first (use if it was recorded in noise)")
    args = p.parse_args()

    sr = CFG["audio"]["analysis_sr"]
    if args.file:
        wav = np.concatenate([load_audio(f, sr) for f in args.file])
    else:
        print(f"\nEnrolling '{args.name}'. Read this paragraph aloud at a normal pace:\n")
        print("    " + PARAGRAPHS[args.lang] + "\n")
        input("Press Enter to START recording...")
        wav = record_until_enter(sr, CFG["enroll"]["max_record_sec"])

    try:
        prof = enroll_audio(args.name, wav, sr, denoise=args.denoise)
    except ValueError as e:
        print(f"\nEnrollment failed: {e}")
        raise SystemExit(1) from e
    print(f"\nEnrolled '{prof.name}': {prof.speech_sec:.1f}s speech, {prof.n_chunks} chunks, "
          f"spread={prof.spread:.3f} (closer to 1 = more consistent)")


if __name__ == "__main__":
    main()
