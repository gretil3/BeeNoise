"""Full pipeline: python -m src.main path/to/recording.mp4

    input video/audio
      [1] denoise      -> cleaned audio track
      [2] diarize      -> "who spoke when" (anonymous labels)
      [3] speaker ID   -> labels -> enrolled names or "Speaker N"
      [4] transcribe   -> words with timestamps (Whisper)
      [5] merge        -> subtitles (.srt/.vtt), optionally put back on the video

Outputs land in outputs/<input-name>/:
    denoised.wav        cleaned audio (48 kHz)
    diarization.rttm    who-spoke-when with final names (compare with eval/der.py)
    subtitles.srt/.vtt  speaker-labelled subtitles
    transcript.json     cues, words, speaker matches + scores, timings
    <name>_subtitled.mp4  (video inputs only) denoised audio + subtitle track

Every stage can be swapped from the CLI, which is how the ablations are run:
    python -m src.main clip.mp4 --denoiser none        # skip stage 1
    python -m src.main clip.mp4 --diarizer ecapa_cluster --num-speakers 2
"""
import argparse
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .audio_io import has_video, load_audio, resample, save_wav
from .config import CFG, ROOT
from .segments import Cue, Segment, Word, write_rttm


@dataclass
class Options:
    denoiser: str = CFG["denoise"]["backend"]
    diarizer: str = CFG["diarize"]["backend"]
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    strategy: str = CFG["stt"]["strategy"]
    language: str | None = None
    diarize_on: str = CFG["pipeline"]["diarize_on"]
    speaker_id_on: str = CFG["pipeline"]["speaker_id_on"]
    transcribe_on: str = CFG["pipeline"]["transcribe_on"]


@dataclass
class Result:
    denoised: np.ndarray                 # at output_sr
    segments: list[Segment]              # diarization, final display names
    matches: dict                        # label -> speaker_id.Match
    words: list[Word]
    cues: list[Cue]
    language: str
    timings: dict = field(default_factory=dict)


def run(raw: np.ndarray, sr: int, opts: Options | None = None, log=print) -> Result:
    """Run stages 1-5 on an in-memory waveform (`sr` should be output_sr)."""
    from .denoise import denoise
    from .diarize import diarize
    from .merge import assign_speakers, build_cues
    from .speaker_id import identify_clusters, relabel
    from .stt import transcribe_turns, transcribe_words

    opts = opts or Options()
    a_sr = CFG["audio"]["analysis_sr"]
    timings = {}

    def timed(name):
        timings[name] = time.time()
        log(f"[{len(timings)}/5] {name}...")

    timed("denoise")
    clean, clean_sr = denoise(raw, sr, opts.denoiser)
    clean = resample(clean, clean_sr, sr)
    tracks = {"raw": resample(raw, sr, a_sr), "denoised": resample(clean, sr, a_sr)}

    timed("diarize")
    segs = diarize(tracks[opts.diarize_on], a_sr, opts.diarizer,
                   opts.num_speakers, opts.min_speakers, opts.max_speakers)
    log(f"      {len(segs)} turns, {len({s.speaker for s in segs})} speakers")

    timed("speaker ID")
    matches = identify_clusters(tracks[opts.speaker_id_on], a_sr, segs)
    for m in matches.values():
        score = f"{m.score:.3f}" if m.score is not None else "n/a"
        log(f"      {m.label} -> {m.name:<14} (best={m.best_candidate}, cos={score}, "
            f"{m.speech_sec:.1f}s speech)")
    named = relabel(segs, matches)

    timed("transcribe")
    if opts.strategy == "segment":
        words, lang = transcribe_turns(tracks[opts.transcribe_on], a_sr, named, opts.language)
    else:
        words, lang = transcribe_words(tracks[opts.transcribe_on], opts.language)

    timed("merge")
    cues = build_cues(assign_speakers(words, named))

    now = time.time()
    names = list(timings)
    durations = {n: round((timings[names[i + 1]] if i + 1 < len(names) else now) - timings[n], 2)
                 for i, n in enumerate(names)}
    return Result(clean, named, matches, words, cues, lang, durations)


def run_file(input_path, out_dir=None, opts: Options | None = None, burn: bool = False,
             log=print) -> dict[str, Path]:
    """Load a file, run the pipeline, write every output. Returns output paths."""
    from .merge import write_outputs
    from .video import mux

    input_path = Path(input_path)
    opts = opts or Options()
    out_dir = Path(out_dir) if out_dir else ROOT / CFG["paths"]["output_dir"] / input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    sr = CFG["audio"]["output_sr"]
    log(f"Loading {input_path.name}...")
    raw = load_audio(input_path, sr)
    log(f"      {len(raw) / sr:.1f}s of audio")

    res = run(raw, sr, opts, log)

    paths = {"denoised": out_dir / "denoised.wav", "rttm": out_dir / "diarization.rttm"}
    save_wav(paths["denoised"], res.denoised, sr)
    write_rttm(res.segments, paths["rttm"], input_path.stem)
    paths.update(write_outputs(res.cues, out_dir, extra={
        "input": str(input_path),
        "language": res.language,
        "options": asdict(opts),
        "speakers": {k: asdict(m) for k, m in res.matches.items()},
        "timings_sec": res.timings,
    }))

    if has_video(input_path):
        log("Writing video...")
        paths["video"] = mux(input_path, paths["denoised"], paths["srt"],
                             out_dir / f"{input_path.stem}_subtitled.mp4", burn=burn)

    total = sum(res.timings.values())
    log(f"\nDone in {total:.1f}s ({total / (len(raw) / sr):.2f}x real time) -> {out_dir}")
    log("  " + ", ".join(f"{k} {v:.1f}s" for k, v in res.timings.items()))
    for k, p in paths.items():
        log(f"  {k:<9} {p}")
    return paths


def main():
    p = argparse.ArgumentParser(description="Denoise, diarize, identify and subtitle a recording.")
    p.add_argument("input")
    p.add_argument("--out", help="output folder (default: outputs/<input name>/)")
    p.add_argument("--denoiser", choices=["deepfilternet", "spectral", "none"])
    p.add_argument("--diarizer", choices=["pyannote", "ecapa_cluster"])
    p.add_argument("--num-speakers", type=int)
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--strategy", choices=["full", "segment"])
    p.add_argument("--language", help="e.g. en, id (default: config / auto-detect)")
    p.add_argument("--burn", action="store_true", help="burn subtitles into the video picture")
    args = p.parse_args()

    opts = Options()
    for k in ("denoiser", "diarizer", "num_speakers", "min_speakers", "max_speakers",
              "strategy", "language"):
        if getattr(args, k) is not None:
            setattr(opts, k, getattr(args, k))
    run_file(args.input, args.out, opts, burn=args.burn)


if __name__ == "__main__":
    main()
