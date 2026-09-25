"""Local web demo: python -m src.demo_ui   (opens http://127.0.0.1:7860)

Tabs:
  Transcribe — upload a video/audio file, run the full pipeline, get the
               denoised audio, subtitled video and subtitle files.
  Enroll     — record or upload a paragraph reading to add a speaker.
  Speakers   — who is enrolled.
"""
import tempfile
from pathlib import Path

import gradio as gr

from . import profiles
from .config import CFG
from .enroll import PARAGRAPHS, enroll_audio
from .main import Options, run_file


def _transcribe(file, denoiser, diarizer, num_speakers, strategy, burn):
    if not file:
        raise gr.Error("Upload a file first.")
    logs = []
    opts = Options(denoiser=denoiser, diarizer=diarizer, strategy=strategy,
                   num_speakers=int(num_speakers) if num_speakers else None)
    out_dir = Path(tempfile.mkdtemp(prefix="beenoise_"))
    try:
        paths = run_file(file, out_dir, opts, burn=burn, log=logs.append)
    except Exception as e:
        raise gr.Error(f"{type(e).__name__}: {e}") from e
    srt_text = paths["srt"].read_text(encoding="utf-8")
    return (str(paths["denoised"]), str(paths["video"]) if "video" in paths else None,
            srt_text, [str(paths[k]) for k in ("srt", "vtt", "json", "rttm")], "\n".join(logs))


def _enroll(name, audio_path, denoise):
    if not name or not audio_path:
        raise gr.Error("Enter a name and record/upload audio.")
    from .audio_io import load_audio
    sr = CFG["audio"]["analysis_sr"]
    try:
        prof = enroll_audio(name.strip(), load_audio(audio_path, sr), sr, denoise=denoise)
    except ValueError as e:
        raise gr.Error(str(e)) from e
    return (f"Enrolled {prof.name}: {prof.speech_sec:.1f}s speech, {prof.n_chunks} chunks, "
            f"spread={prof.spread:.3f}")


def _speakers():
    return [[p.name, round(p.speech_sec, 1), p.n_chunks, round(p.spread, 3), p.created_at[:19]]
            for p in profiles.get_all()]


with gr.Blocks(title="BeeNoise") as demo:
    gr.Markdown("# BeeNoise — denoised, speaker-labelled subtitles")
    with gr.Tab("Transcribe"):
        with gr.Row():
            with gr.Column():
                inp = gr.File(label="Video or audio file", type="filepath")
                denoiser = gr.Radio(["deepfilternet", "spectral", "none"],
                                    value=CFG["denoise"]["backend"], label="Denoiser")
                diarizer = gr.Radio(["pyannote", "ecapa_cluster"],
                                    value=CFG["diarize"]["backend"], label="Diarizer")
                n_spk = gr.Number(label="Number of speakers (blank = auto)", precision=0)
                strategy = gr.Radio(["full", "segment"], value=CFG["stt"]["strategy"],
                                    label="Transcription strategy")
                burn = gr.Checkbox(label="Burn subtitles into video")
                go = gr.Button("Run pipeline", variant="primary")
            with gr.Column():
                out_audio = gr.Audio(label="Denoised audio", type="filepath")
                out_video = gr.Video(label="Subtitled video")
                out_srt = gr.Textbox(label="Subtitles (.srt)", lines=12)
                out_files = gr.File(label="Download", file_count="multiple")
                out_log = gr.Textbox(label="Log", lines=8)
        go.click(_transcribe, [inp, denoiser, diarizer, n_spk, strategy, burn],
                 [out_audio, out_video, out_srt, out_files, out_log])

    with gr.Tab("Enroll"):
        gr.Markdown("Read this paragraph aloud (30-60 s), in a quiet room:\n\n> "
                    + PARAGRAPHS["en"] + "\n\n*(Bahasa Indonesia:)*\n\n> " + PARAGRAPHS["id"])
        name = gr.Textbox(label="Name")
        rec = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Reading")
        den = gr.Checkbox(label="Denoise before enrolling")
        enroll_btn = gr.Button("Enroll", variant="primary")
        enroll_msg = gr.Textbox(label="Result")
        enroll_btn.click(_enroll, [name, rec, den], enroll_msg)

    with gr.Tab("Speakers"):
        table = gr.Dataframe(headers=["name", "speech_sec", "chunks", "spread", "enrolled"],
                             value=_speakers)
        gr.Button("Refresh").click(_speakers, None, table)


if __name__ == "__main__":
    demo.launch()
