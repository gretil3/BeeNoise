"""Local web demo: python -m src.demo_ui   (opens http://127.0.0.1:7860)
Live-reload while editing the UI: gradio app.py

One page, top to bottom (the navbar links jump to About and Features):
  About    — hero, the demo video (assets/demo.mp4, if present) and, under it,
             the pipeline from input to output.
  Features — side by side. Enroll: read a paragraph to add a speaker (the
             enrolled list and deleting are hidden on a Hugging Face Space, so
             visitors can't see each other's names). Transcribe: upload a
             video/audio file and run the full pipeline. The speaker timeline,
             colour-coded transcript, before/after audio, subtitled video and
             subtitle files appear full width below both.

Set BEENOISE_PASSWORD to put the UI behind a login (user: beenoise). It's
required on a Space, because the URL is public and permanent.

Look: always-dark, Apple-style (black, grey cards, one amber accent), with a
fixed top navbar. The theme variables are below;
the rest of the styling is in CSS.
"""
import json
import os
import queue
import re
import shutil
import tempfile
import threading
import time
from html import escape
from pathlib import Path

import gradio as gr

from . import profiles
from .config import CFG
from .enroll import PARAGRAPHS, enroll_audio
from .main import Options, run_file

ON_SPACE = "SPACE_ID" in os.environ  # set by Hugging Face Spaces
MAX_OUTPUT_AGE_SEC = 3600
DEMO_VIDEO = Path(__file__).resolve().parent.parent / "assets" / "demo.mp4"

AMBER = "#FFB300"
# One colour per speaker, in order of appearance (Apple system colours, dark variants).
SPEAKER_COLORS = [AMBER, "#0A84FF", "#30D158", "#FF375F", "#BF5AF2", "#64D2FF", "#FF9F0A", "#5E5CE6"]


def _sweep_old_outputs():
    """Delete result folders from earlier runs. Gradio copies the outputs it
    serves into its own cache, so ours are only needed while a run is going."""
    cutoff = time.time() - MAX_OUTPUT_AGE_SEC
    for d in Path(tempfile.gettempdir()).glob("beenoise_*"):
        if d.is_dir() and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)


def _clock(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d}"


def _speaker_colors(cues: list[dict]) -> dict[str, str]:
    colors = {}
    for c in cues:
        colors.setdefault(c["speaker"], SPEAKER_COLORS[len(colors) % len(SPEAKER_COLORS)])
    return colors


def _timeline_html(cues: list[dict], duration: float, colors: dict[str, str]) -> str:
    """One lane per speaker, a bar wherever they talk. Cues stand in for the
    diarized turns, so the timeline always agrees with the transcript."""
    duration = max(duration, max((c["end"] for c in cues), default=0), 1e-6)
    lanes = []
    for spk, color in colors.items():
        bars = "".join(
            f'<i style="left:{c["start"] / duration * 100:.2f}%;'
            f'width:{max((c["end"] - c["start"]) / duration * 100, 0.4):.2f}%"'
            f' title="{_clock(c["start"])}–{_clock(c["end"])}"></i>'
            for c in cues if c["speaker"] == spk)
        lanes.append(f'<div class="lane" style="--c:{color}"><span>{escape(spk)}</span>'
                     f'<div class="track">{bars}</div></div>')
    return (f'<div class="bn-card"><div class="bn-head">Who spoke when'
            f'<em>{_clock(duration)}</em></div>{"".join(lanes)}</div>')


def _transcript_html(cues: list[dict], colors: dict[str, str]) -> str:
    rows = "".join(
        f'<div class="cue"><time>{_clock(c["start"])}</time>'
        f'<b style="--c:{colors[c["speaker"]]}">{escape(c["speaker"])}</b>'
        f'<p>{escape(c["text"])}</p></div>' for c in cues)
    return (f'<div class="bn-card"><div class="bn-head">Transcript</div>'
            f'<div class="transcript">{rows or "<p class=empty>No speech found.</p>"}</div></div>')


STAGES = ["Denoise", "Diarize", "Identify", "Transcribe", "Subtitles"]


def _status_html(stage: int, elapsed: float) -> str:
    """Loading card. stage 0 = reading the file, 1-5 = that pipeline stage is running."""
    steps = "".join(
        f'<li class="{"done" if i < stage else "now" if i == stage else ""}">{name}</li>'
        for i, name in enumerate(STAGES, 1))
    current = STAGES[stage - 1] if stage else "Reading file"
    return (f'<div class="bn-card"><div class="bn-head">{current}…<em>{_clock(elapsed)}</em></div>'
            f'<ol class="steps">{steps}</ol>'
            '<p class="bn-note">Everything runs on this computer, nothing goes to the cloud. '
            'Expect 1–3 minutes per minute of audio; the first run also loads the models.</p></div>')


def _transcribe(file, recording, denoiser, diarizer, num_speakers, strategy, burn):
    """Generator: streams the loading card while the pipeline runs, then the results."""
    file = file or recording
    if not file:
        raise gr.Error("Upload a file or record something first.")
    opts = Options(denoiser=denoiser, diarizer=diarizer, strategy=strategy,
                   num_speakers=int(num_speakers) if num_speakers else None)
    _sweep_old_outputs()
    out_dir = Path(tempfile.mkdtemp(prefix="beenoise_"))
    sr = CFG["audio"]["analysis_sr"]
    messages: queue.Queue[str] = queue.Queue()
    done: dict = {}

    def work():  # the pipeline blocks for minutes: run it in a thread and keep the card ticking
        from .audio_io import load_audio
        try:
            done["original"] = load_audio(file, sr)  # the "before" player
            done["paths"] = run_file(file, out_dir, opts, burn=burn, log=messages.put)
        except Exception as e:
            done["error"] = e

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    t0, stage, logs = time.time(), 0, []
    yield {go: gr.Button(interactive=False), results: gr.Column(visible=False),
           status: gr.HTML(_status_html(0, 0), visible=True)}
    while worker.is_alive() or not messages.empty():
        try:
            logs.append(messages.get(timeout=1))
            if m := re.match(r"\[(\d)/5\]", logs[-1]):  # stage headers from main.run
                stage = int(m[1])
        except queue.Empty:
            pass
        yield {status: _status_html(stage, time.time() - t0)}

    if "error" in done:
        yield {go: gr.Button(interactive=True), status: gr.HTML(visible=False)}
        e = done["error"]
        raise gr.Error(f"{type(e).__name__}: {e}") from e

    paths, original = done["paths"], done["original"]
    cues = json.loads(paths["json"].read_text(encoding="utf-8"))["cues"]
    colors = _speaker_colors(cues)
    has_video = "video" in paths
    yield {go: gr.Button(interactive=True), status: gr.HTML(visible=False),
           results: gr.Column(visible=True),
           timeline: _timeline_html(cues, len(original) / sr, colors),
           transcript: _transcript_html(cues, colors),
           before: (sr, original), after: str(paths["denoised"]),
           out_video: gr.Video(value=str(paths["video"]) if has_video else None,
                               visible=has_video),
           out_files: [str(paths[k]) for k in ("srt", "vtt", "json", "rttm")],
           out_log: "\n".join(logs)}


def _paragraph_html(lang: str) -> str:
    return f'<p class="bn-quote">{escape(PARAGRAPHS["id" if lang == "Bahasa" else "en"])}</p>'


def _enroll_status_html(name: str, done: bool = False) -> str:
    """Card under the Enroll button: animated waveform while working, a tick when done."""
    icon = ('<svg class="tick" viewBox="0 0 24 24" aria-hidden="true"><path fill="none" '
            'stroke="currentColor" stroke-width="2.5" stroke-linecap="round" '
            'stroke-linejoin="round" d="M5 12.5l4.5 4.5L19 7.5"/></svg>' if done else
            '<span class="wave" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></span>')
    text = f"{escape(name)} is enrolled" if done else f"Enrolling {escape(name)}…"
    return (f'<div class="bn-enroll{" done" if done else ""}" role="status">{icon}'
            f'<span>{text}</span></div>')


def _enroll(name, audio_path, denoise):
    """Generator: shows the enrolling animation while the voiceprint is computed."""
    if not name or not audio_path:
        raise gr.Error("Enter a name and record/upload audio.")
    from .audio_io import load_audio
    sr = CFG["audio"]["analysis_sr"]
    name = name.strip()
    yield {enroll_btn: gr.Button(interactive=False),
           enroll_status: gr.HTML(_enroll_status_html(name), visible=True)}
    try:
        prof = enroll_audio(name, load_audio(audio_path, sr), sr, denoise=denoise)
    except Exception as e:  # ValueError is a user problem, anything else a crash: both end the animation
        yield {enroll_btn: gr.Button(interactive=True), enroll_status: gr.HTML(visible=False)}
        raise gr.Error(str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}") from e
    gr.Info(f"Enrolled {prof.name} ({prof.speech_sec:.0f}s of speech).")
    chips, choices = _speakers()
    yield {enroll_btn: gr.Button(interactive=True),
           enroll_status: gr.HTML(_enroll_status_html(prof.name, done=True)),
           enrolled: chips, who: choices}


def _speakers():
    """Enrolled-speaker chips + a refreshed delete dropdown."""
    ps = profiles.get_all()
    chips = "".join(f'<span class="chip">{escape(p.name)}<em>{p.speech_sec:.0f}s</em></span>'
                    for p in ps)
    html = (f'<div class="bn-chips"><span class="bn-label">Enrolled</span>'
            f'{chips or "<span class=empty>No one yet</span>"}</div>')
    return html, gr.Dropdown(choices=[p.name for p in ps], value=None)


def _delete_speaker(name):
    if not name:  # nothing picked, or the browser confirm() was cancelled: leave the UI as is
        return gr.skip(), gr.skip()
    profiles.delete(name)
    gr.Info(f"Deleted {name}.")
    return _speakers()


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.zinc,
    radius_size=gr.themes.sizes.radius_lg,
    # SF Pro on Apple devices; Inter (closest free match) everywhere else.
    font=["-apple-system", "BlinkMacSystemFont",
          gr.themes.GoogleFont("Inter", weights=(400, 500, 600, 700)), "system-ui", "sans-serif"],
    font_mono=["SF Mono", gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill_dark="#000000",
    body_text_color_dark="#F5F5F7",
    body_text_color_subdued_dark="#8E8E93",
    background_fill_primary_dark="#1C1C1E",
    background_fill_secondary_dark="#2C2C2E",
    block_background_fill_dark="#1C1C1E",
    block_border_color_dark="#2C2C2E",
    block_label_background_fill_dark="#1C1C1E",
    block_label_text_color_dark="#8E8E93",
    block_title_text_color_dark="#8E8E93",
    border_color_primary_dark="#2C2C2E",
    border_color_accent_dark=AMBER,
    color_accent_soft_dark="rgba(255,179,0,0.14)",
    input_background_fill_dark="#2C2C2E",
    input_border_color_dark="#3A3A3C",
    input_border_color_focus_dark=AMBER,
    button_primary_background_fill_dark=AMBER,
    button_primary_background_fill_hover_dark="#FFC53D",
    button_primary_text_color_dark="#000000",
    button_secondary_background_fill_dark="#2C2C2E",
    button_secondary_background_fill_hover_dark="#3A3A3C",
    button_secondary_text_color_dark="#F5F5F7",
    button_cancel_background_fill_dark="#2C2C2E",
    button_cancel_background_fill_hover_dark="#3A3A3C",
    button_cancel_text_color_dark="#FF453A",  # Apple system red: destructive actions
    button_large_radius="980px",
    button_medium_radius="980px",
    button_small_radius="980px",
    checkbox_background_color_selected_dark=AMBER,
    checkbox_border_color_selected_dark=AMBER,
    checkbox_label_background_fill_selected_dark=AMBER,
    checkbox_label_text_color_selected_dark="#000000",
    slider_color_dark=AMBER,
    loader_color_dark=AMBER,
    shadow_drop="none",
    shadow_drop_lg="none",
)

CSS = """
.gradio-container { width: 100% !important; max-width: 1080px !important; margin: 0 auto !important; }
.gradio-container > main { padding-top: 0 !important; }
footer { display: none !important; }
html { scroll-behavior: smooth; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }

/* navbar: a fixed frosted strip, logo left, in-page links right */
#bn-navwrap { position: fixed !important; inset: 0 0 auto 0; margin: 0 !important;
  padding: 0 !important; z-index: 10; }  /* out of the layout flow: no gap above the hero */
#bn-nav { position: fixed; top: 0; left: 0; right: 0; height: 56px; z-index: 10;
  background: rgba(0,0,0,.9); border-bottom: 1px solid #1C1C1E;
  backdrop-filter: saturate(180%) blur(20px); -webkit-backdrop-filter: saturate(180%) blur(20px); }
#bn-nav .inner { max-width: 1080px; height: 100%; margin: 0 auto; padding: 0 32px;
  display: flex; align-items: center; justify-content: space-between; }
#bn-nav a { text-decoration: none; }
#bn-nav .brand { display: flex; align-items: center; gap: 10px; color: #F5F5F7;
  font-size: 17px; font-weight: 600; letter-spacing: -0.01em; }
#bn-nav svg { width: 22px; height: 22px; }
#bn-nav nav { display: flex; gap: 28px; }
#bn-nav nav a { color: #8E8E93; font-size: 14px; font-weight: 500; }
#bn-nav nav a:hover { color: #F5F5F7; }
#bn-features, #bn-status, #bn-results { scroll-margin-top: 72px; }  /* clear the navbar */

/* cards */
.block { border-radius: 18px !important; }
#bn-run { font-size: 17px !important; font-weight: 600 !important; padding: 16px !important;
  transition: transform .15s ease; }
#bn-run:hover { transform: scale(1.01); }
.bn-card { background: #1C1C1E; border-radius: 18px; padding: 24px; }
.bn-head { display: flex; justify-content: space-between; align-items: baseline;
  font-size: 13px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  color: #8E8E93; margin-bottom: 18px; }
.bn-head em { font-style: normal; font-family: var(--font-mono); letter-spacing: 0; }

/* speaker timeline */
.lane { display: grid; grid-template-columns: 110px 1fr; align-items: center; gap: 14px;
  margin: 10px 0; }
.lane span { color: var(--c); font-size: 14px; font-weight: 600; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.track { position: relative; height: 10px; border-radius: 5px; background: #2C2C2E; }
.track i { position: absolute; top: 0; bottom: 0; border-radius: 5px; background: var(--c); }

/* transcript */
.transcript { max-height: 480px; overflow-y: auto; }
.cue { display: grid; grid-template-columns: 44px 110px 1fr; gap: 14px; padding: 12px 0;
  border-top: 1px solid #2C2C2E; }
.cue:first-child { border-top: none; }
.cue time { color: #636366; font-family: var(--font-mono); font-size: 13px; padding-top: 3px; }
.cue b { color: var(--c); font-size: 14px; font-weight: 600; padding-top: 2px; }
.cue p { margin: 0; color: #F5F5F7; font-size: 16px; line-height: 1.5; }
.empty { color: #8E8E93; }

/* loading card */
.steps { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px; list-style: none;
  padding: 0; margin: 0 0 18px; }
.steps li { position: relative; padding-top: 14px; font-size: 13px; color: #636366; }
.steps li::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
  border-radius: 2px; background: #2C2C2E; }
.steps li.done { color: #F5F5F7; }
.steps li.done::before { background: #FFB300; }
.steps li.now { color: #FFB300; font-weight: 600; }
.steps li.now::before { background: #FFB300; animation: bn-pulse 1s ease-in-out infinite; }
@keyframes bn-pulse { 50% { opacity: .3; } }
@media (prefers-reduced-motion: reduce) { .steps li.now::before { animation: none; } }
.bn-note { color: #8E8E93; font-size: 14px; line-height: 1.45; margin: 0; }

/* enroll status: waveform bars while the voiceprint is computed, a tick when done */
.bn-enroll { display: flex; align-items: center; gap: 14px; background: #1C1C1E;
  border-radius: 18px; padding: 16px 20px; color: #F5F5F7; font-size: 15px; font-weight: 500; }
.bn-enroll .wave { display: flex; align-items: center; gap: 4px; height: 28px; }
.bn-enroll .wave i { width: 4px; height: 100%; border-radius: 2px; background: #FFB300;
  transform: scaleY(.25); animation: bn-wave 1s ease-in-out infinite; }
.bn-enroll .wave i:nth-child(2) { animation-delay: .12s; }
.bn-enroll .wave i:nth-child(3) { animation-delay: .24s; }
.bn-enroll .wave i:nth-child(4) { animation-delay: .36s; }
.bn-enroll .wave i:nth-child(5) { animation-delay: .48s; }
@keyframes bn-wave { 50% { transform: scaleY(1); } }
.bn-enroll .tick { width: 28px; height: 28px; color: #30D158; }
.bn-enroll.done .tick path { stroke-dasharray: 24; animation: bn-draw .4s ease-out; }
@keyframes bn-draw { from { stroke-dashoffset: 24; } }
@media (prefers-reduced-motion: reduce) {
  .bn-enroll .wave i { animation: none; transform: scaleY(.6); }
  .bn-enroll.done .tick path { animation: none; }
}

/* one page: hero, the demo video, the pipeline under it, then enroll beside transcribe */
.bn-hero { padding: 96px 0 32px; }  /* 56px navbar + 40px breathing space */
.bn-hero h2 { font-size: clamp(36px, 6vw, 56px); font-weight: 700; line-height: 1.05;
  letter-spacing: -0.03em; margin: 0 0 16px; color: #F5F5F7; }
.bn-hero h2 span { color: #FFB300; }
.bn-hero p { font-size: 19px; line-height: 1.45; color: #8E8E93; margin: 0; max-width: 560px; }
#bn-video { border-radius: 18px !important; overflow: hidden; }
.bn-video-ph { aspect-ratio: 16 / 9; background: #1C1C1E; border-radius: 18px; display: flex;
  flex-direction: column; align-items: center; justify-content: center; gap: 12px;
  color: #636366; font-size: 14px; }
.bn-video-ph svg { width: 64px; height: 64px; }

/* pipeline: a horizontal flow, input -> 5 stages -> output; vertical on narrow screens */
.bn-flow { background: #1C1C1E; border-radius: 18px; padding: 24px 24px 28px; }
.bn-flow ol { display: grid; grid-template-columns: repeat(7, 1fr); gap: 12px;
  list-style: none; padding: 0 !important; margin: 0; }
.bn-flow li { position: relative; margin: 0; padding: 0; }
.bn-flow li:not(:last-child)::after { content: ""; position: absolute; top: 13px; left: 36px;
  right: -4px; height: 2px; background: #2C2C2E; }  /* the connector to the next step */
.bn-flow li > i { width: 28px; height: 28px; margin-bottom: 12px; border-radius: 50%;
  display: grid; place-items: center; font-style: normal; font-size: 11px; font-weight: 600;
  font-family: var(--font-mono); color: #FFB300; background: #2C2C2E; }
.bn-flow li.io > i { background: #FFB300; color: #000; }
.bn-flow b { display: block; color: #F5F5F7; font-size: 15px; font-weight: 600; margin-bottom: 4px; }
.bn-flow span { display: block; color: #8E8E93; font-size: 13px; line-height: 1.4; }
.bn-flow em { display: block; margin-top: 6px; font-style: normal; color: #636366;
  font-size: 11px; font-family: var(--font-mono); }
@media (max-width: 800px) {
  .bn-flow ol { grid-template-columns: 1fr; gap: 18px; }
  .bn-flow li { display: grid; grid-template-columns: 28px 1fr; gap: 14px; }
  .bn-flow li > i { margin: 0; }
  .bn-flow li:not(:last-child)::after { top: 34px; bottom: -14px; left: 13px; right: auto;
    width: 2px; height: auto; }
}

/* features: enroll beside transcribe */
#bn-features { margin-top: 56px; padding-top: 48px; border-top: 1px solid #1C1C1E; gap: 40px; }
#bn-results { margin-bottom: 64px; }
.bn-step { display: flex; gap: 14px; align-items: center; margin: 0 0 4px; }
.bn-step > i { flex: none; width: 32px; height: 32px; border-radius: 50%; display: grid;
  place-items: center; font-style: normal; font-weight: 700; background: #FFB300; color: #000; }
.bn-step h3 { margin: 0; font-size: 22px; font-weight: 700; letter-spacing: -0.01em;
  color: #F5F5F7; }
.bn-step p { margin: 2px 0 0; color: #8E8E93; font-size: 15px; }

.bn-quote { border-left: 3px solid #FFB300; padding: 4px 0 4px 18px; color: #D1D1D6;
  font-size: 15px; line-height: 1.55; margin: 4px 0 8px; }
.bn-label { color: #8E8E93; font-size: 13px; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase; }
.bn-chips { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.bn-chips .bn-label { margin-right: 4px; }
.chip { background: #1C1C1E; border-radius: 980px; padding: 6px 12px; font-size: 14px;
  color: #F5F5F7; }
.chip em { font-style: normal; color: #636366; margin-left: 6px; font-family: var(--font-mono);
  font-size: 12px; }

@media (max-width: 600px) {
  #bn-nav .inner { padding: 0 16px; }
  #bn-nav nav { gap: 18px; }
  .lane { grid-template-columns: 76px 1fr; }
  .cue { grid-template-columns: 1fr; gap: 2px; }
  .steps li { font-size: 0; padding-top: 4px; }  /* bars only; the header names the step */
}
"""

LOGO = """<div id="bn-nav"><div class="inner">
  <a class="brand" href="#bn-about"><svg viewBox="0 0 24 24" aria-hidden="true">
    <path fill="#FFB300" d="M12 1.5 21.1 6.75v10.5L12 22.5l-9.1-5.25V6.75z"/></svg>BeeNoise</a>
  <nav><a href="#bn-about">About</a><a href="#bn-features">Features</a></nav>
</div></div>"""

HERO = """
<div class="bn-hero">
  <h2>Every voice.<br><span>Crystal clear.</span></h2>
  <p>Noisy recording in. Clean audio and subtitles that know who said what, out.
    Everything runs on this computer.</p>
</div>
"""

PIPELINE = """
<div class="bn-flow">
  <div class="bn-head">How it works</div>
  <ol>
    <li class="io"><i>IN</i><div><b>Recording</b>
      <span>Video or audio, or the mic. Noise is fine.</span></div></li>
    <li><i>1</i><div><b>Denoise</b><span>Strips the background noise.</span>
      <em>DeepFilterNet3</em></div></li>
    <li><i>2</i><div><b>Diarize</b><span>Finds who spoke when.</span>
      <em>pyannote</em></div></li>
    <li><i>3</i><div><b>Identify</b><span>Names enrolled voices; others become
      Speaker 1, 2, …</span><em>ECAPA-TDNN</em></div></li>
    <li><i>4</i><div><b>Transcribe</b><span>Turns speech into timed words.</span>
      <em>Whisper</em></div></li>
    <li><i>5</i><div><b>Merge</b><span>Gives each word to whoever was speaking.</span></div></li>
    <li class="io"><i>OUT</i><div><b>Results</b>
      <span>Clean audio, subtitles (.srt, .vtt) and a subtitled video.</span></div></li>
  </ol>
</div>
"""

VIDEO_PLACEHOLDER = """<div class="bn-video-ph"><svg viewBox="0 0 24 24" aria-hidden="true">
  <circle cx="12" cy="12" r="11" fill="none" stroke="currentColor" stroke-width="1.5"/>
  <path fill="currentColor" d="M10 8.2v7.6l6-3.8z"/></svg>Demo video coming soon</div>"""


def _step(n: int, title: str, sub: str) -> str:
    return f'<div class="bn-step"><i>{n}</i><div><h3>{title}</h3><p>{sub}</p></div></div>'


# Always dark, whatever the OS setting: Gradio's dark theme is a `dark` class on <body>.
FORCE_DARK = "() => { document.body.classList.add('dark'); }"

# delete_cache: every hour, drop Gradio's copies of uploads/results older than an hour.
with gr.Blocks(title="BeeNoise", theme=THEME, css=CSS, js=FORCE_DARK,
               delete_cache=(MAX_OUTPUT_AGE_SEC, MAX_OUTPUT_AGE_SEC)) as demo:
    gr.HTML(LOGO, elem_id="bn-navwrap")

    # About: hero, the demo video, and the pipeline under it.
    gr.HTML(HERO, elem_id="bn-about", padding=False)
    if DEMO_VIDEO.exists():
        gr.Video(str(DEMO_VIDEO), show_label=False, interactive=False,
                 show_download_button=False, show_share_button=False, elem_id="bn-video")
    else:
        gr.HTML(VIDEO_PLACEHOLDER, padding=False)
    gr.HTML(PIPELINE, padding=False)

    # Features: enroll beside transcribe.
    with gr.Row(elem_id="bn-features"):
        with gr.Column(min_width=360):
            gr.HTML(_step(1, "Enroll your voice",
                          "Read this aloud for 30–60 seconds. Skip if you're already enrolled."),
                    padding=False)
            lang = gr.Radio(["English", "Bahasa"], value="English", show_label=False,
                            container=False)
            paragraph = gr.HTML(_paragraph_html("English"), padding=False)
            lang.change(_paragraph_html, lang, paragraph)
            name = gr.Textbox(label="Your name", placeholder="e.g. Alex", max_lines=1)
            rec = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Your reading")
            den = gr.Checkbox(label="Denoise before enrolling")
            enroll_btn = gr.Button("Enroll", variant="primary", size="lg")
            enroll_status = gr.HTML(visible=False, padding=False)
            # Who's enrolled, and removing them: hidden on a Space, where it's public.
            enrolled = gr.HTML(visible=not ON_SPACE, padding=False)
            with gr.Accordion("Remove a speaker", open=False, visible=not ON_SPACE):
                with gr.Row(equal_height=True):
                    who = gr.Dropdown(show_label=False, choices=[], scale=3)
                    delete_btn = gr.Button("Delete", variant="stop", scale=1)
            enroll_btn.click(_enroll, [name, rec, den], [enroll_btn, enroll_status, enrolled, who],
                             show_progress="hidden")
            # Voiceprints can't be recovered, so ask first; cancelling sends None.
            delete_btn.click(
                _delete_speaker, who, [enrolled, who],
                js="(n) => n && confirm(`Delete ${n}'s voiceprint? This can't be undone.`)"
                   " ? n : null")

        with gr.Column(min_width=360):
            gr.HTML(_step(2, "Transcribe", "Upload a video or audio file, or record one."),
                    padding=False)
            inp = gr.File(label="Upload video or audio", type="filepath", height=180)
            mic = gr.Audio(sources=["microphone"], type="filepath", label="…or record now")
            # Only one input at a time: a new upload clears the recording and vice versa.
            inp.upload(lambda: None, None, mic)
            mic.stop_recording(lambda: None, None, inp)
            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    denoiser = gr.Radio(["deepfilternet", "spectral", "none"],
                                        value=CFG["denoise"]["backend"], label="Denoiser")
                    diarizer = gr.Radio(["pyannote", "ecapa_cluster"],
                                        value=CFG["diarize"]["backend"], label="Diarizer")
                with gr.Row():
                    strategy = gr.Radio(["full", "segment"], value=CFG["stt"]["strategy"],
                                        label="Transcription strategy")
                    n_spk = gr.Number(label="Speakers (blank = auto)", precision=0)
                burn = gr.Checkbox(label="Burn subtitles into the video picture")
            go = gr.Button("Transcribe", variant="primary", size="lg", elem_id="bn-run")

    # Progress and results span the full width, under both columns.
    status = gr.HTML(visible=False, elem_id="bn-status")
    with gr.Column(visible=False, elem_id="bn-results") as results:
        timeline = gr.HTML()
        transcript = gr.HTML()
        with gr.Row(equal_height=True):
            before = gr.Audio(label="Before", interactive=False)
            after = gr.Audio(label="After — denoised", type="filepath", interactive=False)
        out_video = gr.Video(label="Subtitled video", visible=False)
        with gr.Accordion("Downloads & log", open=False):
            out_files = gr.File(label="Subtitles & data", file_count="multiple")
            out_log = gr.Textbox(label="Log", lines=8, show_label=False)
    outputs = [go, status, results, timeline, transcript, before, after, out_video,
               out_files, out_log]
    # Scroll the loading card, then the results, into view: both land below the fold.
    scroll = ("() => setTimeout(() => document.getElementById('{}')"
              "?.scrollIntoView({{behavior: 'smooth', block: 'start'}}), 300)")
    go.click(None, js=scroll.format("bn-status"))
    go.click(_transcribe, [inp, mic, denoiser, diarizer, n_spk, strategy, burn], outputs,
             show_progress="hidden").success(None, js=scroll.format("bn-results"))
    demo.load(_speakers, None, [enrolled, who])


def launch(share: bool = False):
    """Start the UI, behind a login if BEENOISE_PASSWORD is set."""
    password = os.environ.get("BEENOISE_PASSWORD", "").strip()
    if ON_SPACE and not password:
        raise SystemExit("Refusing to start on a public Space without a login: add a "
                         "BEENOISE_PASSWORD secret in the Space's settings (see DEPLOY_SPACES.md).")
    demo.launch(share=share, auth=("beenoise", password) if password else None)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--share", action="store_true",
                   help="expose a public URL (e.g. when running on a remote machine)")
    args = p.parse_args()
    launch(share=args.share)
