"""Local web demo: python -m src.demo_ui   (opens http://127.0.0.1:7860)
Live-reload while editing the UI: gradio app.py

Tabs:
  Transcribe — upload a video/audio file, run the full pipeline, get a speaker
               timeline, a colour-coded transcript, before/after audio, the
               subtitled video and subtitle files.
  Enroll     — record or upload a paragraph reading to add a speaker.
  Speakers   — who is enrolled, and deleting them (hidden on a Hugging Face
               Space, so visitors can't see each other's names).
  About      — what BeeNoise does and how.

Set BEENOISE_PASSWORD to put the UI behind a login (user: beenoise). It's
required on a Space, because the URL is public and permanent.

Look: always-dark, Apple-style (black, grey cards, one amber accent), with the
tabs styled as a top navbar next to the logo. The theme variables are below;
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
    """Table rows + a refreshed delete dropdown."""
    ps = profiles.get_all()
    rows = [[p.name, round(p.speech_sec, 1), p.n_chunks, round(p.spread, 3), p.created_at[:19]]
            for p in ps]
    return rows, gr.Dropdown(choices=[p.name for p in ps], value=None)


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
.gradio-container { width: 100% !important; max-width: 880px !important; margin: 0 auto !important; }
.gradio-container > main { padding-top: 0 !important; }  /* tabs sit flush in the navbar */
footer { display: none !important; }

/* navbar: a fixed frosted strip holding the logo; Gradio's tab bar is pinned on
   top of it, right-aligned, so the tabs read as nav links */
#bn-navwrap { position: fixed !important; inset: 0 0 auto 0; margin: 0 !important;
  padding: 0 !important; z-index: 10; }  /* out of the layout flow: no gap above the tabs */
#bn-nav { position: fixed; top: 0; left: 0; right: 0; height: 56px; z-index: 10;
  background: rgba(0,0,0,.9); border-bottom: 1px solid #1C1C1E;
  backdrop-filter: saturate(180%) blur(20px); -webkit-backdrop-filter: saturate(180%) blur(20px); }
#bn-nav .inner { max-width: 880px; height: 100%; margin: 0 auto; padding: 0 32px;
  display: flex; align-items: center; gap: 10px; color: #F5F5F7;
  font-size: 17px; font-weight: 600; letter-spacing: -0.01em; }
#bn-nav svg { width: 22px; height: 22px; }
/* fixed, not sticky: a Gradio ancestor breaks sticky, so the tabs scrolled away */
.tab-wrapper { position: fixed !important; top: 0; left: 50%; transform: translateX(-50%);
  width: min(880px, 100%); box-sizing: border-box; z-index: 11; height: 56px !important;
  padding: 0 32px !important; }
.tabs { padding-top: 96px; }  /* room for the fixed bar: 56px + 40px breathing space */
/* Right-align only the visible strip. Gradio measures an absolutely positioned hidden
   copy (.visually-hidden) to decide which tabs fit; any right-alignment that moves that
   copy (including justify-content on .tab-wrapper) pushes tabs into the "…" menu. */
.tab-container { height: 56px !important; gap: 28px; }
.tab-container:not(.visually-hidden) { justify-content: flex-end; }
.tab-container::after { display: none !important; }
.tab-container button { height: 56px; padding: 0 !important; background: none !important;
  border: none !important; color: #8E8E93 !important; font-size: 14px !important;
  font-weight: 500 !important; }
.tab-container button:hover, .tab-container button.selected { color: #F5F5F7 !important; }
.tab-container button.selected::after { background: #FFB300 !important; }
#bn-status, #bn-results { scroll-margin-top: 72px; }  /* don't land under the navbar */

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

/* about */
.bn-about { max-width: 640px; margin: 0 auto; padding: 16px 0 64px; }
.bn-about h2 { font-size: clamp(32px, 6vw, 48px); font-weight: 700; line-height: 1.05;
  letter-spacing: -0.03em; margin: 0 0 16px; color: #F5F5F7; }
.bn-about h2 span { color: #FFB300; }
.bn-about .lead { font-size: 19px; line-height: 1.45; color: #8E8E93; margin: 0 0 40px; }
.bn-about h3 { font-size: 13px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  color: #8E8E93; margin: 40px 0 12px; }
.bn-about p { font-size: 16px; line-height: 1.55; color: #D1D1D6; margin: 0; }
.bn-list { list-style: none; padding: 0 !important; margin: 0; counter-reset: step;
  background: #1C1C1E; border-radius: 18px; }
.bn-list li { counter-increment: step; display: grid; grid-template-columns: 24px 104px 1fr;
  gap: 12px; padding: 16px 20px; margin: 0; border-top: 1px solid #2C2C2E; font-size: 15px; }
.bn-list li:first-child { border-top: none; }
.bn-list li::before { content: counter(step); color: #FFB300; font-weight: 600;
  font-family: var(--font-mono); }
.bn-list b { color: #F5F5F7; font-weight: 600; }
.bn-list span { color: #8E8E93; line-height: 1.45; }

.bn-quote { border-left: 3px solid #FFB300; padding: 4px 0 4px 18px; color: #D1D1D6;
  font-size: 17px; line-height: 1.55; margin: 12px 0 24px; }
.bn-label { color: #8E8E93; font-size: 13px; font-weight: 600; letter-spacing: .06em;
  text-transform: uppercase; }

@media (max-width: 600px) {
  #bn-nav .inner span { display: none; }  /* logo mark only; the tabs need the room */
  .tab-container { gap: 18px; }
  .bn-list li { grid-template-columns: 24px 1fr; }
  .bn-list span { grid-column: 2; }
  .lane { grid-template-columns: 76px 1fr; }
  .cue { grid-template-columns: 1fr; gap: 2px; }
  .steps li { font-size: 0; padding-top: 4px; }  /* bars only; the header names the step */
}
"""

LOGO = """<div id="bn-nav"><div class="inner"><svg viewBox="0 0 24 24" aria-hidden="true">
  <path fill="#FFB300" d="M12 1.5 21.1 6.75v10.5L12 22.5l-9.1-5.25V6.75z"/></svg>
  <span>BeeNoise</span></div></div>"""

ABOUT = """
<div class="bn-about">
  <h2>Every voice.<br><span>Crystal clear.</span></h2>
  <p class="lead">BeeNoise takes a noisy recording and gives back clean audio plus subtitles
    that know who said what.</p>
  <h3>How it works</h3>
  <ol class="bn-list">
    <li><b>Denoise</b><span>DeepFilterNet3 strips out background noise.</span></li>
    <li><b>Diarize</b><span>pyannote works out who spoke when.</span></li>
    <li><b>Identify</b><span>ECAPA-TDNN voiceprints put names on enrolled speakers.
      Everyone else becomes Speaker 1, Speaker 2, …</span></li>
    <li><b>Transcribe</b><span>Whisper turns the speech into timed words.</span></li>
    <li><b>Subtitles</b><span>Words and speakers merge into .srt and .vtt subtitles.</span></li>
  </ol>
  <h3>How to use it</h3>
  <ol class="bn-list">
    <li><b>Enroll</b><span>Each person reads a short paragraph once, 30 to 60 seconds.</span></li>
    <li><b>Transcribe</b><span>Upload a video or audio file, or record on the spot.</span></li>
  </ol>
  <h3>Private by design</h3>
  <p>Every model runs on this computer. Nothing is sent to a cloud API, and voiceprints stay
    in a local database that can be cleared at any time.</p>
</div>
"""

# Always dark, whatever the OS setting: Gradio's dark theme is a `dark` class on <body>.
FORCE_DARK = "() => { document.body.classList.add('dark'); }"

# delete_cache: every hour, drop Gradio's copies of uploads/results older than an hour.
with gr.Blocks(title="BeeNoise", theme=THEME, css=CSS, js=FORCE_DARK,
               delete_cache=(MAX_OUTPUT_AGE_SEC, MAX_OUTPUT_AGE_SEC)) as demo:
    gr.HTML(LOGO, elem_id="bn-navwrap")
    with gr.Tab("Transcribe"):
        with gr.Row(equal_height=True):
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

    with gr.Tab("Enroll"):
        gr.HTML('<p class="bn-label">Read this aloud — 30 to 60 seconds, somewhere quiet</p>'
                f'<p class="bn-quote">{escape(PARAGRAPHS["en"])}</p>'
                '<p class="bn-label">Bahasa Indonesia</p>'
                f'<p class="bn-quote">{escape(PARAGRAPHS["id"])}</p>')
        name = gr.Textbox(label="Name", placeholder="e.g. Alex")
        rec = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Reading")
        den = gr.Checkbox(label="Denoise before enrolling")
        enroll_btn = gr.Button("Enroll", variant="primary", size="lg")
        enroll_msg = gr.Textbox(label="Result")
        enroll_btn.click(_enroll, [name, rec, den], enroll_msg)

    with gr.Tab("Speakers", visible=not ON_SPACE) as speakers_tab:
        table = gr.Dataframe(headers=["name", "speech_sec", "chunks", "spread", "enrolled"])
        with gr.Row(equal_height=True):
            who = gr.Dropdown(label="Remove a speaker", choices=[], scale=3)
            delete_btn = gr.Button("Delete", variant="stop", scale=1)
        refresh = gr.Button("Refresh", size="sm")
        speakers_tab.select(_speakers, None, [table, who])
        refresh.click(_speakers, None, [table, who])
        # Voiceprints can't be recovered, so ask first; cancelling sends None.
        delete_btn.click(_delete_speaker, who, [table, who],
                         js="(n) => n && confirm(`Delete ${n}'s voiceprint? This can't be undone.`)"
                            " ? n : null")

    with gr.Tab("About"):
        gr.HTML(ABOUT)
    demo.load(_speakers, None, [table, who])


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
                   help="expose a public URL (e.g. when running on Colab)")
    args = p.parse_args()
    launch(share=args.share)
