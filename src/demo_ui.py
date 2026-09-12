"""Gradio demo: mic input, live speaker-ID badge + similarity score,
transcript, reply, and an Enroll New User tab.

Run from the project root:
    python -m src.demo_ui
"""
import numpy as np
import gradio as gr

from . import agent, profiles
from .audio_io import SR
from .encoder import embed
from .stt import transcribe_or_none
from .vad import trim_silence, net_speech_seconds
from .verify import identify
from .config import CFG


def _to_float32_mono(sr: int, audio: np.ndarray) -> np.ndarray:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(np.int16).max \
            if np.issubdtype(audio.dtype, np.integer) else audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * SR / sr)).astype(np.float32)
    return audio


def process_turn(mic_input):
    if mic_input is None:
        return "—", "", "", ""

    sr, audio = mic_input
    audio = _to_float32_mono(sr, audio)
    audio = trim_silence(audio)
    net_speech = net_speech_seconds(audio)

    if net_speech < 0.5:
        return "No speech detected", "", "", ""

    emb = embed(audio)
    text = transcribe_or_none(audio)

    if net_speech < CFG["speaker"]["min_net_speech_sec"]:
        badge = f"Too short to verify ({net_speech:.1f}s net speech)"
        user = None
        score = 0.0
    else:
        result = identify(emb)
        user = result.user
        score = result.score
        badge = (f"{user.name}  (cosine={score:.3f})" if user
                  else f"Unrecognized / guest  (best cosine={score:.3f})")

    if text is None:
        return badge, "(low-confidence transcription — please repeat)", "", ""

    reply = agent.respond(user, text)
    return badge, text, reply, ""


def enroll_ui_fn(name, mic_input):
    if not name or not name.strip():
        return "Please enter a name first."
    if mic_input is None:
        return "Record a clip first."

    sr, audio = mic_input
    audio = _to_float32_mono(sr, audio)
    audio = trim_silence(audio)
    net_speech = net_speech_seconds(audio)
    if net_speech < 1.5:
        return f"Not enough speech ({net_speech:.1f}s) — try a longer, clearer clip."

    e = embed(audio)
    existing = profiles.get_user_by_name(name.strip())
    if existing:
        # Blend with existing centroid rather than overwrite outright.
        n_old = existing.n_utts
        combined = (existing.centroid * n_old + e) / (n_old + 1)
        combined = combined / np.linalg.norm(combined)
        profiles.update_user_embedding(existing.id, combined, n_old + 1, existing.spread)
        return f"Added one more sample to '{name}' (now {n_old + 1} utterances). " \
               f"For a robust profile, use `python -m src.enroll --name {name}` " \
               f"to do the full 10-phrase enrollment instead."
    else:
        profiles.add_user(name.strip(), e, 1, 1.0)
        return f"Created a starter profile for '{name}' from one clip. " \
               f"Run `python -m src.enroll --name {name}` for a proper multi-phrase enrollment."


def build_ui():
    with gr.Blocks(title="Voice-Aware Conversational Agent") as demo:
        gr.Markdown("# Voice-Aware Conversational Agent")
        with gr.Tab("Talk"):
            mic = gr.Audio(sources=["microphone"], type="numpy", label="Speak")
            badge = gr.Textbox(label="Identified speaker")
            transcript = gr.Textbox(label="Transcript")
            reply = gr.Textbox(label="Assistant reply")
            btn = gr.Button("Process")
            btn.click(process_turn, inputs=[mic], outputs=[badge, transcript, reply, gr.Textbox(visible=False)])

        with gr.Tab("Enroll New User"):
            name_box = gr.Textbox(label="Name")
            enroll_mic = gr.Audio(sources=["microphone"], type="numpy", label="Record a sample (4+ seconds)")
            enroll_btn = gr.Button("Add sample")
            enroll_status = gr.Textbox(label="Status")
            enroll_btn.click(enroll_ui_fn, inputs=[name_box, enroll_mic], outputs=[enroll_status])

        with gr.Tab("Enrolled Users"):
            refresh_btn = gr.Button("Refresh")
            users_box = gr.Textbox(label="Users", lines=10)

            def list_users():
                users = profiles.get_all_users()
                if not users:
                    return "(none enrolled yet)"
                return "\n".join(f"{u.name}: {u.n_utts} utterances, spread={u.spread:.3f}"
                                  for u in users)

            refresh_btn.click(list_users, outputs=[users_box])

    return demo


if __name__ == "__main__":
    build_ui().launch()
