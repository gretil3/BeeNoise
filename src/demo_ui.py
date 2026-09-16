"""Gradio demo: pick your name, read back a random digit challenge, submit
the recording. Attendance is marked only if the voice matches the claimed
student's enrolled profile AND the digits were read back correctly.

Run from the project root:
    python -m src.demo_ui
"""
from datetime import date

import numpy as np
import gradio as gr

from . import attendance, challenge, profiles
from .audio_io import SR
from .encoder import embed
from .vad import trim_silence, net_speech_seconds


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


def _roster_names():
    return [u.name for u in profiles.get_all_users()]


def new_challenge_fn(name, session):
    if not name:
        return "Pick your name first.", "", gr.update(visible=False)
    user = profiles.find_user_ci(name)
    if user is None:
        return f"No enrolled student named '{name}'.", "", gr.update(visible=False)
    if profiles.is_present(user.id, session):
        return f"{name} is already marked present for '{session}'.", "", gr.update(visible=False)
    locked, until = attendance.lockout_status(user.id, session)
    if locked:
        return (f"Locked out until {until} after too many failed attempts.", "",
                gr.update(visible=True))
    digits = attendance.new_challenge()
    return f"Please read aloud: {challenge.format_for_display(digits)}", digits, gr.update(visible=False)


def submit_fn(name, session, digits, mic_input):
    if not digits:
        return "Generate a challenge first.", "", gr.update(visible=False)
    if mic_input is None:
        return "Record your readback first.", "", gr.update(visible=False)
    user = profiles.find_user_ci(name)
    if user is None:
        return f"No enrolled student named '{name}'.", "", gr.update(visible=False)

    sr, audio = mic_input
    audio = _to_float32_mono(sr, audio)

    result = attendance.attempt(user, session, digits, audio)

    detail = (f"speaker score={result.speaker_score:.3f} "
              f"({'pass' if result.speaker_pass else 'fail'})  |  "
              f"digits {'matched' if result.content_pass else 'did not match'} "
              f"(heard: {result.transcribed!r}, "
              f"expected: {challenge.format_for_display(result.expected)})")

    if result.marked_present:
        return f"Attendance verified for {user.name}.", detail, gr.update(visible=False)
    if result.locked_out:
        return (f"Too many failed attempts — {user.name} is locked out until "
                f"{result.locked_until}.", detail, gr.update(visible=True))
    return f"Verification failed: {result.reason}. Try again.", detail, gr.update(visible=False)


def skip_lockout_fn(name, session):
    user = profiles.find_user_ci(name)
    if user is None:
        return "No enrolled student with that name.", gr.update(visible=False)
    attendance.skip_lockout(user.id, session)
    return (f"Lockout cleared for {user.name}. (Demo-only shortcut — a real deployment "
            f"should not expose this button to students.)", gr.update(visible=False))


def enroll_ui_fn(name, student_id, mic_input):
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
        n_old = existing.n_utts
        combined = (existing.centroid * n_old + e) / (n_old + 1)
        combined = combined / np.linalg.norm(combined)
        profiles.update_user_embedding(existing.id, combined, n_old + 1, existing.spread)
        return (f"Added one more sample to '{name}' (now {n_old + 1} utterances). "
                f"For a robust profile, use `python -m src.enroll --name {name}` "
                f"to do the full multi-phrase enrollment instead.")
    else:
        profiles.add_user(name.strip(), e, 1, 1.0, student_id=(student_id or "").strip() or None)
        return (f"Created a starter profile for '{name}' from one clip. "
                f"Run `python -m src.enroll --name {name}` for a proper multi-phrase enrollment.")


def attendance_log_fn(session):
    rows = profiles.get_attendance(session)
    if not rows:
        return "(no one marked present yet for this session)"
    return "\n".join(
        f"{r['name']}: present at {r['ts']} "
        f"(score={r['speaker_score']:.3f}, attempts={r['attempts_used']})"
        for r in rows
    )


def list_users_fn():
    users = profiles.get_all_users()
    if not users:
        return "(none enrolled yet)"
    return "\n".join(
        f"{u.name}" + (f" ({u.student_id})" if u.student_id else "") +
        f": {u.n_utts} utterances, spread={u.spread:.3f}"
        for u in users
    )


def build_ui():
    with gr.Blocks(title="Student Attendance — Voice Verification") as demo:
        gr.Markdown("# Student Attendance via Voice Verification")
        gr.Markdown(
            "Two checks must both pass to mark attendance: the voice must match the "
            "enrolled student's voiceprint, **and** the student must correctly read "
            "back a freshly generated digit challenge (this is what defeats a simple "
            "replay of a recorded voice)."
        )

        digits_state = gr.State("")

        with gr.Tab("Attendance"):
            with gr.Row():
                session_box = gr.Textbox(label="Session", value=str(date.today()))
                name_box = gr.Dropdown(label="Your name", choices=_roster_names(),
                                        allow_custom_value=True)
                refresh_roster_btn = gr.Button("Refresh roster")
            refresh_roster_btn.click(lambda: gr.update(choices=_roster_names()),
                                      outputs=[name_box])

            challenge_btn = gr.Button("Generate challenge")
            challenge_box = gr.Textbox(label="Read this aloud", interactive=False)

            mic = gr.Audio(sources=["microphone"], type="numpy", label="Record your readback")
            submit_btn = gr.Button("Submit", variant="primary")

            status_box = gr.Textbox(label="Result")
            detail_box = gr.Textbox(label="Detail")
            skip_btn = gr.Button("Skip cooldown (demo only)", visible=False)

            challenge_btn.click(new_challenge_fn, inputs=[name_box, session_box],
                                 outputs=[challenge_box, digits_state, skip_btn])
            submit_btn.click(submit_fn, inputs=[name_box, session_box, digits_state, mic],
                              outputs=[status_box, detail_box, skip_btn])
            skip_btn.click(skip_lockout_fn, inputs=[name_box, session_box],
                            outputs=[status_box, skip_btn])

        with gr.Tab("Enroll New Student"):
            enroll_name = gr.Textbox(label="Name")
            enroll_id = gr.Textbox(label="Student ID (optional)")
            enroll_mic = gr.Audio(sources=["microphone"], type="numpy",
                                   label="Record a sample (4+ seconds)")
            enroll_btn = gr.Button("Add sample")
            enroll_status = gr.Textbox(label="Status")
            enroll_btn.click(enroll_ui_fn, inputs=[enroll_name, enroll_id, enroll_mic],
                              outputs=[enroll_status])

        with gr.Tab("Roster / Attendance Log"):
            log_session = gr.Textbox(label="Session", value=str(date.today()))
            log_btn = gr.Button("Refresh")
            log_box = gr.Textbox(label="Present", lines=10)
            log_btn.click(attendance_log_fn, inputs=[log_session], outputs=[log_box])

            gr.Markdown("---")
            roster_btn = gr.Button("Show enrolled students")
            roster_box = gr.Textbox(label="Enrolled", lines=10)
            roster_btn.click(list_users_fn, outputs=[roster_box])

    return demo


if __name__ == "__main__":
    build_ui().launch()
