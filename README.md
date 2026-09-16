# Student Attendance via Voice Verification

See [BLUEPRINT.md](BLUEPRINT.md) for the full design rationale, architecture
diagram, evaluation plan, and known traps. This README is the "how do I
actually run it" doc.

## What this is

Each student enrolls their voice once. To check in for a session, a student
claims their name from the roster, the system shows a randomly generated
digit challenge (e.g. `8-2-4-9`), and the student reads it aloud. Attendance
is only marked when **both** checks pass:

1. **Speaker verification** — the voice matches that student's enrolled
   voiceprint (cosine similarity vs. their centroid).
2. **Content verification** — the digits read back match the challenge
   generated for *this* attempt.

Requiring both is what stops a recording of the real student's voice from
passing: it won't contain today's random digits. Two failed attempts in a
row lock that name out for a cooldown period (`attendance.lockout_minutes`
in config.yaml) before another attempt is allowed.

## Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

All commands below are run from the project root (`voice_recognition/`),
with the venv activated.

## Windows-specific gotchas already fixed in this repo

- `webrtcvad` needs a C++ compiler to build from source on Windows.
  `requirements.txt` uses `webrtcvad-wheels` instead — same `import webrtcvad`
  API, prebuilt wheel.
- SpeechBrain's model fetcher symlinks by default, which needs Developer Mode
  or admin rights on Windows and fails with `WinError 1314` otherwise.
  `src/encoder.py` passes `local_strategy=LocalStrategy.COPY` to avoid it.

## Quick smoke tests (run in this order)

```bash
python -m src.audio_io      # records 3s, plays it back, saves a wav
python -m src.encoder       # records 4s, prints an embedding shape
python -m src.stt           # records 4s, prints a transcript
python -m src.challenge     # no mic needed — prints a sample challenge + matching logic
```

If any of these fail, fix it before moving on — every later phase builds on
these primitives.

## Day-to-day workflow

**1. Enroll each student:**

```bash
python -m src.enroll --name Alex
python -m src.enroll --name Sam --student-id 2023510042
```

Follow the prompts — 10 short phrases, ~4s each, in a normal speaking voice.
Enrollment uses varied prose sentences, not digits — that builds a more
general voiceprint than repeating the same content would.

**2. Run attendance check-in (CLI):**

```bash
python -m src.main --session "2026-09-16 Speech Recognition"
```

Type the claimed name, read the displayed digits aloud when prompted. The
CLI prints the speaker score, whether the digits matched, and whether
attendance was verified.

**3. Run the Gradio demo (nicer for showing people):**

```bash
python -m src.demo_ui
```

Opens a local web UI with an **Attendance** tab (pick name → generate
challenge → record → submit), an **Enroll New Student** tab, and a
**Roster / Attendance Log** tab. The Attendance tab also has a
"Skip cooldown (demo only)" button so a live demo doesn't have to sit
through the real lockout timer — a real deployment should not expose that
button to students.

## Evaluation

This is the part that earns the grade — don't skip it.

**Speaker verification (EER):**

1. Record held-out utterances (NOT the ones used for enrollment) into:
   - `data/eval/<student_name>/*.wav` — genuine clips per enrolled student (~20 each)
   - `data/eval/_impostors/*.wav` — clips from people who are NOT enrolled
2. Run:
   ```bash
   python -m eval.eer
   ```
   This prints the EER, plots a DET-style curve and a score histogram into
   `eval/results/`, and tells you what to set `speaker.tau` to in
   `config.yaml`.
3. Ablations:
   ```bash
   python -m eval.eer --durations 1 2 4 8
   ```

**Digit-readback accuracy (WER-style):**

1. Create `data/eval/wer/transcripts.json` mapping wav filename → the
   digit string that clip is a readback of, e.g.
   `{"clip1.wav": "8 2 4 9", ...}`, with the matching wav files alongside.
2. Run:
   ```bash
   python -m eval.wer --models tiny base small
   ```
   Produces a WER + latency table in `eval/results/wer_summary.json`.

**Combined-system FAR (the important one — see BLUEPRINT.md Phase 8c):**
Score a genuine recording of student A's readback of challenge X against
student A's centroid, but with a *different* expected challenge Y, and show
`attendance.attempt()` rejects it even though speaker verification alone
would accept it. This demonstrates the digit challenge is load-bearing, not
decorative.

## Project structure

See the "Repo layout" section of [BLUEPRINT.md](BLUEPRINT.md#2-repo-layout).

## Known limitations (say these out loud in your report, don't hide them)

- Verification is text-independent for enrollment but text-*dependent* for
  check-in (the digit challenge) — that's intentional, see BLUEPRINT.md.
- No anti-spoofing against a *live* attacker who hears the challenge and
  splices pre-recorded digit clips together in real time. A static replay
  of an old recording is defended against; a targeted live attack is not.
  Name it as a threat-model gap, not a bug.
- Enrollment and testing on different microphones will inflate EER. Note
  which device(s) you used.
- Voiceprints are biometric data — kept local in `data/profiles.db` only.
  Use `profiles.delete_user(name)` to remove a student's data.
