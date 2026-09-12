# Voice-Aware Conversational Agent

See [BLUEPRINT.md](BLUEPRINT.md) for the full design rationale, architecture
diagram, evaluation plan, and known traps. This README is the "how do I
actually run it" doc.

## Setup

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If using the Anthropic provider (default), set your API key:

```bash
setx ANTHROPIC_API_KEY "sk-..."     # then restart your shell
```

To use a fully offline LLM instead, install [Ollama](https://ollama.com),
run `ollama pull llama3.1:8b`, and set `llm.provider: ollama` in
[config.yaml](config.yaml).

All commands below are run from the project root (`voice_recognition/`),
with the venv activated.

## Windows-specific gotchas already fixed in this repo

- `webrtcvad` needs a C++ compiler to build from source on Windows.
  `requirements.txt` uses `webrtcvad-wheels` instead — same `import webrtcvad`
  API, prebuilt wheel.
- SpeechBrain's model fetcher symlinks by default, which needs Developer Mode
  or admin rights on Windows and fails with `WinError 1314` otherwise.
  `src/encoder.py` passes `local_strategy=LocalStrategy.COPY` to avoid it.
- `pyttsx3`'s Windows driver (SAPI5) needs `pywin32`, included in
  `requirements.txt`.

## Quick smoke tests (run in this order)

```bash
python -m src.audio_io      # records 3s, plays it back, saves a wav
python -m src.encoder       # records 4s, prints an embedding shape
python -m src.stt           # records 4s, prints a transcript
python -m src.agent         # text-only LLM test, no mic needed
```

If any of these fail, fix it before moving on — every later phase builds on
these three primitives.

## Day-to-day workflow

**1. Enroll each team member / test user:**

```bash
python -m src.enroll --name Alex
python -m src.enroll --name Sam
```

Follow the prompts — 10 short phrases, ~4s each, in a normal speaking voice.

**2. Run the full assistant (CLI):**

```bash
python -m src.main
```

Speak into the mic. It identifies who's talking, transcribes, replies via
the LLM, and speaks the reply back.

**3. Run the Gradio demo (nicer for showing people):**

```bash
python -m src.demo_ui
```

Opens a local web UI with a "Talk" tab, an "Enroll New User" tab, and a list
of enrolled users with their embedding stats.

## Evaluation

This is the part that earns the grade — don't skip it.

**Speaker verification (EER):**

1. Record held-out utterances (NOT the ones used for enrollment) into:
   - `data/eval/<user_name>/*.wav` — genuine clips per enrolled user (~20 each)
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

**ASR (WER):**

1. Create `data/eval/wer/transcripts.json`:
   ```json
   {"clip1.wav": "the quick brown fox jumps over the lazy dog", "clip2.wav": "..."}
   ```
   with the matching wav files in `data/eval/wer/`.
2. Run:
   ```bash
   python -m eval.wer --models tiny base small
   ```
   Produces a WER + latency table in `eval/results/wer_summary.json`.

## Project structure

See the "Repo layout" section of [BLUEPRINT.md](BLUEPRINT.md#2-repo-layout).

## Team split (4 people, 16 weeks)

See "Suggested team split" and "Revised 16-week timeline" — divide work by
module ownership:

| Person | Files they own |
|---|---|
| Audio/Speaker | `src/audio_io.py`, `src/vad.py`, `src/encoder.py`, `src/enroll.py`, `src/verify.py` |
| ASR | `src/stt.py`, `eval/wer.py` |
| Backend/Agent | `src/profiles.py`, `src/agent.py`, `src/main.py` |
| Eval/UI | `eval/eer.py`, `src/demo_ui.py`, report figures |

Everyone agrees on the audio contract on day one (16kHz mono float32) and
never touches it again — that's what lets these be developed in parallel.

## Known limitations (say these out loud in your report, don't hide them)

- Verification is text-independent but not content-free: utterances under
  ~1.5s of net speech are unreliable — see `speaker.min_net_speech_sec`.
- No anti-spoofing: a recording of an enrolled user's voice will pass
  verification. Out of scope, but name it as a threat model gap.
- Enrollment and testing on different microphones will inflate EER. Note
  which device(s) you used.
- Voiceprints are biometric data — kept local in `data/profiles.db` only.
  Use `profiles.delete_user(name)` to remove someone's data.
