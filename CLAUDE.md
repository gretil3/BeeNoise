# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

BeeNoise: noisy video/audio in → denoised audio + speaker-labelled subtitles out.
Pipeline: denoise (DeepFilterNet) → diarize (pyannote) → speaker ID (ECAPA-TDNN) → Whisper → subtitles.
College speech-recognition project; see BLUEPRINT.md for design decisions and the eval plan.

## Environment

- **Python 3.10 or 3.11 only** (`requires-python = ">=3.10,<3.12"`). DeepFilterNet has no wheels for 3.12+.
- With `uv`, `requirements.txt`'s `--extra-index-url` (PyTorch CPU index) needs
  `uv pip install --index-strategy unsafe-best-match -r requirements.txt -r requirements-dev.txt`,
  otherwise uv picks an old `packaging` from the torch index and resolution fails.
- `HF_TOKEN` goes in `.env` (read by `src/config.py`); needed only for the `pyannote` diarizer.
  `--diarizer ecapa_cluster` works without it.
- Model downloads go to `.cache/` and `models/` inside the repo (`HF_HOME` is set in `src/config.py`).

## Commands

```bash
python -m pytest                      # unit tests, no models needed (CI runs this + ruff)
python -m pytest tests/test_merge.py::test_words_take_the_speaker_they_overlap_most   # single test
ruff check .                          # lint (line-length 100, rules E,F,W,I,B)
python -m src.check_env [--models]    # verify install; --models loads every model once
python -m src.main path/to/clip.mp4   # full pipeline -> outputs/<clip>/
python -m src.enroll --name Alex      # enroll a speaker (mic, or --file)
python -m src.demo_ui                 # Gradio UI at http://127.0.0.1:7860
gradio app.py                         # same UI with hot reload on file save
python -m eval.{denoise_quality,der,speaker_id,wer}   # metrics -> eval/results/
```

## Architecture

- `src/main.py` orchestrates stages 1–5 (`run` on an in-memory waveform, `run_file` for file I/O);
  `Options` carries per-run overrides of `config.yaml` defaults. CLI and demo UI both go through it.
- Stages communicate only via the dataclasses in `src/segments.py` (`Segment`, `Word`, `Cue`)
  and numpy arrays. Keep a stage's inputs/outputs the same and its internals can change freely.
- Two sample rates: `audio.output_sr` (48 kHz) for the denoised output track, `audio.analysis_sr` (16 kHz)
  for diarization/speaker ID/Whisper. `pipeline.*_on` in config chooses raw vs denoised track per stage (ablation).
- Speaker ID (`src/speaker_id.py`): per-cluster duration-weighted ECAPA voiceprint, cosine vs enrolled
  centroids, one-to-one Hungarian assignment, accepted only if cosine ≥ `speaker.tau`;
  otherwise "Speaker N". Embeddings are L2-normalised in `src/encoder.py`, so cosine is a dot product.
- Enrolled voiceprints live in SQLite `data/profiles.db` (`src/profiles.py`), gitignored biometric data.
- Merge (`src/merge.py`) assigns speakers per word by max overlap with diarized turns (WhisperX-style),
  then groups words into cues.
- `src/demo_ui.py` defines the Gradio `demo`; `app.py` is the Hugging Face Spaces entry point.
  On a Space (`SPACE_ID` set) the UI refuses to start without `BEENOISE_PASSWORD`, and the Speakers tab is hidden.

## Conventions

- `src/config.py` must be imported before any HF/pyannote/speechbrain/faster_whisper import
  (it sets `HF_HOME` and loads `.env`); `src/__init__.py` does this.
- Keep heavy imports (`torch`, `pyannote`, `speechbrain`, `faster_whisper`, `df`) lazy, inside functions.
  Models are loaded once into module globals (`_get_model()` pattern). This keeps tests/CI model-free.
- Tunable numbers go in `config.yaml`, never hardcoded in `src/`.
- New backend: add a branch + name to the stage module's `BACKENDS`, the CLI choices in `src/main.py`,
  and the `config.yaml` comment.
- Never commit recordings, `profiles.db` or `.env`. Branch per change (`<area>/<desc>`), PR into `main`.
