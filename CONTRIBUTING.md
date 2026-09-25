# Working on BeeNoise together

## Who owns what (suggested split)

The pipeline splits cleanly into stages. Each person owns one stage **and its
metric**, so everyone has something to explain in the report and nobody
blocks anyone else. Adjust to the team size: with 3 people, merge rows 4 and 5.

| Owner | Code | Eval | Report section |
|---|---|---|---|
| Member 1 | `src/denoise.py` | `eval/denoise_quality.py` | How DeepFilterNet works; denoising quality results |
| Member 2 | `src/diarize.py` | `eval/der.py` | How pyannote diarization works; DER, pyannote vs. our baseline |
| Member 3 | `src/encoder.py`, `src/speaker_id.py`, `src/enroll.py` | `eval/speaker_id.py` | ECAPA-TDNN, enrollment, threshold tuning, false accepts |
| Member 4 | `src/stt.py`, `src/merge.py`, `src/video.py` | `eval/wer.py` | Whisper; "does denoising help" ablation |
| Everyone | record + label eval data (see `data/README.md`) | | known limitations, demo |

The stages only talk to each other through the types in `src/segments.py`
(`Segment`, `Word`, `Cue`) and plain numpy arrays. If you keep a function's
inputs and outputs the same, you can change anything inside it without
breaking anyone else.

## Git workflow

1. **Never push directly to `main`.** Always branch:
   ```bash
   git checkout main
   ```
   ```bash
   git pull
   ```
   ```bash
   git checkout -b denoise/atten-limit
   ```
   Name branches `<area>/<short-description>`, e.g. `diarize/ecapa-threshold`,
   `eval/wer-plot`, `docs/report-notes`.
2. Commit small, with messages that say *what and why*:
   `denoise: cap attenuation at 12 dB to reduce voice distortion`.
3. Before opening a PR:
   ```bash
   python -m pytest
   ```
   ```bash
   ruff check .
   ```
4. Open a Pull Request into `main`, fill in the template, and ask **one
   teammate** to review. CI runs the tests automatically.
5. The reviewer checks that it runs and makes sense, then merges it. Delete
   the branch after merging.

If `main` moved while you worked: `git pull origin main` into your branch,
fix conflicts, re-run the tests, push.

## Rules that protect the project

- **Never commit data.** No recordings, no `profiles.db`, no `.env`. The
  `.gitignore` blocks the usual paths. Double-check `git status` before
  committing anything under `data/`. Voiceprints are biometric data.
- **Never commit tokens.** Your HF token lives only in your local `.env`.
  If one leaks, revoke it on huggingface.co immediately.
- **Tunable numbers go in `config.yaml`**, not hardcoded in `src/`. If you
  change one, say so in the PR. Results are only comparable if everyone
  knows the settings.
- **Results come from the scripts in `eval/`.** Don't put a number in the
  report unless one of those scripts produced it. Note the config you used.
- **Keep heavy imports lazy** (inside functions: `torch`, `pyannote`,
  `speechbrain`, `faster_whisper`, `df`). That's what lets the unit tests and
  CI run without downloading models.
- If you add a dependency, add it to `requirements.txt` with a comment
  saying why, and tell the team to re-run `pip install -r requirements.txt`.

## Sharing eval data

Put recordings and label files in the team's shared drive folder, mirroring
the `data/eval/` layout in [data/README.md](data/README.md). Everyone
downloads them into their own local `data/eval/`. That way everyone
evaluates on identical files.

## Adding a new backend (e.g. another denoiser)

1. Add a branch in the stage's function (`denoise()`, `diarize()`) and its
   name to that module's `BACKENDS`.
2. Add it to the CLI choices in `src/main.py` and to `config.yaml`'s comment.
3. Run the stage's eval with `--backends <old> <new>` and put the comparison
   in the PR description.
