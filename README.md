# BeeNoise — Multi-Speaker Denoised Transcription

Give it a noisy recording (vlog, lecture, group conversation). It returns:

1. a **denoised** audio/video,
2. **subtitles** with timestamps,
3. **speaker names** on every line: enrolled people by name, everyone else as
   `Speaker 1`, `Speaker 2`, ...

```
video/audio ─► [1] denoise ─► [2] diarize ─► [3] speaker ID ─► [4] Whisper ─► [5] subtitles
               DeepFilterNet   pyannote 3.1    ECAPA-TDNN        faster-whisper   .srt/.vtt/.mp4
```

- **[BLUEPRINT.md](BLUEPRINT.md):** the concept, evaluation plan and design decisions. Read this first.
- **[CONTRIBUTING.md](CONTRIBUTING.md):** how the team works together (branches, PRs, who owns what).
- **[data/README.md](data/README.md):** what eval data to record and how to label it.

---

## Setup (each teammate, once)

**1. Python 3.10 or 3.11.** Not 3.12+, because DeepFilterNet has no wheels
for it. Check with `py -0` on Windows. Install 3.11 from python.org if needed.

**2. Create the environment** (from the repo root):

```bash
py -3.11 -m venv .venv
```
```bash
.venv\Scripts\activate
```
```bash
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

On macOS/Linux use `python3.11 -m venv .venv` and `source .venv/bin/activate`.
With an NVIDIA GPU, install torch from the matching CUDA index instead, then
set `device: cuda` in `config.yaml` (and `compute_type: float16` for stt).

**3. HuggingFace token** (for pyannote diarization):

1. Make a free account → https://huggingface.co/settings/tokens → create a *read* token.
2. While logged in, click "Agree" on **both**
   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
   and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0).
3. `copy .env.example .env` and paste your token into `.env`. It's gitignored, so never commit it.

No token yet? Everything still works with `--diarizer ecapa_cluster`.

**4. Check everything:**

```bash
python -m src.check_env
```
```bash
python -m src.check_env --models
```

The second command loads every model once. The first run downloads ~1–2 GB
into `.cache/` and `models/` inside the project folder, so C: doesn't fill up.

**5. Run the tests** (a few seconds, no models needed):

```bash
python -m pytest
```

---

## Usage

**Enroll each speaker** (reads a paragraph aloud, ~30–60 s):

```bash
python -m src.enroll --name Alex
```
```bash
python -m src.enroll --name Budi --lang id
```
```bash
python -m src.enroll --name Citra --file citra_reading.m4a
```
```bash
python -m src.profiles list
```

**Process a recording:**

```bash
python -m src.main path/to/recording.mp4
```

Outputs go to `outputs/<recording>/`: `denoised.wav`, `subtitles.srt`,
`subtitles.vtt`, `transcript.json`, `diarization.rttm`, and for videos
`<name>_subtitled.mp4` (denoised audio + a subtitle track you can toggle).

Useful flags:

| Flag | Effect |
|---|---|
| `--num-speakers 3` | tell the diarizer how many people there are (big accuracy boost) |
| `--burn` | draw subtitles into the video picture instead of a toggleable track |
| `--denoiser none` / `spectral` | skip denoising / use the classical baseline |
| `--diarizer ecapa_cluster` | our baseline diarizer, no HF token needed |
| `--strategy segment` | run Whisper per speaker turn instead of on the whole track |
| `--language id` | force the language (default: auto-detect) |

Other settings (thresholds, model sizes, subtitle line length) live in
[config.yaml](config.yaml).

**Demo UI** (for presenting):

```bash
python -m src.demo_ui
```

**Denoise one file only:**

```bash
python -m src.denoise noisy.wav
```

---

## Evaluation

First record the eval data described in [data/README.md](data/README.md).
Every script compares **with vs. without denoising** by default (the
blueprint's ablation) and writes JSON + PNG charts to `eval/results/`.

| Metric | Command | Needs |
|---|---|---|
| 1. Denoising quality (ΔSNR, ΔSI-SDR, STOI) | `python -m eval.denoise_quality` | `eval/wer/*.wav`, `eval/noise/` |
| 2. Diarization Error Rate | `python -m eval.der` | `eval/diarization/` + labels |
| 3. Speaker ID accuracy + false accepts, tau tuning | `python -m eval.speaker_id` | enrolled speakers + `eval/diarization/` |
| 4. WER per noise level | `python -m eval.wer` | `eval/wer/` + `transcripts.json`, `eval/noise/` |

Extra comparisons for the report:

```bash
python -m eval.der --diarizers pyannote ecapa_cluster --oracle-num-speakers
```
```bash
python -m eval.denoise_quality --backends deepfilternet spectral --save-mixtures
```
```bash
python -m eval.speaker_id --mode pipeline
```

After running `eval.speaker_id`, copy the recommended `tau` into
`config.yaml` (`speaker.tau` is a placeholder until then).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cargo, the Rust package manager, is not installed` during pip install | You're on Python 3.12+. Recreate the venv with 3.10/3.11. |
| `Could not load pyannote/speaker-diarization-3.1` | Accept the terms on **both** pyannote model pages with the token's account, and check `.env`. |
| `WinError 1314` (symlink) | Already handled (`LocalStrategy.COPY` in `src/encoder.py`). If it's from HF hub, turn on Windows Developer Mode. |
| Model downloads fill up C: | They go to `.cache/` in the repo via `HF_HOME` (set in `src/config.py`). DeepFilterNet's small model goes to `%LOCALAPPDATA%\DeepFilterNet`. |
| Everything is slow | Use `stt.model_size: base` while developing, and `small`/`medium` for final numbers. The first run of each model is slower because it downloads. |
| Names are wrong / everyone is "Speaker N" | Tune `speaker.tau` with `eval.speaker_id`, re-enroll with 30 s+ of clean speech, or try `pipeline.speaker_id_on: raw`. |
