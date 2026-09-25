# Blueprint: Multi-Speaker Denoised Transcription System

> Source of truth for *what* we are building. Sections 1–8 are the team's
> agreed concept (converted from `BLUEPRINT.pdf`). Section 9 records the
> implementation decisions made while setting up the repo. If you change a
> decision, update section 9 in the same PR.

## 1. What this is

Take a noisy recording (vlog, lecture, multi-person conversation) and produce:

1. A **denoised** version of the audio/video
2. **Subtitles** (transcription with timestamps)
3. **Speaker labels** on those subtitles — who said what, for recordings with
   multiple people

Each speaker must **enroll once** by reading a paragraph aloud before the
system can recognize them by name in future recordings. Unenrolled speakers
still get separated and labeled (e.g. "Speaker 2"), just not named.

This replaces the earlier voice-attendance concept. Core techniques carry over
(speaker embeddings, ASR); this version adds denoising and diarization on top,
for a richer end-to-end pipeline.

## 2. Why this scope, not just "call an API"

The professor allows APIs, but still requires the algorithm to be explained.
So every stage uses an **open-source, locally-run deep learning model** the
team can actually describe in the report — no black-box API calls where "how
does it work" has no answer. This also keeps the option to run
offline/on-device.

## 3. Pipeline

```
input video/audio
      │
      ▼
[1] Denoise ─────────► cleaned audio track
      │
      ▼
[2] Diarize ─────────► "who spoke when" segments
      │                (Speaker A: 0:00–0:04, Speaker B: 0:04–0:09, ...)
      ▼
[3] Speaker ID ──────► match each segment's voice against enrolled voiceprints
      │                → name, or "Speaker N" if unknown
      ▼
[4] Transcribe ──────► text for each segment (Whisper)
      │
      ▼
[5] Merge ───────────► subtitle file (.srt/.vtt) with speaker name + text +
                       timestamps, optionally burned into the denoised video
```

## 4. Components & models

| Stage | Model | Library | What it does |
|---|---|---|---|
| 1. Denoise | DeepFilterNet (or Demucs) | `deepfilternet` | Neural net predicts clean audio from noisy input — explainable, local, open-source |
| 2. Diarization | pyannote speaker-diarization pipeline | `pyannote.audio` | Clusters audio into "same speaker" segments without needing to know who they are yet |
| 3. Speaker ID | ECAPA-TDNN speaker embeddings (same as before) | `speechbrain` | Compares each diarized segment's embedding against enrolled voiceprint centroids |
| 4. Transcription | Whisper | `faster-whisper` | Run per-segment, or on the full track and then aligned to segments |

**Reference implementation: [WhisperX](https://github.com/m-bain/whisperX).**
It already wires Whisper + pyannote diarization + word-level alignment
together. Worth reading its source even though we implement our own version,
so we can explain our design choices against it.

## 5. Enrollment

```bash
python -m src.enroll --name Alex
```

The speaker reads a paragraph (~30–60 s of varied speech) → embedding
extracted → stored as their voiceprint centroid. **Text-independent**: a
general voiceprint, not tied to specific words.

## 6. Evaluation plan

This is the part that earns the grade — one metric per pipeline capability:

1. **Denoising quality** — SNR improvement (before vs. after), plus PESQ/STOI
   if we go deeper. Compare denoised vs. raw audio at a few noise levels.
   → `eval/denoise_quality.py`
2. **Diarization accuracy** — Diarization Error Rate (DER): predicted "who
   spoke when" vs. manually labeled ground truth on a held-out multi-speaker
   recording. → `eval/der.py`
3. **Speaker ID accuracy** — for enrolled speakers, % of diarized segments
   matched to the right name, plus a **false-accept check**: does an
   unenrolled voice ever get wrongly matched to an enrolled name?
   → `eval/speaker_id.py`
4. **Transcription accuracy** — WER, reported **per noise level** (clean vs.
   noisy source) to show the denoising step's actual contribution to
   transcript quality. This is the key "does denoising even help" result.
   → `eval/wer.py`

**Ablation:** run stages 2–4 with and without the denoising step (skip stage
1) to isolate what denoising actually buys us. Every eval script compares
`--backends none <denoiser>` by default, so this chart comes for free.

## 7. Repo layout

```
src/
  denoise.py       stage 1
  diarize.py       stage 2
  encoder.py       stage 3 — ECAPA speaker embeddings (reused)
  speaker_id.py    stage 3 — match clusters to enrolled voiceprints
  stt.py           stage 4 — Whisper wrapper (reused)
  merge.py         stage 5 — build subtitle files
  video.py         stage 5 — put audio + subtitles back on the video
  enroll.py        enrollment CLI (reused)
  profiles.py      voiceprint database
  main.py          full pipeline entry point
  demo_ui.py       Gradio demo
  check_env.py     setup checker
  audio_io.py, vad.py, segments.py, config.py   shared helpers
eval/
  denoise_quality.py   metric 1
  der.py               metric 2 — diarization error rate
  speaker_id.py        metric 3
  wer.py               metric 4
  labels_to_rttm.py    Audacity labels → RTTM ground truth
data/                  (gitignored — never committed)
  profiles.db          enrolled voiceprints — biometric data, keep local
  eval/                held-out test clips + ground-truth labels
tests/                 unit tests for the model-free logic
```

## 8. Known limitations (state these openly in the report)

- **Overlapping speech.** Diarization struggles when two people talk at once —
  most pipelines pick one speaker or split awkwardly. Acknowledge it rather
  than hiding a weird test-clip result. (Our `ecapa_cluster` baseline can't
  represent overlap at all; pyannote 3.1 can, partially.)
- **Enrollment quality.** Speaker ID accuracy depends entirely on it — short or
  noisy enrollment paragraphs hurt matching later (same caveat as the
  attendance project's mic-mismatch issue).
- **Denoising can distort voice characteristics** along with removing noise.
  Measure its effect on speaker ID accuracy specifically, not just on how clean
  it sounds (`eval/speaker_id.py` compares denoised vs. raw).
- **Unenrolled speakers are labeled generically** ("Speaker 2"). The system can
  separate voices it has never met but can't name them without enrollment.
  State this explicitly so it isn't read as a bug.

**Constraint recap:** APIs are allowed, but every stage uses an open-source,
locally-run model so the algorithm can be explained. Voiceprints are biometric
data, stored locally only (`data/profiles.db`, gitignored).

---

## 9. Implementation decisions (made during repo setup)

| Decision | Why |
|---|---|
| **Python 3.10 / 3.11 only** | DeepFilterNet publishes Windows wheels only up to 3.11; on 3.12+ pip tries to compile it from Rust source and fails. |
| **Pinned torch 2.5.1, numpy < 2, huggingface_hub < 1.0** | A known-good combination for pyannote 3.3.2 + SpeechBrain 1.0 + DeepFilterNet 0.5.6. DeepFilterNet requires numpy < 2; huggingface_hub 1.x removed an argument pyannote 3.x still uses. |
| **DeepFilterNet over Demucs** | DeepFilterNet is built for speech (full-band 48 kHz, real-time on CPU). Demucs is a music source separator — its "vocals" stem keeps any voice-like sound — and its pip package pins torchaudio < 2.1, which conflicts with the rest of the stack. |
| **Extra `spectral` denoiser (noisereduce)** | A classical, non-DL baseline, so the report can show what the neural model adds over plain spectral gating. |
| **pyannote 3.1 (not 4.x)** | 4.x decodes audio through torchcodec, which needs FFmpeg shared DLLs on Windows. We pass the waveform in memory to 3.1 instead. |
| **Extra `ecapa_cluster` diarizer** | Our own simple baseline (VAD → sliding windows → ECAPA → agglomerative clustering). It's fully explainable, gives the report a comparison point for DER, and works without a HuggingFace token. |
| **Speaker ID per cluster, one-to-one (Hungarian)** | Averaging all of a cluster's segments gives a steadier embedding than scoring each short segment alone. One-to-one assignment stops two clusters from both being named "Alex". Accept only if cosine ≥ `speaker.tau`, tuned with `eval/speaker_id.py`. |
| **Word-level speaker assignment (WhisperX-style) by default** | Transcribing the full track once gives Whisper full context. Each word goes to the diarized turn it overlaps most. `--strategy segment` (one Whisper call per turn) is kept as an alternative to compare. |
| **Configurable source track per stage** (`pipeline.*_on`) | Because of limitation 3, speaker ID might work better on the raw track than the denoised one. That's a measurable question, not an assumption. |
| **ffmpeg via `imageio-ffmpeg`** | Bundles an ffmpeg binary through pip, so nobody on the team has to install ffmpeg by hand to process video. |
| **Audacity labels as the ground-truth format** | The easiest free way to hand-label "who spoke when". The eval scripts read the `.txt` export directly, or convert it to RTTM. |
