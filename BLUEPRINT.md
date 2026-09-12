# Voice-Aware Conversational Agent — Build Blueprint

A local-first assistant that (1) identifies *who* is speaking from a voiceprint,
(2) loads that person's profile, (3) transcribes what they said, and
(4) answers with an LLM that knows who it's talking to.

---

## 0. Architecture at a glance

```
                   ENROLLMENT (offline, once per user)
   mic -> VAD trim -> 8-10 utterances -> embed each -> L2-norm ->
   mean = centroid -> profiles.db (name, prefs, 192-d vector)


   RUNTIME LOOP
   +---------+   +---------+   +---------------+   +---------------------+
   | capture |-->|  VAD /  |-->| speaker enc.  |-->| cosine vs. every    |
   |   mic   |   | endpoint|   | (ECAPA-TDNN)  |   | enrolled centroid   |
   +---------+   +----+----+   +---------------+   +----------+----------+
                      |                                       |
                      |                              max sim >= tau ?
                      |                          +------------+------------+
                      |                        yes                         no
                      v                          v                          v
                +-----------+           +----------------+       +----------------+
                |    STT    |           | load profile   |       |  guest mode    |
                | f-whisper |           | + chat history |       |  offer enroll  |
                +-----+-----+           +--------+-------+       +----------------+
                      |                          |
                      +------------+-------------+
                                   v
                        +----------------------+      +-----+
                        | LLM (system prompt = |  ->  | TTS | -> speaker
                        |  profile + memory)   |      +-----+
                        +----------------------+
```

**Two independent models run on the same audio buffer.** The speaker encoder
and the ASR model do *not* share features — feed both the same 16 kHz mono
float32 array.

---

## 1. Tech stack

| Layer | Pick | Why |
|---|---|---|
| Python | **3.12** (`py -3.12`) | 3.7 is dead; 3.14 has no stable torch wheels |
| Audio capture | `sounddevice` | callback-based, no PyAudio build pain on Windows |
| VAD / endpointing | `silero-vad` (torch) or `webrtcvad` | silero = far fewer false triggers |
| Speaker embedding | **SpeechBrain `spkrec-ecapa-voxceleb`** (192-d) | strong baseline, ~1% EER on VoxCeleb1 |
| — lighter fallback | `resemblyzer` (256-d GE2E) | one pip install, no torch hub download |
| Vector store | SQLite + `numpy` blob | 3 users — don't reach for FAISS |
| STT | **`faster-whisper`** `small`/`base`, `compute_type="int8"` | several× faster than openai-whisper on CPU |
| LLM | Anthropic API (`claude-sonnet-5`) *or* Ollama (`llama3.1:8b`) fully offline | pick one, hide it behind a single function |
| TTS (optional) | `piper-tts` (good) or `pyttsx3` (zero setup, robotic) | |
| UI | CLI first, then Gradio | Gradio gives you a mic widget for free |

---

## 2. Repo layout

```
voice_recognition/
├─ BLUEPRINT.md
├─ requirements.txt
├─ config.yaml              # thresholds, model names, paths
├─ data/
│  ├─ profiles.db           # SQLite: users + embeddings
│  ├─ enroll/<user>/*.wav   # keep raw audio — you WILL re-embed
│  └─ eval/                 # held-out utterances for the EER report
├─ src/
│  ├─ audio_io.py           # record(), play(), resample to 16k mono
│  ├─ vad.py                # trim silence, detect end-of-utterance
│  ├─ encoder.py            # embed(wav) -> np.ndarray[192], L2-normalised
│  ├─ enroll.py             # CLI: python -m src.enroll --name Alex
│  ├─ verify.py             # identify(emb) -> (user_id | None, score)
│  ├─ stt.py                # transcribe(wav) -> text, confidence
│  ├─ profiles.py           # SQLite CRUD + conversation memory
│  ├─ agent.py              # build system prompt, call LLM
│  ├─ tts.py
│  └─ main.py               # the state machine
├─ eval/
│  ├─ eer.py                # DET curve, EER, threshold picker
│  └─ wer.py                # jiwer-based WER on your own recordings
└─ notebooks/               # figures for the report
```

---

## 3. Phase by phase

### Phase 0 — Environment (30 min)

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install sounddevice soundfile numpy scipy pyyaml
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install speechbrain faster-whisper
pip install anthropic jiwer matplotlib
```

Smoke test: record 3 s, save a WAV, play it back. **Do not move on until the
mic works** — every downstream bug looks like a model bug and is actually a
sample-rate bug.

Canonical audio format for the whole project: **16 000 Hz, mono, float32 in
[-1, 1]**. Convert at the boundary, never in the middle.

---

### Phase 1 — Audio I/O + VAD (`audio_io.py`, `vad.py`)

- `record_until_silence()`: stream 30 ms frames, buffer while speech is
  detected, stop after ~800 ms of trailing silence, reject clips under 1.0 s.
- Keep a ~300 ms pre-roll ring buffer and prepend it, or you clip the first
  phoneme of every utterance.
- Return the trimmed array *and* write the WAV to disk for debugging.

Rule of thumb: speaker verification degrades hard below ~2 s of *net speech*.
Silence-trim **before** embedding, not after.

---

### Phase 2 — Speaker encoder (`encoder.py`)

```python
from speechbrain.inference.speaker import EncoderClassifier
import numpy as np, torch

_enc = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/ecapa")

def embed(wav16k: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(wav16k).float().unsqueeze(0)
    e = _enc.encode_batch(t).squeeze().detach().numpy()   # (192,)
    return e / np.linalg.norm(e)                          # L2-normalise
```

Normalising here means cosine similarity later is just a dot product.
Load the model **once** at startup (~2 s), never per utterance.

---

### Phase 3 — Enrollment (`enroll.py`)

1. Prompt for 8–10 short phrases, ~4 s each. Use *varied, phonetically
   diverse* sentences (Harvard sentences work well). Do **not** have the user
   repeat one phrase ten times — you would be modelling the phrase, not the
   voice.
2. Record → VAD-trim → embed each → drop any clip whose embedding scores
   cosine < 0.5 against the running mean (that one was a cough).
3. Centroid = mean of the kept embeddings, re-normalised to unit length.
4. Persist:

```sql
CREATE TABLE users(
  id INTEGER PRIMARY KEY,
  name TEXT UNIQUE,
  embedding BLOB,            -- np.float32 .tobytes()
  n_utts INTEGER,
  prefs_json TEXT,           -- language, tone, interests, pronouns
  created_at TEXT);

CREATE TABLE turns(
  id INTEGER PRIMARY KEY,
  user_id INTEGER, role TEXT, text TEXT, score REAL, ts TEXT);
```

Also store each user's **intra-speaker spread** (mean cosine of their
utterances to their own centroid). You will use it for per-user thresholds,
and it makes a good figure.

---

### Phase 4 — Verification (`verify.py`)

```python
def identify(emb, users, tau=0.30, min_margin=0.05):
    scores = {u.name: float(emb @ u.centroid) for u in users}  # both unit-norm
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, s = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    if s < tau or (s - runner_up) < min_margin:
        return None, s          # unknown or ambiguous -> guest mode
    return best, s
```

Three decision knobs, and the write-up should discuss all three:

| Knob | What it protects against |
|---|---|
| `tau` (absolute threshold) | unknown speakers (open-set identification) |
| `margin` (top-1 minus top-2) | confusable enrolled users |
| temporal smoothing | one noisy turn flipping the active profile |

Temporal smoothing — require two consecutive agreeing turns before switching
users — is what makes the live demo feel solid.

**Do not hardcode `tau = 0.30` because a blog post said so.** Derive it in
Phase 8 and cite your own number.

---

### Phase 5 — STT (`stt.py`)

```python
from faster_whisper import WhisperModel
import numpy as np

_m = WhisperModel("small", device="cpu", compute_type="int8")

def transcribe(wav16k):
    segs, info = _m.transcribe(wav16k, language="en", beam_size=5,
                               vad_filter=True)
    segs = list(segs)
    text = " ".join(s.text for s in segs).strip()
    conf = float(np.mean([s.avg_logprob for s in segs])) if segs else -9.9
    return text, conf
```

- `base` is roughly real-time on CPU; `small` is 2–3× slower and noticeably
  more accurate. Benchmark both and put the table in the report.
- Gate on confidence: if `avg_logprob < -1.0` or `no_speech_prob > 0.6`, ask
  the user to repeat rather than feeding garbage to the LLM.
- Run STT and the encoder **in parallel** (`ThreadPoolExecutor`) — they are
  independent, and this roughly halves perceived latency.

---

### Phase 6 — Conversational agent (`agent.py`, `profiles.py`)

Build the system prompt from the identified profile:

```
You are a voice assistant. You are speaking with {name}.
Profile: {prefs}
Recent conversation summary: {summary}
Rules: answer in 1-3 sentences. You are read aloud by TTS, so no markdown,
no lists, no code blocks. If confidence about who is speaking is low, ask
them to confirm their name before referring to anything personal.
```

Memory: keep the last 10 turns verbatim per user in `turns`, and roll
everything older into a one-paragraph summary refreshed every 10 turns.
Per-user isolation is the entire point — **never** let user A's history reach
user B's context.

Guest path: unknown speaker → generic persona, nothing stored, offer
"Would you like to create a profile?" which drops into Phase 3.

---

### Phase 7 — Orchestrator (`main.py`)

```
IDLE --speech detected--> LISTENING --endpoint--> PROCESSING
PROCESSING: parallel { embed -> identify , transcribe }
            unknown speaker      -> GUEST
            user changed         -> announce "Welcome back, {name}!"
            -> LLM -> TTS -> IDLE
```

Voice commands worth wiring: "enroll me", "who am I", "switch user",
"forget this conversation", "goodbye".

Log every turn as JSON — `ts, speaker_pred, score, margin, text, latency_ms`.
That log *is* your evaluation dataset.

---

### Phase 8 — Evaluation (this is what earns the grade)

A Speech Recognition class wants numbers, not a demo. Budget real time here.

**8a. Speaker verification — EER**
- Held-out set: ≥ 20 utterances per enrolled speaker, plus ≥ 3 non-enrolled
  impostors.
- Score every utterance against every centroid → *target* scores (same
  speaker) and *non-target* scores (different speaker).
- Sweep tau from −1 to 1, plot FAR and FRR; **EER is where they cross**, and
  that crossing point becomes your default `tau`.
- Deliverables: DET curve, EER %, score-distribution histogram, confusion
  matrix at the chosen tau.

**8b. Ablations that make the report interesting**
- EER vs. number of enrollment utterances (1, 3, 5, 10) → justifies your 10.
- EER vs. utterance duration (1 s, 2 s, 4 s, 8 s).
- EER clean vs. noisy (mix in fan/café noise at +10 dB and 0 dB SNR).
- ECAPA vs. Resemblyzer under the identical protocol.

**8c. ASR — WER**
- Read 30 sentences with known transcripts; compute WER with `jiwer` after
  normalising case and punctuation.
- WER for `tiny`/`base`/`small` × (quiet, noisy) = a six-cell table.

**8d. End-to-end latency**
- Milliseconds per stage: VAD → embed → STT → LLM → TTS. A stacked bar chart
  of this is the single best slide in the presentation.

---

### Phase 9 — Demo UI

Gradio, roughly 60 lines: mic input, a "who's speaking" badge showing the
similarity score, the transcript, the reply, and an "Enroll new user" tab.
Displaying the live score makes the verification legible to whoever grades it.

---

## 4. Suggested timeline (6 weeks, ~6 h/week)

| Week | Deliverable |
|---|---|
| 1 | Env + mic + VAD; WAVs landing on disk |
| 2 | Encoder + enrollment for 3 people; embeddings in SQLite |
| 3 | Verification + a first EER number (even a bad one) |
| 4 | faster-whisper integrated; WER measured |
| 5 | LLM + profiles + memory + orchestrator; demo runs end to end |
| 6 | Ablations, figures, report, slides. Freeze the code Friday of week 5. |

---

## 5. Known traps

1. **Sample-rate mismatch.** Your mic runs at 44.1/48 kHz; ECAPA and Whisper
   both want 16 kHz. Resample once, at capture.
2. **Channel mismatch.** Enroll on a headset, test on a laptop mic, and EER
   triples. Enroll and test on the same device — and *say so* in the report.
   It is a real limitation, not a bug to hide.
3. **Text-independent is not content-free.** "yes" and "ok" are too short to
   trust. Enforce a ≥ 1.5 s net-speech minimum before accepting an ID.
4. **Cosine on unnormalised vectors** quietly becomes a magnitude contest.
   Normalise at the source.
5. **Replay attack.** A recording of Alex passes verification. Name it as a
   limitation; anti-spoofing (ASVspoof-style countermeasures) is out of scope,
   but naming it shows you understand the threat model.
6. **Privacy.** Voiceprints are biometric data. Ship a `--delete-user` command
   and state in the report that embeddings never leave the machine.
7. **First-run model downloads** (~80 MB ECAPA, ~480 MB Whisper `small`).
   Do not discover this five minutes before the demo on venue wifi.

---

## 6. Scope control

Minimum viable graded project = Phases 0–5 + 8a + 8c. The LLM layer is the
easiest part of this system; do **not** let prompt-tuning eat the time budget
that belongs to the evaluation section.

Stretch goals, in priority order: streaming with barge-in → diarization for
two speakers in one clip → per-user adaptive thresholds → wake word
("Hey Aria") → speaker adaptation that updates the centroid from
confidently-verified live turns, with a drift guard.
