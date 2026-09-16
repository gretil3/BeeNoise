# Student Attendance via Voice Verification — Build Blueprint

A local-first attendance system that enrolls each student's voiceprint once,
then verifies attendance with **two-factor voice authentication**: (1) does
this voice match the student who claims to be present, and (2) did they
correctly read back a randomly generated digit challenge generated for this
exact attempt. Both must pass — that combination is what makes the system
resistant to a simple "play back a recording of the real student" attack,
not just a speaker-ID demo.

---

## 0. Architecture at a glance

```
                   ENROLLMENT (offline, once per student)
   mic -> VAD trim -> 8-10 varied phrases -> embed each -> L2-norm ->
   mean = centroid -> profiles.db (name, student_id, 192-d vector)


                   ATTENDANCE CHECK-IN (per student, per session)

   +-------------+     +--------------------+     +------------------+
   | student     | --> | system generates a | --> | student reads it |
   | claims name |     | random digit       |     | aloud into mic   |
   | from roster |     | challenge, e.g.    |     +--------+---------+
   +-------------+     | "8-2-4-9"          |               |
                        +--------------------+               v
                                                     +-------------------+
                                                     | capture -> VAD -> |
                                                     | 16kHz mono float32|
                                                     +---------+---------+
                                                               |
                                      +------------------------+------------------------+
                                      v                                                 v
                          +---------------------+                          +----------------------+
                          | speaker encoder      |                          | STT (digit-biased)   |
                          | (ECAPA-TDNN) -> emb  |                          | -> transcribed digits|
                          +----------+-----------+                          +-----------+-----------+
                                     |                                                  |
                                     v                                                  v
                         cosine(emb, claimed_user.centroid)              transcript == today's challenge?
                                     |                                                  |
                                     +--------------------+---------------------------+
                                                            v
                                         BOTH pass?  --yes-->  ATTENDANCE VERIFIED
                                              |
                                              no
                                              v
                               fail -> retry (up to attendance.max_attempts)
                                       -> then lockout for attendance.lockout_minutes
```

**Two independent models run on the same audio buffer.** The speaker
encoder and the ASR model do *not* share features — feed both the same
16 kHz mono float32 array.

**Verification is 1:1, not open-set identification.** The student claims an
identity first (picked from the roster), so the system only ever compares
against that one enrolled centroid — no top1-vs-top2 margin logic needed,
just a single cosine threshold (`speaker.tau`).

---

## 1. Tech stack

| Layer | Pick | Why |
|---|---|---|
| Python | **3.12** (`py -3.12`) | 3.7 is dead; 3.14 has no stable torch wheels |
| Audio capture | `sounddevice` | callback-based, no PyAudio build pain on Windows |
| VAD / endpointing | `webrtcvad-wheels` | prebuilt wheel, no C++ toolchain needed on Windows |
| Speaker embedding | **SpeechBrain `spkrec-ecapa-voxceleb`** (192-d) | strong baseline, ~1% EER on VoxCeleb1 |
| Vector store | SQLite + `numpy` blob | a class roster — don't reach for FAISS |
| STT (digit readback) | **`faster-whisper`** `small`, `compute_type="int8"` | several× faster than openai-whisper on CPU; digit-biased via `initial_prompt` |
| UI | CLI first (`src/main.py`), then Gradio (`src/demo_ui.py`) | Gradio gives a mic widget + roster/log view for free |

Deliberately **not** in this project: an LLM, TTS, conversational memory.
Those belonged to an earlier version of this repo (a voice-aware chat
assistant) and added surface area with no role in an attendance check-in.
The challenge is shown as on-screen text, not spoken aloud — keeps scope
tight; a teammate can add TTS readout later without touching the
verification logic.

---

## 2. Repo layout

```
voice_recognition/
├─ BLUEPRINT.md
├─ requirements.txt
├─ config.yaml              # thresholds, model names, digit-challenge params
├─ data/
│  ├─ profiles.db           # SQLite: students, embeddings, attendance log
│  ├─ enroll/<student>/*.wav
│  └─ eval/                 # held-out utterances for the EER report
├─ src/
│  ├─ audio_io.py           # record(), play(), resample to 16k mono
│  ├─ vad.py                # trim silence, detect end-of-utterance
│  ├─ encoder.py            # embed(wav) -> np.ndarray[192], L2-normalised
│  ├─ enroll.py             # CLI: python -m src.enroll --name Alex
│  ├─ verify.py             # verify_claim(emb, claimed_user) -> pass/fail + score
│  ├─ stt.py                # transcribe(wav) -> text; transcribe_digits() for readback
│  ├─ challenge.py          # random digit generator + transcript normalization/matching
│  ├─ profiles.py           # SQLite CRUD: students + attendance log/state
│  ├─ attendance.py         # combined verification + retry/lockout state machine
│  └─ main.py               # attendance CLI
├─ eval/
│  ├─ eer.py                # DET curve, EER, threshold picker (speaker verification)
│  └─ wer.py                # jiwer-based WER (digit-readback ASR accuracy)
└─ notebooks/                # figures for the report
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
pip install jiwer matplotlib scikit-learn gradio
```

Smoke test: record 3 s, save a WAV, play it back. **Do not move on until the
mic works** — every downstream bug looks like a model bug and is actually a
sample-rate bug.

Canonical audio format for the whole project: **16 000 Hz, mono, float32 in
[-1, 1]**. Convert at the boundary, never in the middle.

---

### Phase 1 — Audio I/O + VAD (`audio_io.py`, `vad.py`)

Unchanged from a general voice pipeline: record a fixed-length clip for
enrollment/attendance prompts, trim silence before embedding. Speaker
verification degrades hard below ~2 s of *net speech* — a 4-digit challenge
read at a normal pace comfortably clears the `min_net_speech_sec` floor, but
don't shrink the digit count much below that without re-checking.

---

### Phase 2 — Speaker encoder (`encoder.py`)

Same ECAPA-TDNN embed()/cosine() as any speaker-verification project.
Normalise at the source (L2-norm) so cosine similarity downstream is just a
dot product. Load the model once at startup.

---

### Phase 3 — Enrollment (`enroll.py`)

1. Prompt for 8–10 short, phonetically varied phrases (Harvard-sentence
   style) — **not** digits. A voiceprint built only from digit-reading would
   model "this person reading numbers," not their voice in general, and
   would generalize worse. Enroll on prose, verify on digits.
2. Record → VAD-trim → embed each → drop any clip whose embedding scores
   cosine < 0.5 against the running mean (that one was a cough).
3. Centroid = mean of kept embeddings, re-normalised to unit length.
   Persist `(name, student_id, embedding, n_utts, spread, created_at)` —
   see `profiles.py` for the schema.
4. Also store each student's **intra-speaker spread** (mean cosine of their
   utterances to their own centroid) — used for threshold analysis and a
   good report figure.

---

### Phase 4 — Digit challenge (`challenge.py`)

- `generate(n_digits)` — a fresh random digit string per attempt (default 4
  digits, `attendance.n_digits`). Never reused across attempts; that's the
  entire anti-replay property.
- `normalize_transcript(text)` — maps an ASR transcript to a canonical
  digit string, handling both digit characters ("8 2 4 9") and number words
  ("eight two four nine") since Whisper is inconsistent about which form it
  emits for short number sequences.
- `matches(expected, transcript, tolerance)` — exact match by default
  (`attendance.digit_edit_distance_tolerance: 0`); can be loosened to an
  edit-distance budget if ASR noise turns out to reject too many honest
  attempts (tune this from the Phase 8 digit-accuracy numbers, don't guess).

---

### Phase 5 — Verification (`verify.py`)

1:1 claim verification, not open-set identification:

```python
def verify_claim(emb, claimed_user):
    score = cosine(emb, claimed_user.centroid)
    return VerifyResult(claimed_user, score, score >= CFG["speaker"]["tau"])
```

**Do not hardcode `tau = 0.30` because a blog post said so.** Derive it in
Phase 8 and cite your own number.

---

### Phase 6 — STT for the readback (`stt.py`)

`transcribe_digits()` calls `faster-whisper` with an `initial_prompt`
biasing decoding toward digit vocabulary — free-form decoding otherwise
sometimes turns a short, low-context digit string into an unrelated short
word. Gate on confidence (`min_avg_logprob`, `max_no_speech_prob`) same as
any ASR call; a garbled transcript should fail content-verification, not
get force-matched.

---

### Phase 7 — Combined decision + retry/lockout (`attendance.py`)

```
attempt(claimed_user, session, expected_digits, audio):
    if locked out for (claimed_user, session): reject, report until when
    if already marked present this session: short-circuit success
    speaker_pass   = verify_claim(embed(audio), claimed_user).passed
    content_pass   = challenge.matches(expected_digits, transcribe_digits(audio))
    passed = speaker_pass AND content_pass
    passed  -> mark_present, reset retry state
    failed  -> increment fail_count; at attendance.max_attempts, set a
               lockout until now + attendance.lockout_minutes
```

Every attempt (pass or fail) is logged to `attendance_attempts` — that log
*is* the evaluation dataset for Phase 8. `attendance_log` holds one row per
(student, session): the first successful check-in.

The lockout is intentionally simple (a timestamp in SQLite, not a job
queue) — fine for a classroom-scale roster. The Gradio demo exposes a
"skip cooldown" button so a live demo doesn't have to sit through 15
minutes; a real deployment must not expose that to students.

---

### Phase 8 — Evaluation (this is what earns the grade)

**8a. Speaker verification — EER** (`eval/eer.py`, unchanged from a general
speaker-verification pipeline)
- Held-out set: ≥ 20 utterances per enrolled student, plus ≥ 3 non-enrolled
  impostors.
- Score every utterance against every centroid → target vs. non-target
  scores. Sweep tau, **EER is where FAR and FRR cross**, and that becomes
  the default `speaker.tau`.
- Deliverables: DET curve, EER %, score-distribution histogram.

**8b. Digit-readback accuracy** (`eval/wer.py`, repointed at digit clips)
- Record N students each reading M random challenges; compute digit-string
  accuracy (exact match) and WER as a secondary metric.
- Report false-reject-from-ASR separately from false-reject-from-speaker-ID
  — a failed attempt for the wrong reason is a different bug than a failed
  attempt for the right reason.

**8c. The interesting one — combined-system FAR under a replay-style attack**
- Take a genuine recording of student A correctly reading challenge X.
- Score it against student A's centroid using a *different* challenge Y
  (i.e., simulate someone replaying an old recording for a new challenge).
- Show: speaker-only verification accepts it (same voice); the combined
  system rejects it (wrong digits). This is the single best table/figure in
  the report — it's the whole reason the digit challenge exists.

**8d. Retry/lockout behavior**
- Confirm empirically that `max_attempts` failures trigger a lockout and
  that a locked-out student's attempts are rejected without even running
  the encoder/STT (cheap to verify, easy to demo going wrong if the state
  machine has an off-by-one).

**8e. End-to-end latency**
- Per-stage milliseconds: VAD → embed → STT → decision. A stacked bar chart
  is a good slide.

---

### Phase 9 — Demo UI (`demo_ui.py`)

Gradio, roughly three tabs: **Attendance** (pick name, generate challenge,
record, submit, see pass/fail + score), **Enroll New Student**, **Roster /
Attendance Log** (who's checked in this session, plus the raw enrolled-user
list). Showing the live speaker score and the matched/mismatched digits
makes the two-factor decision legible to whoever is grading it.

---

## 4. Known traps

1. **Sample-rate mismatch.** Mic runs at 44.1/48 kHz; ECAPA and Whisper
   both want 16 kHz. Resample once, at capture.
2. **Channel mismatch.** Enroll on a headset, test on a laptop mic, and EER
   triples. Enroll and test on the same device — and *say so* in the report.
3. **Digits are short.** Enforce the net-speech-seconds floor; a 2-digit
   challenge read quickly may not clear it. 4 digits at a normal pace does.
4. **Cosine on unnormalised vectors** quietly becomes a magnitude contest.
   Normalise at the source.
5. **Replay attack — now actually mitigated, not just named.** A plain
   recording of the real student's voice fails content verification because
   it won't contain the current session's digits. Still name the residual
   gap in the report: a *live* attacker who can hear the challenge and
   splice together pre-recorded digit clips in real time is not defended
   against; that's a genuine anti-spoofing (ASVspoof-style) problem, out of
   scope here.
6. **Privacy.** Voiceprints are biometric data. `profiles.delete_user(name)`
   removes a student's embedding and their attendance history; state in the
   report that everything stays local (`data/profiles.db`).
7. **First-run model downloads** (~80 MB ECAPA, ~480 MB Whisper `small`).
   Do not discover this five minutes before the demo on venue wifi.

---

## 5. Scope control

Minimum viable graded project = Phases 0–7 + 8a + 8b. Phase 8c (the
combined-system FAR experiment) is what elevates this from "a speaker-ID
demo" to "an attendance system with a stated, tested threat model" — worth
prioritizing over UI polish if time is short.

Stretch goals, in priority order: TTS readout of the challenge (a teammate
can build this against `attendance.new_challenge()` without touching
verification logic) → per-session CSV export of `attendance_log` for
instructors → adaptive `tau` per student from their own `spread` →
short-audio ("liveness") countermeasures against splicing attacks.
