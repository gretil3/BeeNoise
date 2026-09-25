# data/ — local only, never committed

Everything in this folder except this README is gitignored. Recordings are
personal data and voiceprints are **biometric** data. Share eval data through
the team drive, not git.

```
data/
  profiles.db                 enrolled voiceprints (created by src.enroll)
  enroll/<Name>/enrollment.wav   copy of each enrollment reading
  eval/
    noise/                    background-noise recordings (any length)
      cafe.wav  traffic.wav  fan.wav ...
    wer/                      clean, single-speaker read speech + transcripts
      clip01.wav  clip02.wav ...
      transcripts.json
    diarization/              held-out multi-speaker recordings + ground truth
      meeting1.mp4   meeting1.txt   (Audacity labels)  — or meeting1.rttm
      vlog1.wav      vlog1.rttm
```

## What to record (suggested split: each teammate contributes some)

**`eval/noise/`**: 5–10 noise-only recordings, 30 s+ each: cafe/canteen,
street traffic, fan/AC hum, keyboard typing, rain, crowd babble. Phone
recordings are fine.

**`eval/wer/`**: 20–40 clean clips of 5–15 s read speech, several speakers,
recorded in a quiet room. `transcripts.json` maps each filename to exactly
what was said:

```json
{
  "clip01.wav": "the quick brown fox jumps over the lazy dog",
  "clip02.wav": "please call stella and ask her to bring these things"
}
```

Case and punctuation don't matter (they're normalised away). These same
clips are the clean references for `eval/denoise_quality.py`. The scripts mix
in the noise files themselves at 0/5/10/20 dB, so record them **clean**.

**`eval/diarization/`**: 3–5 real multi-speaker recordings, 2–5 min each,
ideally naturally noisy (canteen chat, a group discussion, a vlog-style walk).
Include at least one person who is **not enrolled** so the false-accept check
has data. Label each one in Audacity:

1. Open the file, `Edit > Labels > Add Label at Selection` (Ctrl+B) over each
   stretch of speech.
2. Label text = speaker. Use the **exact enrolled name** for enrolled people
   (`Alex`) and anything else for non-enrolled people (`guest1`).
3. Overlapping speech: label both people over the overlap.
4. `File > Export > Export Labels...` → save next to the recording with the
   same name, `.txt` extension.

Don't enroll from the same recordings you evaluate on. The eval recordings
must be *held out*.
