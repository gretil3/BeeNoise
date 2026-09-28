# BeeNoise — first run checklist

Full reference: [README.md](README.md) and [BLUEPRINT.md](BLUEPRINT.md). This
is the fast path, with the gotchas people actually hit while setting it up.

## 1. Get the code + environment

```bash
git clone https://github.com/gretil3/BeeNoise
cd BeeNoise
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt
```

⚠️ Must be Python 3.10 or 3.11. 3.12+ fails to install (no Windows wheel for
DeepFilterNet). Check with `py -0` if unsure which you have.

## 2. Verify everything installed

```bash
python -m src.check_env
```

Should end with "All good." apart from one `[warn]` about `HF_TOKEN` —
that's expected until step 3.

## 3. Get diarization working (HuggingFace token)

1. Make a free account at huggingface.co, then create a token:
   Settings → Access Tokens → New token → type **Read** → name it
   whatever → Create.
2. ⚠️ Copy the token now — it's only shown once.
3. Accept the terms on **both** of these pages (logged in as the same
   account), scroll down and click **"Agree and access repository"** on
   each. They ask for a company/university and a website — put the repo
   URL (`https://github.com/gretil3/BeeNoise`) as the website:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

   Missing either one of these two is the #1 reason diarization fails to
   load.
4. In the repo:

   ```bash
   copy .env.example .env
   ```

   Open `.env` and paste your token in place of the placeholder:

   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

5. Confirm it worked (also pre-downloads every model, ~1-2 GB, one time):

   ```bash
   python -m src.check_env --models
   ```

   Look for `[ok] pyannote loads and runs`.

## 4. Enroll yourself

```bash
python -m src.enroll --name YourName
```

Press Enter, read the paragraph out loud at a normal pace, press Enter
again to stop. ⚠️ If you get "Only 0.0s of speech detected", your mic
input wasn't picked up — check Windows' default recording device and try
again.

Check it saved:

```bash
python -m src.profiles list
```

## 5. Run it on a real file

⚠️ Use an actual path to a file **you** have — a phone video/voice memo
works great. Don't copy a path from a doc as-is; check it exists first:

```bash
dir "%USERPROFILE%\Downloads" | findstr /i "mp4 mp3 wav m4a mov"
```

Then:

```bash
python -m src.main "C:\full\path\to\your\clip.mp4" --num-speakers 1
```

(`--num-speakers N` = how many people talk in the clip, if you know it —
big accuracy boost. Drop it if unsure.)

Results land in `outputs\<filename>\` — open `subtitles.srt` to see the
speaker-labelled transcript, or the `_subtitled.mp4` if your input was a
video.

## 6. Easier: the web demo instead of the CLI

```bash
python -m src.demo_ui
```

Opens a browser tab at http://127.0.0.1:7860 — upload a file, see results
without any commands.

---

Stuck? Paste your terminal output (command + full output) into the group
chat — full output is what makes these fixable fast.
