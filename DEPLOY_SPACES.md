# Deploying BeeNoise as a Hugging Face Space

A permanent, public, free URL for the same app the Colab notebook and
`python -m src.demo_ui` already run — no session to keep alive, no tunnel
link that expires. This is a genuine deployment, not a demo trick, but read
[the two tradeoffs](#tradeoffs-of-the-free-tier) below before relying on it.

## 1. Create the Space

1. Go to https://huggingface.co/new-space (log in first).
2. **Space name**: `beenoise` (or anything).
3. **SDK**: Gradio.
4. **Hardware**: CPU basic — free.
5. Click **Create Space**. You now have an empty git repository at
   `https://huggingface.co/spaces/<your-username>/beenoise`.

## 2. Add your HuggingFace token as a secret

Same token the local setup and Colab notebook use, for the same reason
(pyannote diarization is gated).

1. On your new Space's page: **Settings** → **Variables and secrets** → **New secret**.
2. Name: `HF_TOKEN`. Value: your token (the same one from `.env` locally —
   copy it from there, don't make a new one).
3. Save.

This makes `HF_TOKEN` available to the app as a normal environment variable
at runtime. Nothing in the code needs to change for this — `src/config.py`
already reads `HF_TOKEN` from the environment first, before falling back to
a local `.env` file (which won't exist on the Space, and that's fine).

## 3. Push this repo to the Space

From your local clone of this repo (the same one you've been running):

```bash
git remote add space https://huggingface.co/spaces/<your-username>/beenoise
git push space main
```

It'll ask for your HuggingFace username and a token with **write** access as
the password (create one at https://huggingface.co/settings/tokens if
you don't have one — this can be the same `HF_TOKEN` if you made it with
write scope, or a separate one).

That's it — no file-swapping, no separate config. The root `README.md`
already carries the metadata block (`sdk: gradio`, `app_file: app.py`,
`python_version: "3.10"`, ...) that Spaces reads to know how to build and
run this repo, and `requirements.txt` bootstraps everything — including a
CPU build of torch — from a single `pip install -r requirements.txt`, which
is all a Space's build step does.

## 4. Wait for the build, then open it

The Space's page shows build logs live. First build installs ~1-2GB of
dependencies and can take several minutes — slower than your own machine,
comparable to the Colab notebook's first run. Once it says "Running," open
the Space's URL and you'll see the same Gradio UI as the local demo:
Transcribe, Enroll, Speakers tabs.

## Updating it later

```bash
git push space main
```

Any commit you push to `main` on GitHub, you can also push to `space` the
same way, any time. The Space rebuilds automatically.

## Tradeoffs of the free tier

Both of these are settings you can change later from the Space's
**Settings** tab, without touching any code — nothing here is a one-way
door.

- **CPU only.** No GPU on the free tier — expect the same per-clip speed
  you saw on the Colab demo without a GPU (minutes, not seconds). A paid
  GPU tier is available in Settings → Hardware if this matters for how
  you're using it.
- **Storage resets when the Space goes to sleep.** A CPU Space with no
  visitors for a while sleeps, and its disk resets when it wakes back up on
  the next visit — so `data/profiles.db` (enrolled speakers) and anything in
  `outputs/` don't survive that. Fine for showing people a live demo; not
  fine if you want speakers to stay enrolled indefinitely. Settings →
  **Persistent storage** (small monthly cost) fixes this — enable it, then
  point `paths.db` / `paths.enroll_dir` / `paths.output_dir` in
  `config.yaml` at the mounted persistent directory (Spaces mounts it at
  `/data`) instead of the repo-relative defaults.

## If the build fails

Read the actual error in the Space's build log first — it's almost always
one of:

- **A missing/wrong `HF_TOKEN`** → diarization fails at runtime (not build
  time), same fix as the Colab notebook: check the secret is set and that
  the account behind it accepted both gated pyannote model pages.
- **Out of memory during build or at runtime** → CPU basic is 16GB, which
  should be enough; if it isn't, that's a Settings → Hardware upgrade, not a
  code problem.
- Anything else → paste the log here and we'll debug it the same way we
  debugged the Colab notebook.
