"""Entry point for Hugging Face Spaces.

Spaces (Gradio SDK) looks for app.py at the repo root and runs it directly.
The actual app is defined in src/demo_ui.py -- same file used for the local
`python -m src.demo_ui`. This file only exists so
Spaces has something to run; it adds no behavior of its own.

See DEPLOY_SPACES.md for how to deploy this repo as a Space.
"""
from src.demo_ui import demo, launch  # noqa: F401  `demo` lets `gradio app.py` hot-reload

if __name__ == "__main__":
    launch()
