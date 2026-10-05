"""Entry point for `gradio app.py` (hot reload while editing the UI).

The actual app is defined in src/demo_ui.py -- the same one `python -m src.demo_ui`
runs. This file only exists so the `gradio` reloader has a top-level script to
watch; it adds no behavior of its own.
"""
from src.demo_ui import demo, launch  # noqa: F401  `demo` lets `gradio app.py` hot-reload

if __name__ == "__main__":
    launch()
