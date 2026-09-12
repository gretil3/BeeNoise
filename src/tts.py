"""Text-to-speech. pyttsx3 is offline and zero-setup (uses SAPI5 on
Windows) — swap in piper-tts later for better quality if desired."""
from .config import CFG

_engine = None


def _get_engine():
    global _engine
    if _engine is None and CFG["tts"]["engine"] == "pyttsx3":
        import pyttsx3
        _engine = pyttsx3.init()
    return _engine


def speak(text: str):
    if not text:
        return
    if CFG["tts"]["engine"] == "pyttsx3":
        engine = _get_engine()
        engine.say(text)
        engine.runAndWait()
    else:
        raise NotImplementedError(f"TTS engine '{CFG['tts']['engine']}' not wired up yet")


if __name__ == "__main__":
    speak("Hello! This is a test of the text to speech engine.")
