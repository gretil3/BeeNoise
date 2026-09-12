"""Orchestrator: the full IDLE -> LISTENING -> PROCESSING loop.

    IDLE --speech detected--> LISTENING --endpoint--> PROCESSING
    PROCESSING: parallel { embed -> identify , transcribe }
                unknown speaker -> GUEST
                user changed    -> announce "Welcome back, {name}!"
                -> LLM -> TTS -> IDLE

Run: python -m src.main
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

from . import agent, profiles
from .audio_io import record_until_silence
from .config import CFG, path as cfg_path
from .encoder import embed
from .stt import transcribe_or_none
from .vad import net_speech_seconds
from .verify import identify, SpeakerTracker

VOICE_COMMANDS = {
    "who am i": "_cmd_who_am_i",
    "goodbye": "_cmd_goodbye",
    "forget this conversation": "_cmd_forget",
}


def _log_turn_event(event: dict):
    log_path = cfg_path("logs")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _cmd_who_am_i(tracker: SpeakerTracker, score: float):
    if tracker.active_user:
        return f"You're {tracker.active_user.name}."
    return "I'm not sure yet — I don't recognize your voice as an enrolled user."


def _cmd_forget(tracker: SpeakerTracker, score: float):
    if tracker.active_user:
        profiles.set_summary(tracker.active_user.id, "")
        return "Okay, I've cleared our conversation history."
    return "There's no history to clear for a guest."


def run():
    print("Voice-aware assistant starting up. Loading models (this can take a bit)...")
    executor = ThreadPoolExecutor(max_workers=2)
    tracker = SpeakerTracker()
    print("Ready. Speak into the mic (Ctrl+C to quit).\n")

    try:
        while True:
            audio = record_until_silence()
            if len(audio) == 0:
                continue

            t0 = time.time()
            net_speech = net_speech_seconds(audio)

            fut_embed = executor.submit(embed, audio)
            fut_text = executor.submit(transcribe_or_none, audio)
            emb = fut_embed.result()
            text = fut_text.result()
            t1 = time.time()

            if text is None:
                print("(low-confidence transcription, ignoring — please repeat)")
                continue

            if net_speech < CFG["speaker"]["min_net_speech_sec"]:
                result_user, score, margin = None, 0.0, 0.0
            else:
                result = identify(emb)
                result_user, score, margin = result.user, result.score, result.margin

            from .verify import IdentifyResult
            active_before = tracker.active_user
            active = tracker.update(IdentifyResult(result_user, score, margin, {}))

            if active and (active_before is None or active_before.id != active.id):
                print(f"[speaker switch] Welcome back, {active.name}! (score={score:.3f})")

            low = text.lower().strip().rstrip(".!?")
            handled = False
            for phrase, fn_name in VOICE_COMMANDS.items():
                if phrase in low:
                    reply = globals()[fn_name](tracker, score)
                    handled = True
                    break

            if not handled:
                reply = agent.respond(active, text)

            t2 = time.time()
            print(f"You ({active.name if active else 'guest'}, score={score:.2f}): {text}")
            print(f"Assistant: {reply}\n")

            try:
                from .tts import speak
                speak(reply)
            except Exception as e:
                print(f"(TTS unavailable: {e})")

            _log_turn_event({
                "ts": time.time(),
                "speaker_pred": active.name if active else None,
                "score": score,
                "margin": margin,
                "text": text,
                "reply": reply,
                "latency_embed_stt_ms": int((t1 - t0) * 1000),
                "latency_total_ms": int((t2 - t0) * 1000),
            })

    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    run()
