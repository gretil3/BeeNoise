"""LLM layer: builds a per-user system prompt from profile + memory, calls
the configured provider, and manages rolling conversation summarization.

Swap providers by changing config.yaml -> llm.provider; callers never touch
the provider-specific API.
"""
import os

from . import profiles
from .config import CFG

SYSTEM_TEMPLATE = """You are a voice assistant. You are speaking with {name}.
Profile notes: {prefs}
Summary of earlier conversation: {summary}

Rules: answer in 1-3 sentences. Your reply will be read aloud by
text-to-speech, so do not use markdown, bullet lists, or code blocks.
If you are unsure who you're speaking with, ask them to confirm their name
before referencing anything personal."""

GUEST_SYSTEM = """You are a voice assistant speaking with an unidentified guest.
Be friendly and generic. After a couple of exchanges, offer: "Would you like
me to set up a profile for you?" Do not assume any personal details.
Rules: answer in 1-3 sentences, no markdown (you are read aloud by TTS)."""


def _build_system_prompt(user: profiles.User | None) -> str:
    if user is None:
        return GUEST_SYSTEM
    summary = profiles.get_summary(user.id) or "(no prior conversation yet)"
    prefs = user.prefs or {}
    prefs_str = ", ".join(f"{k}={v}" for k, v in prefs.items()) or "(none set)"
    return SYSTEM_TEMPLATE.format(name=user.name, prefs=prefs_str, summary=summary)


def _call_anthropic(system: str, messages: list[dict]) -> str:
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    resp = client.messages.create(
        model=CFG["llm"]["model"],
        max_tokens=200,
        system=system,
        messages=messages,
    )
    return resp.content[0].text.strip()


def _call_ollama(system: str, messages: list[dict]) -> str:
    import requests
    prompt_messages = [{"role": "system", "content": system}] + messages
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={"model": CFG["llm"]["ollama_model"], "messages": prompt_messages, "stream": False},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _call_llm(system: str, messages: list[dict]) -> str:
    provider = CFG["llm"]["provider"]
    if provider == "anthropic":
        return _call_anthropic(system, messages)
    elif provider == "ollama":
        return _call_ollama(system, messages)
    raise ValueError(f"Unknown llm.provider: {provider}")


def _maybe_summarize(user: profiles.User):
    """Roll older turns into a running summary every N turns, so context
    doesn't grow unbounded and user A's history never leaks into user B."""
    turns = profiles.recent_turns(user.id, limit=1000)
    if len(turns) < CFG["llm"]["summarize_every"]:
        return
    max_keep = CFG["llm"]["max_history_turns"]
    if len(turns) <= max_keep:
        return

    to_summarize = turns[:-max_keep]
    text_blob = "\n".join(f"{t['role']}: {t['text']}" for t in to_summarize)
    prior_summary = profiles.get_summary(user.id)

    system = ("Summarize this conversation history in one short paragraph, "
              "preserving names, preferences, and any commitments made. "
              "Merge with the prior summary if given.")
    messages = [{"role": "user", "content":
                 f"Prior summary: {prior_summary}\n\nNew turns:\n{text_blob}"}]
    new_summary = _call_llm(system, messages)
    profiles.set_summary(user.id, new_summary)


def respond(user: profiles.User | None, user_text: str) -> str:
    """Main entry point: given the identified user (or None for guest) and
    their transcribed utterance, return the assistant's reply text."""
    system = _build_system_prompt(user)

    if user is not None:
        history = profiles.recent_turns(user.id, limit=CFG["llm"]["max_history_turns"])
        messages = [{"role": t["role"], "content": t["text"]} for t in history]
    else:
        messages = []

    messages.append({"role": "user", "content": user_text})

    reply = _call_llm(system, messages)

    if user is not None:
        profiles.log_turn(user.id, "user", user_text)
        profiles.log_turn(user.id, "assistant", reply)
        _maybe_summarize(user)

    return reply


if __name__ == "__main__":
    # Quick text-only test, no audio needed.
    name = input("Enter an enrolled user's name (or blank for guest): ").strip()
    user = profiles.get_user_by_name(name) if name else None
    if name and user is None:
        print(f"No such user '{name}', proceeding as guest.")
        user = None
    print("Type messages, Ctrl+C to quit.")
    while True:
        text = input("> ")
        print(respond(user, text))
