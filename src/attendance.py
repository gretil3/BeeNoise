"""Attendance verification flow — shared between the CLI (main.py) and the
Gradio demo (demo_ui.py) so both drive the exact same decision logic.

Two checks must BOTH pass to mark a student present:
  1) speaker verification — cosine(voice, claimed student's centroid) >= tau
  2) content verification — the digit challenge generated for THIS attempt
     was read back correctly

Retry policy: a failed attempt may be retried; after `attendance.max_attempts`
failures in a row the student is locked out for `attendance.lockout_minutes`
(see profiles.record_failure / profiles.get_attendance_state). The Gradio
demo exposes a "skip lockout" shortcut for live demos — see skip_lockout()
below; a real deployment should not expose that button to students.
"""
from dataclasses import dataclass

import numpy as np

from . import challenge, profiles
from .config import CFG
from .encoder import embed
from .stt import transcribe_digits
from .vad import net_speech_seconds, trim_silence
from .verify import verify_claim


@dataclass
class AttemptResult:
    ok: bool                       # True only when this exact attempt passed both checks
    reason: str
    speaker_score: float
    speaker_pass: bool
    content_pass: bool
    transcribed: str
    expected: str
    locked_out: bool = False
    locked_until: str | None = None
    marked_present: bool = False


def new_challenge() -> str:
    return challenge.generate(CFG["attendance"]["n_digits"])


def lockout_status(user_id: int, session: str) -> tuple[bool, str | None]:
    """Returns (locked, until_ts). A lockout whose timestamp has already
    passed is treated as not locked (no explicit expiry job needed)."""
    state = profiles.get_attendance_state(user_id, session)
    until = state["locked_until"]
    if until is None:
        return False, None
    from datetime import datetime, timezone
    if datetime.now(timezone.utc).isoformat() < until:
        return True, until
    return False, None


def skip_lockout(user_id: int, session: str):
    """Demo-only shortcut: clears the lockout and resets the fail counter
    so grading/demoing doesn't require actually waiting out the cooldown."""
    profiles.reset_attendance_state(user_id, session)


def attempt(claimed_user: profiles.User, session: str, expected_digits: str,
            audio: np.ndarray) -> AttemptResult:
    """Runs one verification attempt against already-captured raw audio."""
    locked, until = lockout_status(claimed_user.id, session)
    if locked:
        return AttemptResult(False, f"locked out until {until}", 0.0, False, False,
                              "", expected_digits, locked_out=True, locked_until=until)

    if profiles.is_present(claimed_user.id, session):
        return AttemptResult(True, "already marked present", 1.0, True, True,
                              "", expected_digits, marked_present=True)

    trimmed = trim_silence(audio)
    net_speech = net_speech_seconds(trimmed)
    if net_speech < CFG["speaker"]["min_net_speech_sec"]:
        result = _record_failure(claimed_user, session, expected_digits, "", 0.0, False, False)
        result.reason = f"too little speech detected ({net_speech:.2f}s)"
        return result

    emb = embed(trimmed)
    v = verify_claim(emb, claimed_user)

    transcribed_raw = transcribe_digits(trimmed) or ""
    content_pass = challenge.matches(
        expected_digits, transcribed_raw,
        tolerance=CFG["attendance"]["digit_edit_distance_tolerance"],
    )

    if v.passed and content_pass:
        state = profiles.get_attendance_state(claimed_user.id, session)
        attempts_used = state["fail_count"] + 1
        profiles.record_attempt(claimed_user.id, session, attempts_used,
                                 expected_digits, transcribed_raw, v.score,
                                 v.passed, content_pass, True)
        profiles.mark_present(claimed_user.id, session, v.score, attempts_used)
        profiles.reset_attendance_state(claimed_user.id, session)
        return AttemptResult(True, "verified", v.score, v.passed, content_pass,
                              transcribed_raw, expected_digits, marked_present=True)

    result = _record_failure(claimed_user, session, expected_digits, transcribed_raw,
                              v.score, v.passed, content_pass)
    if not v.passed and not content_pass:
        result.reason = "voice did not match the enrolled profile, and the digits were misread"
    elif not v.passed:
        result.reason = "voice did not match the enrolled profile for this name"
    else:
        result.reason = "digits read back did not match the challenge"
    return result


def _record_failure(user: profiles.User, session: str, expected: str, transcribed: str,
                     score: float, speaker_pass: bool, content_pass: bool) -> AttemptResult:
    state = profiles.record_failure(user.id, session, CFG["attendance"]["max_attempts"],
                                     CFG["attendance"]["lockout_minutes"])
    profiles.record_attempt(user.id, session, state["fail_count"], expected, transcribed,
                             score, speaker_pass, content_pass, False)
    locked = state["locked_until"] is not None
    return AttemptResult(False, "", score, speaker_pass, content_pass, transcribed, expected,
                          locked_out=locked, locked_until=state["locked_until"])
