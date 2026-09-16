"""Attendance CLI.

For each student: claim a name -> read back a randomly generated digit
challenge -> the system checks BOTH that the voice matches the claimed
student's enrolled profile AND that the digits were read correctly. Two
failed attempts in a row lock that name out for `attendance.lockout_minutes`
(config.yaml) before another attempt is allowed.

Run: python -m src.main [--session "2026-09-16 Speech Recognition"]
"""
import argparse
from datetime import date

from . import attendance, challenge, profiles
from .audio_io import record_fixed


def _prompt_claimed_user() -> profiles.User | None:
    users = profiles.get_all_users()
    if not users:
        print("No enrolled students yet. Run `python -m src.enroll --name <name>` first.")
        return None
    print("\nEnrolled students:", ", ".join(u.name for u in users))
    name = input("Name for attendance (blank to quit): ").strip()
    if not name:
        raise KeyboardInterrupt
    user = profiles.find_user_ci(name)
    if user is None:
        print(f"No enrolled student named '{name}'.")
    return user


def run(session: str):
    print(f"Attendance session: {session}")
    print("Enter a blank name (or Ctrl+C) to quit.\n")
    try:
        while True:
            user = _prompt_claimed_user()
            if user is None:
                continue

            if profiles.is_present(user.id, session):
                print(f"{user.name} is already marked present for this session.\n")
                continue

            locked, until = attendance.lockout_status(user.id, session)
            if locked:
                print(f"{user.name} is locked out until {until} after too many failed "
                      f"attempts. (The Gradio demo exposes a Skip-cooldown button for "
                      f"live demos; the CLI intentionally does not.)\n")
                continue

            digits = attendance.new_challenge()
            print(f"\nPlease read the following numbers aloud: "
                  f"{challenge.format_for_display(digits)}")
            input("Press Enter, then read the numbers clearly...")
            audio = record_fixed(4.0)

            result = attendance.attempt(user, session, digits, audio)
            print(f"  transcribed: {result.transcribed!r}")
            print(f"  speaker score: {result.speaker_score:.3f} "
                  f"({'pass' if result.speaker_pass else 'fail'})")
            print(f"  digits: {'match' if result.content_pass else 'no match'}")

            if result.marked_present:
                print(f"ATTENDANCE VERIFIED for {user.name}.\n")
            elif result.locked_out:
                print(f"Too many failed attempts. {user.name} is locked out until "
                      f"{result.locked_until}.\n")
            else:
                print(f"Verification failed: {result.reason}. Try again.\n")

    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=str(date.today()),
                         help="Session/class identifier, e.g. a date or class name")
    args = parser.parse_args()
    run(args.session)
