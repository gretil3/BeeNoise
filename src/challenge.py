"""Randomly generated digit-readback challenge — the liveness / anti-replay
half of attendance verification.

A recording of the real student's voice does not contain today's randomly
generated digits, so passing attendance requires BOTH:
  1) speaker verification — the voice matches the claimed student's
     enrolled voiceprint (see verify.py)
  2) content verification — the student correctly read back THIS attempt's
     challenge (this module)

This is what turns "known replay-attack limitation" (BLUEPRINT.md, Known
Traps) from an out-of-scope caveat into something the system actually
defends against, and it's a good experiment for the report: show a
same-voice, wrong-digits attempt getting rejected where speaker-ID alone
would have accepted it.
"""
import random

WORD_TO_DIGIT = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def generate(n_digits: int) -> str:
    """Returns a canonical digit string, e.g. '8249'."""
    return "".join(str(random.randint(0, 9)) for _ in range(n_digits))


def format_for_display(digits: str) -> str:
    """'8249' -> '8-2-4-9'"""
    return "-".join(digits)


def normalize_transcript(text: str) -> str:
    """Extracts a canonical digit string from an ASR transcript, handling
    both digit characters ('8 2 4 9', '8249') and number words
    ('eight two four nine'). Non-digit words are dropped."""
    tokens = []
    token = ""
    for ch in text.lower():
        if ch.isalnum():
            token += ch
        elif token:
            tokens.append(token)
            token = ""
    if token:
        tokens.append(token)

    digits = []
    for tok in tokens:
        if tok.isdigit():
            digits.extend(list(tok))
        elif tok in WORD_TO_DIGIT:
            digits.append(WORD_TO_DIGIT[tok])
    return "".join(digits)


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def matches(expected: str, transcript: str, tolerance: int = 0) -> bool:
    """tolerance = max allowed edit distance between the expected digit
    string and what was actually transcribed (0 = exact match required)."""
    got = normalize_transcript(transcript)
    if tolerance <= 0:
        return got == expected
    return _edit_distance(expected, got) <= tolerance


if __name__ == "__main__":
    d = generate(4)
    print(f"Challenge: {format_for_display(d)}")
    print(f"normalize('eight two four nine') -> {normalize_transcript('eight two four nine')!r}")
    print(f"normalize('8, 2, 4, 9.') -> {normalize_transcript('8, 2, 4, 9.')!r}")
    print(f"matches(exact readback) -> {matches(d, ' '.join(list(d)))}")
    print(f"matches(wrong digits) -> {matches(d, '0 0 0 0')}")
