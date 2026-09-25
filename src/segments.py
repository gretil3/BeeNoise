"""Shared data types for the pipeline + RTTM / label-file I/O.

Pure Python — no model imports — so the unit tests and eval scripts can use
it without loading anything heavy.

RTTM is the standard diarization file format (one line per speaker turn):
    SPEAKER <uri> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Segment:
    """One speaker turn: `speaker` spoke from `start` to `end` (seconds)."""
    start: float
    end: float
    speaker: str

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Word:
    start: float
    end: float
    text: str                  # as Whisper emits it, usually with a leading space
    speaker: str | None = None
    probability: float | None = None


@dataclass
class Cue:
    """One subtitle entry."""
    start: float
    end: float
    speaker: str
    text: str
    words: list[Word] = field(default_factory=list, repr=False)


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def merge_adjacent(segments: list[Segment], max_gap: float = 0.5) -> list[Segment]:
    """Join consecutive turns of the same speaker separated by < max_gap."""
    out: list[Segment] = []
    for s in sorted(segments, key=lambda s: s.start):
        if out and out[-1].speaker == s.speaker and s.start - out[-1].end <= max_gap:
            out[-1].end = max(out[-1].end, s.end)
        else:
            out.append(Segment(s.start, s.end, s.speaker))
    return out


def speaker_order(segments: list[Segment]) -> list[str]:
    """Speaker labels in order of first appearance."""
    seen: list[str] = []
    for s in sorted(segments, key=lambda s: s.start):
        if s.speaker not in seen:
            seen.append(s.speaker)
    return seen


# ---------------------------------------------------------------------------
# File formats
# ---------------------------------------------------------------------------

def write_rttm(segments: list[Segment], path, uri: str):
    lines = [
        f"SPEAKER {uri} 1 {s.start:.3f} {s.duration:.3f} <NA> <NA> "
        f"{s.speaker.replace(' ', '_')} <NA> <NA>"
        for s in sorted(segments, key=lambda s: s.start)
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_rttm(path) -> list[Segment]:
    segs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start, dur = float(parts[3]), float(parts[4])
        segs.append(Segment(start, start + dur, parts[7]))
    return sorted(segs, key=lambda s: s.start)


def read_audacity_labels(path) -> list[Segment]:
    """Audacity 'Export Labels' format: start<TAB>end<TAB>label per line.
    This is the easiest way for the team to hand-label ground truth."""
    segs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3 or parts[0].startswith("\\"):
            continue  # skip blank lines and Audacity's spectral-selection rows
        start, end, label = float(parts[0]), float(parts[1]), parts[2].strip()
        if end > start and label:
            segs.append(Segment(start, end, label.replace(" ", "_")))
    return sorted(segs, key=lambda s: s.start)


def read_reference(path) -> list[Segment]:
    """Ground truth from either .rttm or an Audacity .txt label export."""
    path = Path(path)
    return read_rttm(path) if path.suffix.lower() == ".rttm" else read_audacity_labels(path)
