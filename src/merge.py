"""Stage 5 — merge: words + speaker turns -> subtitle cues -> .srt / .vtt / .json.

Speaker assignment is word-level (the WhisperX approach): each word gets the
speaker whose diarized turn overlaps it the most; a word in a gap between
turns takes the nearest turn. Words are then grouped into cues, starting a
new cue whenever the speaker changes, a pause is too long, or the cue gets
too long to read.

Pure Python — unit-tested in tests/test_merge.py.
"""
import json
from dataclasses import asdict
from pathlib import Path

from .config import CFG
from .segments import Cue, Segment, Word, overlap


def assign_speakers(words: list[Word], segments: list[Segment]) -> list[Word]:
    if not segments:
        return words
    out = []
    for w in words:
        if w.speaker is not None:  # `segment` strategy already labelled it
            out.append(w)
            continue
        best = max(segments, key=lambda s: overlap(w.start, w.end, s.start, s.end))
        if overlap(w.start, w.end, best.start, best.end) == 0:
            mid = (w.start + w.end) / 2
            best = min(segments, key=lambda s: min(abs(mid - s.start), abs(mid - s.end)))
        out.append(Word(w.start, w.end, w.text, best.speaker, w.probability))
    return out


def build_cues(words: list[Word], max_sec: float | None = None, max_chars: int | None = None,
               max_gap: float | None = None) -> list[Cue]:
    sub = CFG["subtitles"]
    max_sec = sub["max_cue_sec"] if max_sec is None else max_sec
    max_chars = sub["max_cue_chars"] if max_chars is None else max_chars
    max_gap = sub["max_gap_sec"] if max_gap is None else max_gap

    # 1) Phrases: break on speaker change, a long pause, or the end of a sentence.
    phrases: list[list[Word]] = []
    for w in words:
        if (phrases and w.speaker == phrases[-1][0].speaker
                and w.start - phrases[-1][-1].end <= max_gap
                and phrases[-1][-1].text.strip()[-1:] not in (".", "?", "!")):
            phrases[-1].append(w)
        else:
            phrases.append([w])

    # 2) Split phrases that are too long into n *balanced* pieces, instead of
    #    filling cues greedily — greedy leaves one-word orphans like "rate."
    cues: list[Cue] = []
    for ph in phrases:
        for n in range(1, len(ph) + 1):
            pieces = _balanced_split(ph, n)
            if all(_fits(p, max_sec, max_chars) for p in pieces):
                break
        for p in pieces:
            text = _text(p)
            if text:
                # No diarized turns at all -> words have no speaker; don't print "[] text".
                speaker = p[0].speaker or f"{CFG['subtitles']['unknown_prefix']} 1"
                cues.append(Cue(p[0].start, p[-1].end, speaker, text, p))
    return cues


def _text(ws: list[Word]) -> str:
    return "".join(w.text for w in ws).strip()


def _fits(ws: list[Word], max_sec: float, max_chars: int) -> bool:
    return len(ws) == 1 or (ws[-1].end - ws[0].start <= max_sec and len(_text(ws)) <= max_chars)


def _balanced_split(ws: list[Word], n: int) -> list[list[Word]]:
    """Cut `ws` into n contiguous pieces with roughly equal character counts."""
    if n <= 1:
        return [ws]
    lengths = [len(w.text) for w in ws]
    total = sum(lengths)
    pieces, start, acc = [], 0, 0
    for i, length in enumerate(lengths):
        acc += length
        remaining_cuts = n - 1 - len(pieces)
        if (remaining_cuts and acc >= total * (len(pieces) + 1) / n
                and len(ws) - (i + 1) >= remaining_cuts):
            pieces.append(ws[start:i + 1])
            start = i + 1
    pieces.append(ws[start:])
    return [p for p in pieces if p]


def _ts(sec: float, sep: str) -> str:
    ms = int(round(max(0.0, sec) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(cues: list[Cue], label_format: str | None = None) -> str:
    fmt = label_format or CFG["subtitles"]["label_format"]
    blocks = [f"{i}\n{_ts(c.start, ',')} --> {_ts(c.end, ',')}\n"
              f"{fmt.format(speaker=c.speaker, text=c.text)}\n"
              for i, c in enumerate(cues, 1)]
    return "\n".join(blocks)


def to_vtt(cues: list[Cue]) -> str:
    # WebVTT has a native speaker ("voice") tag: <v Name>text
    blocks = [f"{_ts(c.start, '.')} --> {_ts(c.end, '.')}\n<v {c.speaker}>{c.text}\n"
              for c in cues]
    return "WEBVTT\n\n" + "\n".join(blocks)


def write_outputs(cues: list[Cue], out_dir, extra: dict | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"srt": out_dir / "subtitles.srt", "vtt": out_dir / "subtitles.vtt",
             "json": out_dir / "transcript.json"}
    paths["srt"].write_text(to_srt(cues), encoding="utf-8")
    paths["vtt"].write_text(to_vtt(cues), encoding="utf-8")
    payload = {**(extra or {}),
               "cues": [{k: v for k, v in asdict(c).items() if k != "words"} for c in cues],
               "words": [asdict(w) for c in cues for w in c.words]}
    paths["json"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths
