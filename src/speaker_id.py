"""Stage 3 — speaker ID: turn anonymous diarization labels into names.

For each diarized speaker (cluster), embed its segments with ECAPA-TDNN,
average them (weighted by duration) into one cluster voiceprint, and score it
against every enrolled centroid with cosine similarity.

Assignment is one-to-one (Hungarian algorithm): two different clusters can't
both be "Alex". A match is only accepted if its cosine >= speaker.tau;
everything else becomes "Speaker 1", "Speaker 2", ... in order of appearance.
That's the open-set part: unenrolled voices get separated but not named.

`assign` and `unknown_names` are pure numpy/scipy so they're unit-testable
without loading a model.
"""
from dataclasses import dataclass

import numpy as np

from .audio_io import cut
from .config import CFG
from .segments import Segment, speaker_order


@dataclass
class Match:
    label: str               # diarization label, e.g. SPEAKER_00
    name: str                # display name: enrolled name or "Speaker N"
    enrolled: bool           # True if matched to an enrolled profile
    score: float | None      # cosine to the assigned profile (or best profile, if rejected)
    best_candidate: str | None
    speech_sec: float


def assign(scores: np.ndarray, tau: float) -> list[int | None]:
    """scores: (n_clusters, n_profiles) cosine matrix. Returns, per cluster,
    the profile index it's assigned to, or None. One-to-one, maximising total
    similarity, and a pair is only kept if its score >= tau."""
    n_clusters = scores.shape[0]
    result: list[int | None] = [None] * n_clusters
    if scores.size == 0:
        return result
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-scores)
    for r, c in zip(rows, cols, strict=True):
        if scores[r, c] >= tau:
            result[r] = int(c)
    return result


def unknown_names(labels_in_order: list[str], named: set[str], prefix: str) -> dict[str, str]:
    """Number the unnamed labels 1..N by first appearance."""
    out, k = {}, 0
    for label in labels_in_order:
        if label not in named:
            k += 1
            out[label] = f"{prefix} {k}"
    return out


def _segment_embeddings(wav16: np.ndarray, sr: int, segs: list[Segment]):
    """Embed each usable segment; returns (embeddings, durations)."""
    from .encoder import embed
    min_sec, max_sec = CFG["speaker"]["min_segment_sec"], CFG["speaker"]["max_embed_sec"]
    usable = [s for s in segs if s.duration >= min_sec] or segs
    embs, durs = [], []
    for s in usable:
        clip = cut(wav16, sr, s.start, min(s.end, s.start + max_sec))
        if len(clip) < int(0.3 * sr):
            continue
        embs.append(embed(clip))
        durs.append(len(clip) / sr)
    return embs, durs


def identify_clusters(wav16: np.ndarray, sr: int, segments: list[Segment],
                      profiles: list | None = None, tau: float | None = None) -> dict[str, Match]:
    """Name every diarization label. Returns {label: Match}."""
    from . import profiles as profiles_db
    profiles = profiles_db.get_all() if profiles is None else profiles
    tau = CFG["speaker"]["tau"] if tau is None else tau
    min_speech = CFG["speaker"]["min_cluster_speech_sec"]

    order = speaker_order(segments)
    cluster_embs, speech = {}, {}
    for label in order:
        segs = [s for s in segments if s.speaker == label]
        speech[label] = sum(s.duration for s in segs)
        embs, durs = _segment_embeddings(wav16, sr, segs)
        if embs and speech[label] >= min_speech:
            e = np.average(np.stack(embs), axis=0, weights=durs)
            cluster_embs[label] = e / np.linalg.norm(e)

    scorable = [lab for lab in order if lab in cluster_embs]
    if profiles and scorable:
        centroids = np.stack([p.centroid for p in profiles])
        scores = np.stack([cluster_embs[lab] for lab in scorable]) @ centroids.T
    else:
        scores = np.zeros((len(scorable), 0))
    assignment = assign(scores, tau)

    named = {}
    best = {}
    for i, label in enumerate(scorable):
        if scores.shape[1]:
            j = int(np.argmax(scores[i]))
            best[label] = (profiles[j].name, float(scores[i, j]))
        if assignment[i] is not None:
            named[label] = (profiles[assignment[i]].name, float(scores[i, assignment[i]]))

    anon = unknown_names(order, set(named), CFG["subtitles"]["unknown_prefix"])
    matches = {}
    for label in order:
        if label in named:
            name, score = named[label]
            matches[label] = Match(label, name, True, score, name, speech[label])
        else:
            cand, score = best.get(label, (None, None))
            matches[label] = Match(label, anon[label], False, score, cand, speech[label])
    return matches


def identify_clip(wav16: np.ndarray, profiles: list, tau: float | None = None):
    """Segment-level ID for one clip (used by eval). Returns (name|None, best_name, best_score)."""
    from .encoder import embed
    tau = CFG["speaker"]["tau"] if tau is None else tau
    if not profiles:
        return None, None, None
    e = embed(wav16)
    scores = np.stack([p.centroid for p in profiles]) @ e
    j = int(np.argmax(scores))
    best_name, best_score = profiles[j].name, float(scores[j])
    return (best_name if best_score >= tau else None), best_name, best_score


def relabel(segments: list[Segment], matches: dict[str, Match]) -> list[Segment]:
    return [Segment(s.start, s.end, matches[s.speaker].name) for s in segments]
