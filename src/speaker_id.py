"""Stage 3 — speaker ID: turn anonymous diarization labels into names.

For each diarized speaker (cluster), embed its segments with ECAPA-TDNN,
average them (weighted by duration) into one cluster voiceprint, and score it
against every enrolled centroid with cosine similarity.

Diarizers often split one voice into two clusters. Before naming, clusters
whose voiceprints are near-identical (cosine >= speaker.merge_cosine) are
grouped, so a split voice gets one name instead of "Alex" plus a phantom
"Speaker 1".

Assignment is one-to-one over those groups (Hungarian algorithm): two
different speakers can't both be "Alex". A match is only accepted if its
cosine >= speaker.tau;
everything else becomes "Speaker 1", "Speaker 2", ... in order of appearance.
That's the open-set part: unenrolled voices get separated but not named.

`assign`, `group_clusters` and `unknown_names` are pure numpy/scipy so they're unit-testable
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


def group_clusters(embs: np.ndarray, merge_cosine: float | None) -> list[int]:
    """embs: (n_clusters, dim) unit-norm voiceprints. Returns a group id per
    cluster; clusters that are the same voice share one. Average linkage, so a
    chain of "A is like B, B is like C" can't glue together voices that differ.
    merge_cosine=None disables merging (every cluster is its own group)."""
    n = len(embs)
    if n < 2 or merge_cosine is None:
        return list(range(n))
    from scipy.cluster.hierarchy import fcluster, linkage
    Z = linkage(embs, method="average", metric="cosine")
    groups = fcluster(Z, t=1.0 - merge_cosine, criterion="distance")
    # Renumber 0..k-1 in order of first appearance so output is stable.
    remap: dict[int, int] = {}
    return [remap.setdefault(int(g), len(remap)) for g in groups]


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
    budget = CFG["speaker"]["max_cluster_sec"]
    # Longest segments first: they carry the most reliable voice evidence, and
    # the budget keeps a 20-minute speaker from costing 20 minutes of embedding.
    usable = sorted((s for s in segs if s.duration >= min_sec) or segs,
                    key=lambda s: s.duration, reverse=True)
    embs, durs = [], []
    for s in usable:
        if sum(durs) >= budget:
            break
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
    # Same voice split across clusters -> one group, one name.
    gids = group_clusters(np.stack([cluster_embs[lab] for lab in scorable]) if scorable
                          else np.zeros((0, 0)), CFG["speaker"]["merge_cosine"])
    group_of = dict(zip(scorable, gids, strict=True))
    n_groups = max(gids, default=-1) + 1
    group_embs = []
    for g in range(n_groups):
        members = [lab for lab in scorable if group_of[lab] == g]
        e = np.average(np.stack([cluster_embs[lab] for lab in members]), axis=0,
                       weights=[speech[lab] for lab in members])
        group_embs.append(e / np.linalg.norm(e))

    if profiles and n_groups:
        scores = np.stack(group_embs) @ np.stack([p.centroid for p in profiles]).T
    else:
        scores = np.zeros((n_groups, 0))
    assignment = assign(scores, tau)

    named, best = {}, {}  # keyed by group id
    for g in range(n_groups):
        if scores.shape[1]:
            j = int(np.argmax(scores[g]))
            best[g] = (profiles[j].name, float(scores[g, j]))
        if assignment[g] is not None:
            named[g] = (profiles[assignment[g]].name, float(scores[g, assignment[g]]))

    # A label too short to embed stays on its own, never merged or named.
    key = {lab: group_of.get(lab, lab) for lab in order}
    keys_in_order = list(dict.fromkeys(key[lab] for lab in order))
    anon = unknown_names(keys_in_order, set(named), CFG["subtitles"]["unknown_prefix"])
    matches = {}
    for label in order:
        k = key[label]
        if k in named:
            name, score = named[k]
            matches[label] = Match(label, name, True, score, name, speech[label])
        else:
            cand, score = best.get(k, (None, None))
            matches[label] = Match(label, anon[k], False, score, cand, speech[label])
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
