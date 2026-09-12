"""Speaker identification: compare an embedding against all enrolled
centroids and decide who (if anyone) is speaking.

Three independent rejection knobs (see config.yaml):
  tau           - absolute cosine threshold (rejects unknown/open-set speakers)
  min_margin    - top1-vs-top2 margin (rejects confusable enrolled users)
  smoothing     - N consecutive agreeing turns required before switching the
                  *active* user in a live session (see SpeakerTracker)
"""
from dataclasses import dataclass

import numpy as np

from . import profiles
from .config import CFG
from .encoder import cosine


@dataclass
class IdentifyResult:
    user: profiles.User | None
    score: float
    margin: float
    all_scores: dict


def identify(emb: np.ndarray, users: list[profiles.User] | None = None) -> IdentifyResult:
    users = users if users is not None else profiles.get_all_users()
    if not users:
        return IdentifyResult(None, 0.0, 0.0, {})

    scores = {u.name: cosine(emb, u.centroid) for u in users}
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_name, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
    margin = best_score - runner_up

    tau = CFG["speaker"]["tau"]
    min_margin = CFG["speaker"]["min_margin"]

    if best_score < tau or margin < min_margin:
        return IdentifyResult(None, best_score, margin, scores)

    user = next(u for u in users if u.name == best_name)
    return IdentifyResult(user, best_score, margin, scores)


class SpeakerTracker:
    """Keeps the pipeline from flipping the active profile on one noisy turn.
    Requires `smoothing_turns` consecutive agreeing identifications before
    the active user actually changes."""

    def __init__(self, smoothing_turns: int | None = None):
        self.smoothing_turns = smoothing_turns or CFG["speaker"]["smoothing_turns"]
        self.active_user: profiles.User | None = None
        self._pending_name: str | None = None
        self._pending_count = 0

    def update(self, result: IdentifyResult) -> profiles.User | None:
        candidate_name = result.user.name if result.user else None
        current_name = self.active_user.name if self.active_user else None

        if candidate_name == current_name:
            self._pending_name = None
            self._pending_count = 0
            return self.active_user

        if candidate_name == self._pending_name:
            self._pending_count += 1
        else:
            self._pending_name = candidate_name
            self._pending_count = 1

        if self._pending_count >= self.smoothing_turns:
            self.active_user = result.user
            self._pending_name = None
            self._pending_count = 0

        return self.active_user
