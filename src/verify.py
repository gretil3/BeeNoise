"""Speaker verification: does this embedding match a specific CLAIMED
student's enrolled voiceprint?

Attendance is always 1:1 verification, not open-set identification — the
student states/selects who they are (from the roster) first, then the
system checks whether their voice matches that specific enrolled
voiceprint. This is the standard design for biometric attendance/access
systems and keeps the accept/reject decision to a single cosine threshold
(`speaker.tau` in config.yaml, tuned via the Phase 8 EER analysis in
eval/eer.py) instead of the top1-vs-top2 margin logic an open-set
identification system would need.
"""
from dataclasses import dataclass

import numpy as np

from . import profiles
from .config import CFG
from .encoder import cosine


@dataclass
class VerifyResult:
    user: profiles.User
    score: float
    passed: bool


def verify_claim(emb: np.ndarray, claimed_user: profiles.User) -> VerifyResult:
    tau = CFG["speaker"]["tau"]
    score = cosine(emb, claimed_user.centroid)
    return VerifyResult(claimed_user, score, score >= tau)
