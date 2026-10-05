import numpy as np

from src import speaker_id
from src.segments import Segment
from src.speaker_id import assign, group_clusters, unknown_names


def test_assignment_is_one_to_one():
    # Both clusters are closest to profile 0, but only one can be it.
    scores = np.array([[0.80, 0.60],
                       [0.75, 0.20]])
    # Best total: cluster0->profile1 (0.60) + cluster1->profile0 (0.75) = 1.35 > 0.80 + 0.20
    assert assign(scores, tau=0.5) == [1, 0]


def test_below_tau_stays_unknown():
    scores = np.array([[0.9, 0.1],
                       [0.2, 0.25]])
    assert assign(scores, tau=0.3) == [0, None]


def test_more_clusters_than_profiles():
    scores = np.array([[0.9], [0.8], [0.1]])
    assert assign(scores, tau=0.3) == [0, None, None]


def test_no_profiles():
    assert assign(np.zeros((2, 0)), tau=0.3) == [None, None]


def test_unknown_names_numbered_by_appearance():
    names = unknown_names(["SPK_2", "SPK_0", "SPK_1"], named={"SPK_0"}, prefix="Speaker")
    assert names == {"SPK_2": "Speaker 1", "SPK_1": "Speaker 2"}


# --- merging clusters that are really one voice --------------------------------

def _unit(*v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


ALEX = _unit(1, 0, 0, 0)
ALEX_2 = _unit(1, 0.15, 0, 0)    # same voice, split into a second cluster (cos ~0.99)
BOB = _unit(0, 1, 0, 0)
BOB_2 = _unit(0.1, 1, 0, 0)


def test_group_clusters_merges_the_same_voice_only():
    embs = np.stack([ALEX, BOB, ALEX_2])
    assert group_clusters(embs, merge_cosine=0.65) == [0, 1, 0]


def test_group_clusters_can_be_disabled():
    assert group_clusters(np.stack([ALEX, ALEX_2]), merge_cosine=None) == [0, 1]


def test_group_clusters_trivial_inputs():
    assert group_clusters(np.zeros((0, 0)), merge_cosine=0.65) == []
    assert group_clusters(np.stack([ALEX]), merge_cosine=0.65) == [0]


def _identify(monkeypatch, voices: dict[str, np.ndarray], profiles, merge_cosine=0.65):
    monkeypatch.setitem(speaker_id.CFG["speaker"], "merge_cosine", merge_cosine)
    monkeypatch.setattr(speaker_id, "_segment_embeddings",
                        lambda wav, sr, segs: ([voices[segs[0].speaker]], [5.0]))
    segs = [Segment(i * 5.0, i * 5.0 + 5.0, label) for i, label in enumerate(voices)]
    out = speaker_id.identify_clusters(np.zeros(16000), 16000, segs, profiles=profiles, tau=0.35)
    return {k: m.name for k, m in out.items()}


class _Profile:
    def __init__(self, name, centroid):
        self.name, self.centroid = name, centroid


def test_split_cluster_of_an_enrolled_speaker_keeps_their_name(monkeypatch):
    voices = {"SPEAKER_00": ALEX, "SPEAKER_01": BOB, "SPEAKER_02": ALEX_2}
    names = _identify(monkeypatch, voices, [_Profile("Alex", ALEX)])
    assert names == {"SPEAKER_00": "Alex", "SPEAKER_01": "Speaker 1", "SPEAKER_02": "Alex"}


def test_without_merging_the_second_cluster_is_a_phantom_speaker(monkeypatch):
    voices = {"SPEAKER_00": ALEX, "SPEAKER_01": ALEX_2}
    names = _identify(monkeypatch, voices, [_Profile("Alex", ALEX)], merge_cosine=None)
    assert sorted(names.values()) == ["Alex", "Speaker 1"]


def test_split_cluster_of_an_unknown_speaker_shares_one_number(monkeypatch):
    voices = {"SPEAKER_00": BOB, "SPEAKER_01": ALEX, "SPEAKER_02": BOB_2}
    names = _identify(monkeypatch, voices, [])
    assert names == {"SPEAKER_00": "Speaker 1", "SPEAKER_01": "Speaker 2",
                     "SPEAKER_02": "Speaker 1"}
