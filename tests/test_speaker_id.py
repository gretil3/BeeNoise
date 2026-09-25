import numpy as np

from src.speaker_id import assign, unknown_names


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
