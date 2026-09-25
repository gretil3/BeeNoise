from src.segments import (
    Segment,
    merge_adjacent,
    read_audacity_labels,
    read_reference,
    read_rttm,
    speaker_order,
    write_rttm,
)


def test_rttm_roundtrip(tmp_path):
    segs = [Segment(0.0, 1.5, "Alex"), Segment(1.5, 3.25, "guest1")]
    p = tmp_path / "x.rttm"
    write_rttm(segs, p, uri="x")
    assert p.read_text().splitlines()[0] == "SPEAKER x 1 0.000 1.500 <NA> <NA> Alex <NA> <NA>"
    back = read_rttm(p)
    assert [(s.start, s.end, s.speaker) for s in back] == [(0.0, 1.5, "Alex"), (1.5, 3.25, "guest1")]


def test_audacity_labels(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("0.5\t2.0\tAlex\n\\\t100\t200\n2.0\t4.0\tguest one\n3.0\t3.0\tempty\n")
    segs = read_audacity_labels(p)
    assert [(s.start, s.end, s.speaker) for s in segs] == [(0.5, 2.0, "Alex"), (2.0, 4.0, "guest_one")]
    assert read_reference(p)[0].speaker == "Alex"


def test_merge_adjacent_joins_same_speaker_only():
    segs = [Segment(0, 1, "A"), Segment(1.2, 2, "A"), Segment(2, 3, "B"), Segment(5, 6, "B")]
    out = merge_adjacent(segs, max_gap=0.5)
    assert [(s.start, s.end, s.speaker) for s in out] == [(0, 2, "A"), (2, 3, "B"), (5, 6, "B")]


def test_speaker_order_is_first_appearance():
    segs = [Segment(3, 4, "B"), Segment(0, 1, "C"), Segment(1, 2, "B")]
    assert speaker_order(segs) == ["C", "B"]
