from src.merge import assign_speakers, build_cues, to_srt, to_vtt
from src.segments import Segment, Word


def W(start, end, text, speaker=None):
    return Word(start, end, text, speaker)


def test_words_take_the_speaker_they_overlap_most():
    segs = [Segment(0.0, 2.0, "Alex"), Segment(2.0, 4.0, "Speaker 1")]
    words = [W(0.1, 0.5, " hi"), W(1.8, 2.6, " there"), W(3.0, 3.4, " yes")]
    out = assign_speakers(words, segs)
    assert [w.speaker for w in out] == ["Alex", "Speaker 1", "Speaker 1"]


def test_word_in_a_gap_takes_the_nearest_turn():
    segs = [Segment(0.0, 1.0, "A"), Segment(5.0, 6.0, "B")]
    out = assign_speakers([W(4.2, 4.6, " late")], segs)
    assert out[0].speaker == "B"


def test_words_already_labelled_are_kept():
    out = assign_speakers([W(0, 1, " x", "B")], [Segment(0, 1, "A")])
    assert out[0].speaker == "B"


def test_cues_split_on_speaker_change_and_long_pause():
    words = [W(0.0, 0.4, " Hello", "A"), W(0.5, 0.9, " world", "A"),
             W(1.0, 1.3, " Hi", "B"),
             W(5.0, 5.3, " again", "B")]  # 3.7 s pause -> new cue
    cues = build_cues(words, max_sec=6, max_chars=84, max_gap=1.0)
    assert [(c.speaker, c.text) for c in cues] == [("A", "Hello world"), ("B", "Hi"), ("B", "again")]


def test_cues_split_when_too_long():
    words = [W(i * 0.5, i * 0.5 + 0.4, " word", "A") for i in range(20)]
    cues = build_cues(words, max_sec=3.0, max_chars=1000, max_gap=1.0)
    assert len(cues) > 1
    assert all(c.end - c.start <= 3.0 for c in cues)


def test_long_phrase_is_split_evenly_without_orphans():
    # 12 words over 7.2 s, limit 6 s: greedy would give 10 words + a 2-word orphan.
    words = [W(i * 0.6, i * 0.6 + 0.5, " word", "A") for i in range(12)]
    cues = build_cues(words, max_sec=6.0, max_chars=1000, max_gap=1.0)
    assert [len(c.words) for c in cues] == [6, 6]


def test_sentence_end_starts_a_new_cue():
    words = [W(0.0, 0.3, " Okay.", "A"), W(0.4, 0.8, " Next", "A"), W(0.9, 1.2, " week.", "A")]
    cues = build_cues(words, max_sec=6, max_chars=84, max_gap=1.0)
    assert [c.text for c in cues] == ["Okay.", "Next week."]


def test_srt_and_vtt_format():
    cues = build_cues([W(61.5, 62.25, " Hello", "Alex")])
    srt = to_srt(cues, label_format="[{speaker}] {text}")
    assert srt == "1\n00:01:01,500 --> 00:01:02,250\n[Alex] Hello\n"
    vtt = to_vtt(cues)
    assert vtt.startswith("WEBVTT\n\n00:01:01.500 --> 00:01:02.250\n<v Alex>Hello\n")
