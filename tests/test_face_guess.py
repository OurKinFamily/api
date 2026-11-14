"""Who an unnamed face probably is.

The gate is a lead over the runner-up, not a raw score. A picture of Amelia
scored 0.746 against her own brain with the next-best person at 0.336 —
unmistakably her — and an earlier 0.75 floor discarded it over four
thousandths. These tests pin the rule that replaced it.
"""
import numpy as np

from app.services.face_guess import FLOOR, MARGIN, guess_people


def brain(people):
    """Assemble the loader's tuple from {person_id: (name, [centroids], negatives)}.

    Mirrors `_load_brain`: buckets sorted by person, with the reduceat segment
    starts precomputed.
    """
    centroids, bucket_person_ix, order, persons = [], [], [], {}
    for pid, (name, cents, negs) in people.items():
        ix = len(order)
        order.append(pid)
        persons[pid] = {"name": name, "negatives": set(negs or ())}
        for c in cents:
            centroids.append(c)
            bucket_person_ix.append(ix)
    C = np.asarray(centroids, dtype=np.float32)
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    bix = np.asarray(bucket_person_ix, dtype=np.int32)
    sort_idx = np.argsort(bix, kind="stable")
    sorted_persons = bix[sort_idx]
    seg_starts = np.concatenate(([0], np.where(np.diff(sorted_persons) != 0)[0] + 1)).astype(np.int64)
    seg_persons = sorted_persons[seg_starts]
    return (C, bix, order, persons, [], seg_starts, seg_persons, sort_idx)


DIM = 8


def vec(*weights):
    """A unit vector leaning on the first few axes."""
    v = np.zeros(DIM, dtype=np.float32)
    for i, w in enumerate(weights):
        v[i] = w
    return (v / np.linalg.norm(v)).tolist()


AMELIA = vec(1, 0, 0)
HENRY = vec(0, 1, 0)


def face(idx, v):
    return {"face_index": idx, "embedding": v}


class TestClearAnswers:
    def test_names_the_person_when_nobody_else_is_close(self):
        b = brain({"a": ("Amelia", [AMELIA], None), "h": ("Henry", [HENRY], None)})
        out = guess_people(b, "archive/2026/11/x.jpg", [face(0, AMELIA)])
        assert out[0]["person_id"] == "a"
        assert out[0]["name"] == "Amelia"
        assert out[0]["score"] == 1.0

    def test_accepts_a_middling_score_with_a_commanding_lead(self):
        # regression(2026-11-13): the real case. 0.746 against Amelia with the
        # runner-up at 0.336 was discarded by a flat 0.75 floor, though no
        # other person was remotely close.
        mix = vec(0.746, 0.0, 0.666)          # ~0.746 to Amelia, ~0 to Henry
        b = brain({"a": ("Amelia", [AMELIA], None), "h": ("Henry", [HENRY], None)})
        out = guess_people(b, "archive/2026/11/x.jpg", [face(0, mix)])
        assert out[0]["person_id"] == "a"
        assert 0.70 <= out[0]["score"] < 0.75

    def test_accepts_a_face_off_a_video_frame(self):
        # regression(2026-11-13): faces sampled from video frames score far
        # lower than photographs — a night-time clip put Guillaume at 0.45
        # with the runner-up at 0.20, unmistakably him. A floor set from
        # photographs alone rejected every face in a 28-face video.
        dim = vec(0.45, 0.0, 0.893)
        b = brain({"g": ("Guillaume", [AMELIA], None), "h": ("Henry", [HENRY], None)})
        out = guess_people(b, "archive/2026/11/clip.mov", [face(0, dim)])
        assert out[0]["person_id"] == "g"
        assert out[0]["score"] < 0.50

    def test_answers_several_faces_in_one_photograph(self):
        b = brain({"a": ("Amelia", [AMELIA], None), "h": ("Henry", [HENRY], None)})
        out = guess_people(b, "archive/x.jpg", [face(0, AMELIA), face(3, HENRY)])
        assert out[0]["person_id"] == "a"
        assert out[3]["person_id"] == "h"

    def test_keys_the_answer_to_the_face_it_is_about(self):
        b = brain({"a": ("Amelia", [AMELIA], None), "h": ("Henry", [HENRY], None)})
        out = guess_people(b, "archive/x.jpg", [face(7, AMELIA)])
        assert list(out) == [7]


class TestRefusingToGuess:
    def test_stays_quiet_on_a_close_contest(self):
        # Two sisters, both plausible. A near-tie is a question for a person,
        # not a one-click confirmation.
        close = vec(1, 0.97, 0)
        b = brain({"a": ("Amelia", [AMELIA], None), "m": ("Margaret", [HENRY], None)})
        assert guess_people(b, "archive/x.jpg", [face(0, close)]) == {}

    def test_stays_quiet_below_the_floor(self):
        faint = vec(0.3, 0, 0.954)      # 0.3 to Amelia, nothing to Henry
        b = brain({"a": ("Amelia", [AMELIA], None), "h": ("Henry", [HENRY], None)})
        assert guess_people(b, "archive/x.jpg", [face(0, faint)]) == {}

    def test_will_not_offer_somebody_already_turned_down_for_that_face(self):
        b = brain({
            "a": ("Amelia", [AMELIA], [("archive/x.jpg", 0)]),
            "h": ("Henry", [HENRY], None),
        })
        assert guess_people(b, "archive/x.jpg", [face(0, AMELIA)]) == {}

    def test_the_rejection_is_only_about_that_one_face(self):
        b = brain({
            "a": ("Amelia", [AMELIA], [("archive/x.jpg", 0)]),
            "h": ("Henry", [HENRY], None),
        })
        out = guess_people(b, "archive/x.jpg", [face(1, AMELIA)])
        assert out[1]["person_id"] == "a"

    def test_the_rejection_is_only_about_that_one_photograph(self):
        b = brain({
            "a": ("Amelia", [AMELIA], [("archive/x.jpg", 0)]),
            "h": ("Henry", [HENRY], None),
        })
        out = guess_people(b, "archive/other.jpg", [face(0, AMELIA)])
        assert out[0]["person_id"] == "a"


class TestNothingToGoOn:
    def test_no_brain_yet(self):
        assert guess_people(None, "archive/x.jpg", [face(0, AMELIA)]) == {}

    def test_no_faces(self):
        b = brain({"a": ("Amelia", [AMELIA], None)})
        assert guess_people(b, "archive/x.jpg", []) == {}

    def test_skips_a_face_with_no_embedding(self):
        # Sidecars written before the field existed, and boxes the detector
        # could not describe.
        b = brain({"a": ("Amelia", [AMELIA], None), "h": ("Henry", [HENRY], None)})
        faces = [{"face_index": 0}, {"face_index": 1, "embedding": []}, face(2, AMELIA)]
        out = guess_people(b, "archive/x.jpg", faces)
        assert list(out) == [2]

    def test_skips_a_face_with_no_index(self):
        b = brain({"a": ("Amelia", [AMELIA], None)})
        assert guess_people(b, "archive/x.jpg", [{"embedding": AMELIA}]) == {}

    def test_ignores_embeddings_of_the_wrong_width(self):
        # A model change would otherwise raise inside the matmul and take the
        # whole detail response with it.
        b = brain({"a": ("Amelia", [AMELIA], None)})
        assert guess_people(b, "archive/x.jpg", [face(0, [0.1, 0.2])]) == {}

    def test_a_zero_vector_does_not_win_everything(self):
        # It normalises to NaN, and NaN compares false — but it used to be
        # divided by zero first, which warned and then poisoned the row.
        b = brain({"a": ("Amelia", [AMELIA], None), "h": ("Henry", [HENRY], None)})
        assert guess_people(b, "archive/x.jpg", [face(0, [0.0] * DIM)]) == {}

    def test_copes_with_a_single_person_having_a_brain(self):
        # No runner-up to measure a lead against.
        b = brain({"a": ("Amelia", [AMELIA], None)})
        out = guess_people(b, "archive/x.jpg", [face(0, AMELIA)])
        assert out[0]["person_id"] == "a"

    def test_falls_back_to_the_id_when_a_brain_has_no_name(self):
        b = brain({"a": (None, [AMELIA], None), "h": ("Henry", [HENRY], None)})
        out = guess_people(b, "archive/x.jpg", [face(0, AMELIA)])
        assert out[0]["name"] == "a"


class TestTheRule:
    def test_the_lead_is_what_does_the_work(self):
        # Documented so a future change to these numbers is deliberate: the
        # floor is loose and the margin is strict, not the other way round.
        # A hold-out over 71,292 faces put the margin at the knee of the
        # curve — 0.20 to 0.25 halves the rate at which strangers get named,
        # while raising the floor costs recall and buys nothing. See the
        # module docstring for the table.
        assert FLOOR == 0.45
        assert MARGIN == 0.25
