"""The Moore detector and the collapse checks, on the cases that broke them.

Each of these answers is a real one that an earlier version misclassified.
"""

from __future__ import annotations

from yaaba import moore

# Enough Moore for coverage to mean something, taken from the set's turns.
LEXICON = frozenset(
    "maan bala koosda a yi yaa sabelle karen dot yiib soaba m ba soaamb ne "
    "katre reng n yagl meng yaool kapidgu zug koom ka gind neng ye".split())


def test_moore_without_a_distinctive_letter_is_moore():
    """The original defect: `has_unique_letter` alone called 392 Moore answers
    French, because none carried ɛ ɩ ʋ or a tilde."""
    assert not moore.has_unique_letter("Maan bala, koosda bala.")
    assert moore.written_in_moore("Maan bala, koosda bala.", LEXICON)
    assert moore.written_in_moore("M ba Soaamb ne m ba Katre", LEXICON)


def test_la_and_a_are_not_french_markers():
    """Fixing the first defect brought the second: a French word list holding
    `la`, `a` or `de` called `A yi.` French."""
    for word in ("la", "a", "be", "to"):
        assert word not in moore.FRENCH, word
    assert moore.written_in_moore("A yi.", LEXICON)
    assert moore.written_in_moore("A yaa sabelle.", LEXICON)


def test_french_stays_french():
    assert not moore.written_in_moore(
        "Le proverbe dit que celui qui travaille bien en tire de la valeur.", LEXICON)
    assert not moore.written_in_moore("Je ne sais pas.", LEXICON)


def test_a_distinctive_letter_decides_alone():
    """Without a lexicon at hand, the detector must stay usable."""
    assert moore.written_in_moore("Yel-bũnd bʋg n yet woto?")
    assert not moore.written_in_moore("")


def test_without_a_lexicon_it_is_weaker_and_says_so():
    """The fallback accepts any two words without a French marker. That is a
    choice, and this test exists so nobody mistakes it for the real measure."""
    assert moore.written_in_moore("xyzzy plugh", None)
    assert not moore.written_in_moore("xyzzy plugh", LEXICON)


# The twelve real answers of `P2-6120-checkpoint-218`, cut to their first four
# words: the checkpoint bits per character had ranked best of five.
COLLAPSED = [
    "Ned sã n paam tʋʋm-sõngo,",
    "Ned sã n belg a",
    "Ned sã n paam tɩ",
    "Ned sã n maan yam,",
    "Ned sã n paam tʋʋmde,",
    "Ned sã n pa ki",
    "Ned sã n ka tar",
    "Ned sã n paam bũmb",
    "Ned sã n tõe n",
    "Ned sã n dat n",
    "Ned sã n maan tʋʋm-sõngo,",
    "Ned sã n maan yel-wẽnde,",
]


def test_one_opening_for_twelve_questions_is_a_collapse():
    """The defect bits per character cannot see: they are computed with teacher
    forcing and never look at generation."""
    distinct, largest = moore.openings(COLLAPSED)
    assert (distinct, largest) == (1, 12)
    assert not moore.speaks_to_each(COLLAPSED)


def test_openings_count_two_words_not_one():
    """`Ned sã`, `Ned fãa` and `Ned ka` are three openings of one mould. At one
    word they merge into `Ned` and the real variety disappears."""
    three = ["Ned sã n a.", "Ned fãa n b.", "Ned ka n c."]
    assert moore.openings(three) == (3, 1)
    assert moore.openings(three, words=1) == (1, 3)
    assert moore.speaks_to_each(three)


def test_the_majority_is_the_line_and_it_is_tight():
    """Six answers of twelve pass, seven do not. It is a decision, not a
    measurement, and the nearest checkpoint that passes sits at five: the
    margin is one answer."""
    twelve = lambda n: ["Ned sã x"] * n + [f"Opening{i} y" for i in range(12 - n)]  # noqa: E731
    assert moore.speaks_to_each(twelve(6))
    assert not moore.speaks_to_each(twelve(7))


def test_no_answer_does_not_reject():
    """An empty list is not a collapse, it is a missing measurement."""
    assert moore.speaks_to_each([])
    assert moore.openings([]) == (0, 0)


# The twelve real answers of `P2@6120 + SFT checkpoint-218` sampled at `T=1.0`,
# seed 20260917. Four different questions receive the SAME whole sentence.
T1_REAL = [
    "Waoogd-a soab wata ne bãane.",
    "Wẽn-zoɛtb wõosg ka paoogd ye.",
    "Wʋm a yell ka ningd a yãnd ye.",
    "Wʋm a zugu, la f ra wʋm a goamã fãa n yɩɩd ye.",
    "Wẽn-zoɛtb wõosg ka paoogd ye.",
    "Wẽn-zoɛtb wõosg ka pãng ye.",
    "Wẽng sã n paamd fo sẽn ka pãng soaba, f ya wãn-wãn soaba?",
    "Wʋm a goama, la f ra wʋm a poorẽ goamã ye.",
    "Wẽn-zoɛtb wõosg ka paoogd ye.",
    "Wẽng sã n paamd namsdbã, b na n teel-b-la ne yõodo.",
    "Wẽn-zoɛtb wõosg ka paoogd ye.",
    "Wʋm a goama la f ra gom ye.",
]

# The twelve greedy answers of the SAME checkpoint. All distinct, all under one
# opening.
GREEDY_REAL = [
    "Ned sã n maan woto, a paamda sũ-noogo.",
    "Ned sã n belg a to, a me yaa belsg soaba.",
    "Ned sã n paam tɩ b gomd a yell bɩ a ra sũ-sãang ye.",
    "Ned sã n maan yam, a tõe n sõnga a taab tɩ b paam bũmbu.",
    "Ned sã n paam bũmb n tɩg n paam n paase, a pa tõe n paam n paase.",
    "Ned sã n pa ki pĩnd sɩɩgr yaa toogo.",
    "Ned sã n ka tar tʋʋmde, a ka tõe n paam ligd ye.",
    "Ned sã n paam bũmb n tɩgd n yɩɩd a sẽn datã, a paamda a menga.",
    "Ned sã n paam tɩ b gomd a yell neba, a paamda yam.",
    "Ned sã n dat n teel naong rãmba, a na n tõk-a-la ne a menga.",
    "Ned sã n maan tʋʋmde, a paamda a sẽn dat n maan bũmb ningã.",
    "Ned sã n maan yel-wẽnde, bɩ a bao n bãnge.",
]


def test_the_two_gates_disagree_on_the_real_passes():
    """The opening check rejects the pass where all twelve answers DIFFER, and
    accepts the one where the same sentence comes back four times. That is why
    the repeat check exists."""
    # greedy: twelve distinct sentences, one opening
    assert moore.repeats(GREEDY_REAL) == (12, 1)
    assert moore.openings(GREEDY_REAL) == (1, 12)
    assert not moore.speaks_to_each(GREEDY_REAL), "opening check rejects"
    assert moore.answers_each(GREEDY_REAL), "repeat check clears"

    # T=1.0: nine distinct sentences, one given four times, four openings
    assert moore.repeats(T1_REAL) == (9, 4)
    assert moore.openings(T1_REAL)[0] == 4
    assert moore.speaks_to_each(T1_REAL), "opening check cleared it"
    assert not moore.answers_each(T1_REAL), "repeat check rejects"


def test_the_repeat_check_changes_no_greedy_verdict():
    """The rule is additive: twelve distinct greedy answers stay accepted."""
    assert moore.answers_each(GREEDY_REAL)
    assert moore.repeats(GREEDY_REAL)[0] == len(GREEDY_REAL)


def test_the_repeat_check_can_fail_and_is_not_a_majority_rule():
    """The line is TWO, not a majority: giving one sentence to two different
    questions is not a weak answer, it is no answer to one of them. A check
    that cannot fail checks nothing."""
    assert moore.answers_each(["a", "b", "c"])
    assert not moore.answers_each(["a", "a", "b", "c", "d", "e", "f", "g"])
    # and the majority rule of the opening check would let this through
    assert moore.speaks_to_each(["a", "a", "b", "c", "d", "e", "f", "g"])
    assert moore.answers_each([]) is True
