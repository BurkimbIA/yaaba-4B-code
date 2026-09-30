"""Moore orthography: the alphabet, and cheap text tests built on it."""

from __future__ import annotations

import collections
import re
import unicodedata
from collections.abc import Iterable
from typing import Final

# 25 letters plus five nasals, per *Moor ɡom wʋɡbo*, CIER de Guiè (2023).
ALPHABET: Final[frozenset[str]] = frozenset(
    "a ã b d e ẽ ɛ f g h i ĩ ɩ k l m n o õ p r s t u ũ ʋ v w y z".split())

# Letters that neither French nor English uses, so one of them marks Moore.
# `ɔ` and `ŋ` are left out: they are not Moore letters and come from encoding
# errors.
UNIQUE: Final[frozenset[str]] = frozenset("ãẽĩõũɛɩʋ")

_WORD: Final = re.compile(r"[^\W\d_]+", re.UNICODE)


def has_unique_letter(text: str) -> bool:
    """Whether the text carries a letter exclusive to Moore.

    Case-folding covers capitals (`Ɛ` folds to `ɛ`); NFC guards against a
    combining tilde being read as two characters.
    """
    return bool(UNIQUE & set(unicodedata.normalize("NFC", text).casefold()))


def lexicon_coverage(text: str, lexicon: Iterable[str]) -> float:
    """Fraction of the text's words present in `lexicon`, 0.0 if it has none."""
    words = _WORD.findall(text.casefold())
    if not words:
        return 0.0
    known = frozenset(lexicon)
    return sum(w in known for w in words) / len(words)


# French words that are not also Moore words. `la`, `a`, `be` and `to` are left
# out because Moore has them too.
FRENCH: Final[frozenset[str]] = frozenset(
    "le les des une est pas qui que dans pour avec sur son sa ses ils elle nous"
    " vous leur cette ce au aux du il elles mais donc par plus tout tous comme"
    " quand sont etre avoir fait dit".split())


def written_in_moore(text: str, lexicon: Iterable[str] | None = None,
                     coverage: float = 0.6) -> bool:
    """Whether an answer is written in Moore.

    A letter unique to Moore settles it, but many Moore sentences have none
    (`Maan bala, koosda bala.`): judging on the alphabet alone once called 392
    Moore answers French. Without such a letter, a text is French when 15 % or
    more of its words are in `FRENCH`. That list leaves out `la`, `a`, `be` and
    `to`, which are also Moore words; with them in it, `A yi.` read as French.

    With `lexicon` (the corpus word forms), the remaining texts are judged on
    lexicon coverage, which recognises Moore that has no distinctive letter.
    Without it, any two words pass.
    """
    if has_unique_letter(text):
        return True
    words = _WORD.findall(text.casefold())
    if not words:
        return False
    if sum(w in FRENCH for w in words) / len(words) >= 0.15:
        return False
    if lexicon is not None:
        return lexicon_coverage(text, lexicon) >= coverage
    return len(words) >= 2


def openings(answers: Iterable[str], words: int = 2) -> tuple[int, int]:
    """(distinct openings, size of the largest group) over a set of answers.

    Counted on the answers already on disk, with no reader. A model that gives
    one sentence to every question can still score well in bits per character,
    which are teacher-forced and never look at generation.

    Two words, because Moore maxims share their first word: `Ned sã`, `Ned fãa`
    and `Ned ka` are three openings, and one word would merge them into `Ned`.
    """
    groups = collections.Counter(" ".join(a.split()[:words]) for a in answers)
    return len(groups), max(groups.values(), default=0)


def speaks_to_each(answers: Iterable[str], words: int = 2) -> bool:
    """False when a single opening answers a majority of the questions.

    The majority threshold is a choice: no single form can answer most of a
    set of different questions. On one run it rejects only the collapsed
    checkpoint (`Ned sã` twelve times out of twelve); the nearest checkpoint it
    accepts has five of twelve, so the margin is one answer.

    It does not stop a curve. Collapse fades as training goes on, so a failing
    point is skipped and later points can still be kept.
    """
    answers = list(answers)
    if not answers:
        return True
    return openings(answers, words)[1] * 2 <= len(answers)


def repeats(answers: Iterable[str]) -> tuple[int, int]:
    """(distinct answers, size of the largest group of identical answers).

    `openings` only sees the first two words. On one checkpoint the greedy pass
    gave twelve distinct sentences all opening with `Ned sã` (rejected by
    `speaks_to_each`), while sampling at T=1.0 gave the same sentence,
    `Wẽn-zoɛtb wõosg ka paoogd ye.`, to four questions (accepted). The two
    counts catch different failures.
    """
    groups = collections.Counter(answers)
    return len(groups), max(groups.values(), default=0)


def answers_each(answers: Iterable[str]) -> bool:
    """False when the same complete answer is given to two questions.

    The threshold is two, stricter than the majority of `speaks_to_each`: one
    sentence given to two different questions leaves one of them unanswered.
    It catches repeats that `speaks_to_each` accepts.
    """
    answers = list(answers)
    if not answers:
        return True
    return repeats(answers)[1] < 2


def outside_alphabet(text: str) -> set[str]:
    """Letters absent from the Moore alphabet, for quality reports.

    `c j q x` usually mark French borrowings; filter with `foreign_letters`.
    """
    folded = unicodedata.normalize("NFC", text).casefold()
    return {c for c in folded if c.isalpha() and c not in ALPHABET}


# Letters a French borrowing brings. They appear in 4.93 % of the lines of
# Moore press, so they are not treated as noise.
BORROWED: Final[frozenset[str]] = frozenset("cjqxéèêëàâôûîïçùœæ")


def foreign_letters(text: str) -> set[str]:
    """Letters that are neither Moore nor from a French borrowing, for filtering.

    They come from broken encoding (`ɔ` and `ŋ` are not Moore letters) or from
    other orthographies (`ə ƙ ↄ`).
    """
    return outside_alphabet(text) - BORROWED
