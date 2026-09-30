"""Moore orthography: the alphabet and the cheap tests built on it.

Shares only what eight extractors had identically. `is_moore` stays in each
extractor because the three existing versions apply different filters, and a
common signature would mean unused parameters or silent ones.
"""

from __future__ import annotations

import collections
import re
import unicodedata
from typing import Final, Iterable

# 25 letters plus five nasals, per *Moor ɡom wʋɡbo*, CIER de Guiè (2023).
ALPHABET: Final[frozenset[str]] = frozenset(
    "a ã b d e ẽ ɛ f g h i ĩ ɩ k l m n o õ p r s t u ũ ʋ v w y z".split())

# Letters neither French nor English carries, so their presence identifies Moore
# on its own. `ɔ` and `ŋ` are excluded: they are not Moore and signal an
# encoding error.
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


# French words Moore does not also have. What the list LEAVES OUT is the point:
# `la`, `a`, `be` and `to` are Moore words.
FRENCH: Final[frozenset[str]] = frozenset(
    "le les des une est pas qui que dans pour avec sur son sa ses ils elle nous"
    " vous leur cette ce au aux du il elles mais donc par plus tout tous comme"
    " quand sont etre avoir fait dit".split())


def written_in_moore(text: str, lexicon: Iterable[str] | None = None,
                     coverage: float = 0.6) -> bool:
    """Whether an answer is written in Moore.

    `has_unique_letter` alone is **not** enough, and that is the whole reason
    this exists. `Maan bala, koosda bala.` is pure Moore and carries no letter
    exclusive to it: judging on the alphabet alone once called 392 Moore
    answers French.

    The second trap is the fix for the first. A French word list must not carry
    `la`, `a`, `be` or `to`, which are Moore words; a list that did called
    `A yi.` and `A yaa sabelle.` French. Hence `FRENCH`, and hence the lexicon.

    Pass `lexicon` (the corpus forms) whenever it is at hand: it is the only
    witness that recognises Moore written with no distinctive letter at all.
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

    The other half of what can be counted without a reader, and it costs
    nothing: the answers are already on disk. A generator holding
    one sentence cannot answer twelve different questions, whatever its bits
    per character say, because those are teacher-forced and never look at what
    generation produces.

    Two words, because that is where the Moore maxim template shows: `Ned sã`,
    `Ned fãa`, `Ned ka` are three different openings of one same mould, and
    counting one word would merge them into `Ned` and hide the variety that
    does exist.
    """
    groups = collections.Counter(" ".join(a.split()[:words]) for a in answers)
    return len(groups), max(groups.values(), default=0)


def speaks_to_each(answers: Iterable[str], words: int = 2) -> bool:
    """False when one single opening answers a majority of the questions.

    A DECISION, not a measurement: the majority is the line because no single
    form can be the answer to most of a set of different questions. On one
    run it rejects only the collapsed checkpoint, `Ned sã` twelve times out of
    twelve; the nearest checkpoint it lets through sits at five of twelve, so
    the margin is one answer and the rule is coarse.

    It is deliberately not a curve stop signal. Collapse *loosens* as training
    goes on, the opposite of overfitting, so a failing point is excluded from
    the points one may keep, and the curve continues past it.
    """
    answers = list(answers)
    if not answers:
        return True
    return openings(answers, words)[1] * 2 <= len(answers)


def repeats(answers: Iterable[str]) -> tuple[int, int]:
    """(distinct answers, size of the largest identical group), whole strings.

    `openings` looks at the first two words and is blind to everything after
    them, in both directions. On two decoding passes of one checkpoint it got
    both wrong at once: the greedy pass gives **twelve distinct
    sentences** all opening `Ned sã` and `speaks_to_each` rejects it, while the
    `T=1,0` pass answers four different questions with the same complete
    sentence, `Wẽn-zoɛtb wõosg ka paoogd ye.`, and it passes.

    So the two counts are not interchangeable and neither replaces the other:
    one sees a shared mould, this one sees a repeated string.
    """
    groups = collections.Counter(answers)
    return len(groups), max(groups.values(), default=0)


def answers_each(answers: Iterable[str]) -> bool:
    """False when the same complete answer is given to two questions.

    A DECISION, and a strict one on purpose: the line is *twice*, not a
    majority as in `speaks_to_each`. Giving one same sentence to two different
    questions is not a weak answer, it is the absence of an answer to one of
    them, and no threshold discussion is needed to say so.

    It fires on what `speaks_to_each` clears, which is the only reason it
    exists.
    """
    answers = list(answers)
    if not answers:
        return True
    return repeats(answers)[1] < 2


def outside_alphabet(text: str) -> set[str]:
    """Letters absent from the Moore alphabet.

    For quality reporting, not filtering: `c j q x` mark French borrowings.
    """
    folded = unicodedata.normalize("NFC", text).casefold()
    return {c for c in folded if c.isalpha() and c not in ALPHABET}


# Letters a French borrowing legitimately brings. Moore press carries them in
# 4.93 % of its lines, so treating them as junk would throw away real text.
BORROWED: Final[frozenset[str]] = frozenset("cjqxéèêëàâôûîïçùœæ")


def foreign_letters(text: str) -> set[str]:
    """Letters that are neither Moore nor a French borrowing.

    The filtering counterpart of `outside_alphabet`. What it catches is broken
    encoding or another language entirely: `ɔ` and `ŋ` are not Moore letters
    and signal an encoding error, `ə ƙ ↄ` come from other orthographies.
    """
    return outside_alphabet(text) - BORROWED
