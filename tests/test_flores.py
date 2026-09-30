import pytest
from flores import first_line, pairs


def test_pairs_align_by_id_not_by_order():
    french = [{"id": 0, "text": "un"}, {"id": 1, "text": "deux"}]
    moore = [{"id": 1, "text": "yiibu"}, {"id": 0, "text": "yembre"}]
    assert pairs(french, moore) == [("un", "yembre"), ("deux", "yiibu")]


def test_pairs_refuse_a_missing_line():
    with pytest.raises(ValueError):
        pairs([{"id": 5, "text": "cinq"}], [{"id": 0, "text": "yembre"}])


def test_first_line_drops_commentary_and_quotes():
    assert first_line('\n« Ne y taabo »\nCela veut dire bonne année.') == "Ne y taabo"
    assert first_line("   ") == ""



def test_target_language_is_checked_per_direction():
    from flores import in_target_language

    assert in_target_language("A yeelame tɩ b tara yũug.", "french_to_moore")
    assert not in_target_language("Le docteur a averti que la recherche commence.",
                                  "french_to_moore")
    assert in_target_language("Le docteur a averti que la recherche commence.",
                              "moore_to_french")


def test_bootstrap_is_paired():
    import numpy as np
    from flores_bootstrap import interval, sentence_stats

    refs = ["a b c d", "e f g h", "i j k l"]
    good = sentence_stats(["a b c d", "e f g h", "i j k x"], refs)
    draws = [np.array([0, 1, 2]), np.array([2, 2, 0])]
    assert interval(good, good, draws) == (0.0, 0.0, 0.0)
    diff, lo, hi = interval(good, sentence_stats(["x", "y", "z"], refs), draws)
    assert diff > 0 and lo > 0
