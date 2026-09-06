"""
Recognising Korean artefact terms in the user's note.

This is what lets the bilingual catalogue reach the AI backends, so the risks
are getting the wrong type (a generic term swallowing a specific one) and
firing on ordinary prose.
"""

import pytest

from archeoglyph.generators import subject_terms
from archeoglyph.generators.subject_terms import find_subjects, subject_hint, term_index
from archeoglyph.generators.template_catalog import TEMPLATE_INFO
from archeoglyph.i18n_ko import CATALOG


def _english(text):
    return [name for _korean, name in find_subjects(text)]


# -- the index ----------------------------------------------------------

def test_every_template_is_matchable_or_has_a_stated_reason():
    """
    The point of the matcher is the catalogue; a type missing from the index
    is a type the AI never hears about. Anything left out must say why, so
    nothing drops out silently.
    """
    indexed = set(term_index().values())
    excluded = subject_terms.unmatchable_terms()
    unexplained = [
        name for name in TEMPLATE_INFO
        if name not in indexed and name not in excluded
    ]
    assert not unexplained, f"templates absent from the term index: {unexplained}"
    for name, reason in excluded.items():
        assert reason and name in TEMPLATE_INFO


def test_each_template_is_found_by_its_own_korean_name():
    """Writing a type's Korean name must identify that exact type."""
    unfound = []
    for name in TEMPLATE_INFO:
        korean = CATALOG.get(name)
        if not korean or name in subject_terms.unmatchable_terms():
            continue
        if name not in _english(korean):
            unfound.append(f"{korean} -> expected {name}, got {_english(korean)}")
    assert not unfound, "\n".join(unfound)


def test_index_maps_to_real_template_names():
    unknown = sorted(set(term_index().values()) - set(TEMPLATE_INFO))
    assert not unknown, f"index points at names that are not templates: {unknown}"


# -- matching -----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("빗살무늬토기 조각", ["Comb-pattern Pottery"]),
    ("고배 실측도", ["Mounted Dish (Gobae)"]),
    ("굽다리접시", ["Mounted Dish (Gobae)"]),          # the bracketed synonym
    ("흑도장경호", ["Black Burnished Long-necked Jar"]),
    ("이 사진은 옹관묘입니다", ["Jar Coffin Tomb"]),
])
def test_terms_are_recognised_inside_a_sentence(text, expected):
    assert _english(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("돌화살촉", "Stone Arrowhead"),
    ("탁자식 지석묘", "Dolmen (Table-type)"),
    ("원형 수혈주거지", "Pit Dwelling (Round)"),
])
def test_a_specific_type_beats_the_generic_term_inside_it(text, expected):
    """
    "돌화살촉" contains "화살촉" and "탁자식 지석묘" contains "지석묘"; reporting the
    generic type would send the model to the wrong shape entirely.
    """
    assert _english(text) == [expected]


def test_the_generic_term_still_matches_on_its_own():
    assert _english("화살촉") == ["Arrowhead"]
    assert _english("지석묘") == ["Dolmen"]


def test_several_terms_keep_the_order_they_appear_in():
    assert _english("세형동검과 다뉴세문경") == [
        "Bronze Dagger (Slender)", "Fine-lined Bronze Mirror"
    ]
    assert _english("다뉴세문경 옆의 세형동검") == [
        "Fine-lined Bronze Mirror", "Bronze Dagger (Slender)"
    ]


def test_a_repeated_term_is_reported_once():
    assert _english("빗살무늬토기, 빗살무늬토기 조각") == ["Comb-pattern Pottery"]


@pytest.mark.parametrize("text", [
    "", None, "   ",
    "a photograph of a pot",
    "please make this look cleaner",
    "사진을 깔끔하게 만들어 주세요",       # Korean, but no artefact type
])
def test_nothing_is_invented_when_no_type_is_named(text):
    assert find_subjects(text) == []


# -- the prompt line ----------------------------------------------------

def test_the_hint_is_empty_when_no_type_is_named():
    assert subject_hint("make it cleaner") == ""
    assert subject_hint("") == ""


def test_the_hint_carries_both_names_and_a_caution():
    hint = subject_hint("빗살무늬토기 조각")
    assert "Comb-pattern Pottery" in hint
    assert "빗살무늬토기" in hint
    # Without this the model treats the type as permission to draw a textbook
    # example instead of the object in the photograph.
    assert "photograph does not show" in hint


def test_the_hint_is_capped():
    """A long note must not turn into a wall of subject terms."""
    text = "빗살무늬토기 세형동검 다뉴세문경 금관 철검"
    matched = find_subjects(text)
    assert len(matched) > 3
    hint = subject_hint(text, limit=2)
    named = [korean for korean, _english in matched if korean in hint]
    assert len(named) == 2


# -- false positives ----------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("논 유구", ["Paddy Field"]),
    ("밭 경작유구", ["Dry Field"]),
    ("끌 사진", ["Chisel"]),
])
def test_a_one_syllable_type_matches_as_a_whole_word(text, expected):
    assert _english(text) == expected


@pytest.mark.parametrize("text", ["논의 결과입니다", "논문에서 인용", "구덩이 사진", "단순한 사진"])
def test_a_one_syllable_type_does_not_match_inside_another_word(text):
    """
    "논" sits inside 논의 and 논문; matching it there would tell the model the
    photograph shows a paddy field.
    """
    assert find_subjects(text) == []


def test_ambiguous_syllables_are_never_matchable():
    for skipped in subject_terms._SKIP:
        assert skipped not in term_index(), f"{skipped} should not be matchable"
