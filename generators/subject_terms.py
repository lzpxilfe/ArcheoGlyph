# -*- coding: utf-8 -*-
"""
Recognise Korean archaeological terms in what the user types.

The plugin knows 188 artefact and feature types in both Korean and English
(template_catalog.py plus i18n_ko.py), but the AI backends only ever saw the
user's raw note. A Korean archaeologist writing "빗살무늬토기 조각" therefore got
no benefit from that vocabulary: the model had to guess the subject from the
photograph alone.

This turns those terms into a short English subject line the model can act on.
It changes nothing about what the user typed - their note is still passed
through verbatim - and it adds nothing when no term is recognised.

Nothing here imports QGIS, so it is testable on its own.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .template_catalog import TEMPLATE_INFO

# Hangul syllables, plus the Hanja that appear in a few type names (凸, 呂).
_HANGUL = re.compile(r"[가-힣]")

# A term this short is a syllable as much as a word - "논" sits inside 논의 and
# 논문 - so it only matches when nothing Korean is written against it.
BOUNDARY_ONLY_LENGTH = 1

# Syllables that are common words in their own right, where even a
# boundary-matched hit would more often be prose than an artefact type.
_SKIP = {"구"}


def _split(label: str) -> Tuple[str, List[str]]:
    """Separate a display name into its head and its bracketed parts."""
    label = str(label or "").strip()
    head = re.split(r"[(（]", label)[0].strip()
    brackets = [
        inner.strip()
        for inner in re.findall(r"[(（]([^)）]*)[)）]", label)
        if inner.strip()
    ]
    return head, brackets


def _usable(form: str) -> bool:
    """Whether a term is specific enough to match on at all."""
    return bool(form) and form not in _SKIP and bool(_HANGUL.search(form))


def _needs_boundary(term: str) -> bool:
    """Whether a term may only match as a whole word."""
    return len(term) <= BOUNDARY_ONLY_LENGTH


def _is_bounded(text: str, start: int, end: int) -> bool:
    """True when the match is not part of a longer Korean word."""
    before = text[start - 1] if start > 0 else ""
    after = text[end] if end < len(text) else ""
    return not _HANGUL.match(before or " ") and not _HANGUL.match(after or " ")


def _forms(label: str, shared_head: bool) -> List[str]:
    """
    The searchable forms of a display name.

    A bracket means one of two things, and they need opposite treatment:

    * a synonym, when the head is that type's own name - "굽다리접시 (고배)",
      "검은간토기 (흑도장경호)". Either form identifies the type.
    * the distinguishing part, when the head is a generic term shared with
      other entries - "고분 (즙석)", "고분 (원분)". Here the head alone cannot
      identify the type, so the whole label and the bracket carry the meaning.

    ``shared_head`` says which case this is.
    """
    head, brackets = _split(label)
    forms = []
    if label != head:
        # Written in full, brackets and all.
        forms.append(label)
    forms.extend(brackets)
    if head and not shared_head:
        forms.append(head)
    elif head:
        # A generic head still names the generic type; first entry claims it.
        forms.append(head)
    # A slash separates two names for the same thing: "팔찌 / 반지".
    expanded = []
    for form in forms:
        expanded.extend(part.strip() for part in form.split("/"))
    return [f for f in expanded if f]


def _build_index() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build the term index.

    Returns ``(index, unmatchable)`` - the second names the templates that
    have no term specific enough to search for, with the reason, so nothing
    drops out silently.
    """
    from ..i18n_ko import CATALOG

    heads: Dict[str, int] = {}
    for english in TEMPLATE_INFO:
        korean = CATALOG.get(english)
        if korean:
            head, _brackets = _split(korean)
            heads[head] = heads.get(head, 0) + 1

    index: Dict[str, str] = {}
    unmatchable: Dict[str, str] = {}
    for english in TEMPLATE_INFO:
        korean = CATALOG.get(english)
        if not korean:
            unmatchable[english] = "no Korean name"
            continue
        head, _brackets = _split(korean)
        usable = [f for f in _forms(korean, heads.get(head, 0) > 1) if _usable(f)]
        if not usable:
            unmatchable[english] = f"{korean!r} is too common a word to match on"
            continue
        for form in usable:
            # First entry wins, so a shared generic term keeps one meaning.
            index.setdefault(form, english)
    return index, unmatchable


_INDEX: Dict[str, str] = {}
_UNMATCHABLE: Dict[str, str] = {}


def _ensure_index() -> None:
    global _INDEX, _UNMATCHABLE
    if not _INDEX:
        _INDEX, _UNMATCHABLE = _build_index()


def term_index() -> Dict[str, str]:
    """The Korean-to-English term index, built once."""
    _ensure_index()
    return _INDEX


def unmatchable_terms() -> Dict[str, str]:
    """Templates with no searchable Korean term, and why."""
    _ensure_index()
    return _UNMATCHABLE


def find_subjects(text: str) -> List[Tuple[str, str]]:
    """
    The artefact types named in ``text``, as (Korean term, English name).

    Longer terms win: "돌화살촉" must not be reported as "화살촉", and
    "탁자식 지석묘" must not be reported as "지석묘". Matched spans are consumed,
    so each part of the text contributes at most one type. Results keep the
    order they appear in the text.
    """
    text = str(text or "")
    if not text:
        return []

    index = term_index()
    # Longest first so a specific type is matched before the generic one it
    # contains.
    terms = sorted(index, key=len, reverse=True)

    claimed = [False] * len(text)
    found: List[Tuple[int, str, str]] = []
    for term in terms:
        bounded = _needs_boundary(term)
        start = text.find(term)
        while start != -1:
            end = start + len(term)
            if not any(claimed[start:end]) and (
                not bounded or _is_bounded(text, start, end)
            ):
                for i in range(start, end):
                    claimed[i] = True
                found.append((start, term, index[term]))
            start = text.find(term, start + 1)

    seen = set()
    ordered = []
    for _position, term, english in sorted(found):
        if english in seen:
            continue
        seen.add(english)
        ordered.append((term, english))
    return ordered


def subject_hint(text: str, limit: int = 3) -> str:
    """
    An English subject line for the prompt, or "" when nothing is recognised.

    Kept short deliberately: it tells the model what the object is, and says
    explicitly that this is not licence to draw features the photograph does
    not show.
    """
    subjects = find_subjects(text)[:limit]
    if not subjects:
        return ""
    named = ", ".join(f"{english} ({korean})" for korean, english in subjects)
    return (
        f"SUBJECT: the user names this artefact type in Korean: {named}. "
        "Use it to read the photograph correctly - not as licence to add "
        "features the photograph does not show."
    )
