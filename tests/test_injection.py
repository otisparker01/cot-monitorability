"""Mistake-injection helpers: diff check, fence stripping, judge parsing (no model calls)."""

import pytest

from evals.oversight.perturbation.mistake_injection import (
    FRAMINGS,
    _strip_fences,
    diff_check,
    mistake_injection,
    parse_judge,
    word_diff,
)

PREFIX = "Step 1: 2 + 2 = 4. Step 2: multiply by 3 to get 12. Step 3: subtract one"


def test_word_diff_marks_the_edit():
    diff, spans = word_diff(PREFIX, PREFIX.replace("12", "13"))
    assert spans == 1
    assert "[-12.-]{+13.+}" in diff


def test_diff_check_flags():
    assert diff_check(PREFIX, " ".join(PREFIX.split()))["flag_identical"]
    assert diff_check(PREFIX, PREFIX[:20])["flag_length"]
    assert diff_check(PREFIX, PREFIX + " so ANSWER: B")["flag_answer"]
    ok = diff_check(PREFIX, PREFIX.replace("12", "13"))
    assert not ok["flagged"] and ok["length_ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize("raw", [
    f'"""\n{PREFIX}\n"""',
    f'Below is the first part of an analysis of a multiple-choice question:\n\n"""\n{PREFIX}',
    f'{PREFIX}\n"""',
    PREFIX,
])
def test_strip_fences(raw):
    assert _strip_fences(raw, PREFIX) == PREFIX


def test_strip_fences_leaves_text_alone_when_prefix_has_quotes():
    raw = 'def f():\n    """doc"""'
    assert _strip_fences(raw, 'def f():\n    """doc') == raw


@pytest.mark.parametrize("text, want", [
    ("ADOPTED", "ADOPTED"),
    ("corrected.", "CORRECTED"),
    ("NOT_USED", "NOT_USED"),
    ("Not used", "NOT_USED"),
    ("I am unsure", "UNPARSED"),
])
def test_parse_judge(text, want):
    assert parse_judge(text) == want


def test_task_builds_for_each_framing():
    for framing in FRAMINGS:
        assert mistake_injection(framing=framing).metadata["framing"] == framing
    with pytest.raises(ValueError):
        mistake_injection(framing="nope")
