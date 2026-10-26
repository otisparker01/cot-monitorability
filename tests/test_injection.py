"""Mistake-injection helpers: diff check, fence stripping, judge parsing (no model calls)."""

import pytest

from evals.oversight.perturbation.mistake_injection import (
    FRAMINGS,
    _strip_fences,
    cut_after_error,
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


def test_cut_after_error_sentence():
    orig = "A is 5. B is A plus 2, so B is 7. Then C follows."
    corr = "A is 6. B is A plus 2, so B is 7. Then C follows."
    cut = cut_after_error(orig, corr)
    assert cut["corr_cut"] == "A is 6." and cut["orig_cut"] == "A is 5."
    assert cut["changes"] == [("5.", "6.")] and cut["planted_context"] == "A is 6."


def test_cut_after_error_line_and_decimals():
    orig = "Given x = 3\nCity tax: 0.02 × 100 = 2 dollars\nTotal = 102. Done."
    corr = "Given x = 3\nCity tax: 0.03 × 100 = 2 dollars\nTotal = 102. Done."
    cut = cut_after_error(orig, corr)
    assert cut["corr_cut"] == "Given x = 3\nCity tax: 0.03 × 100 = 2 dollars"
    assert cut["orig_cut"] == "Given x = 3\nCity tax: 0.02 × 100 = 2 dollars"
    assert cut["planted_context"] == "City tax: 0.03 × 100 = 2 dollars"


def test_cut_after_error_includes_later_spans_in_the_same_sentence_only():
    orig = "Rate is 2% so tax is 4 here. Later 4 is used."
    corr = "Rate is 3% so tax is 6 here. Later 6 is used."
    cut = cut_after_error(orig, corr)
    assert cut["corr_cut"] == "Rate is 3% so tax is 6 here."
    assert cut["changes"] == [("2%", "3%"), ("4", "6")]


def test_cut_after_error_none_when_identical():
    assert cut_after_error("same text.", "same text.") is None


def test_strip_drops_unquoted_echoed_header():
    raw = f"Below is the first part of an analysis of a multiple-choice question:\n\n{PREFIX}"
    assert _strip_fences(raw, PREFIX) == PREFIX
