# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""Tests for answer checking and the epistemic token count."""

import pytest

from gram_sdpo.scoring import (
    EPISTEMIC_MARKERS,
    answers_match,
    epistemic_token_count,
    extract_boxed,
    is_correct,
    normalize_answer,
    strip_thinking,
)


def test_there_are_ten_markers():
    """Kim et al. (2026) define exactly ten uncertainty markers."""
    assert len(EPISTEMIC_MARKERS) == 10


def test_epistemic_count_is_case_insensitive_and_whole_word():
    """The word "wait" counts in any case. "checked" and "waiting" do not."""
    text = "Wait, hmm. Maybe I should check. I checked it while waiting. Perhaps."
    assert epistemic_token_count(text) == 5


def test_epistemic_count_of_empty_text_is_zero():
    """No text, no markers."""
    assert epistemic_token_count("") == 0


def test_strip_thinking_removes_the_block():
    """Only the text outside the thinking tags survives."""
    assert (
        strip_thinking("<think>wait, 2+2\nis 4</think>\n\nThe answer is 4.") == "The answer is 4."
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("so \\boxed{42}", "42"),
        ("\\boxed{\\frac{1}{2}} done", "\\frac{1}{2}"),
        ("first \\boxed{1} then \\boxed{2}", "2"),
        ("\\boxed {7}", "7"),
        ("no box here", None),
        ("cut off \\boxed{12", None),
    ],
)
def test_extract_boxed(text, expected):
    """The last complete box wins, with nested braces kept."""
    assert extract_boxed(text) == expected


def test_extract_boxed_falls_back_to_an_earlier_complete_box():
    """A truncated final box does not hide a complete one before it."""
    assert extract_boxed("\\boxed{5} and then \\boxed{6") == "5"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(" $1,000$ ", "1000"), ("\\text{42}", "42"), ("27.0", "27"), ("3.", "3"), ("x + 1", "x+1")],
)
def test_normalize_answer(raw, expected):
    """Formatting noise is removed before comparison."""
    assert normalize_answer(raw) == expected


def test_answers_match_numbers_written_differently():
    """Equal numbers match whatever their formatting."""
    assert answers_match("0.50", "0.5")
    assert answers_match("\\dfrac{1}{2}", "\\frac{1}{2}")
    assert not answers_match("41", "42")


def test_is_correct_needs_a_box_after_the_thinking():
    """A box inside the reasoning is not the final answer."""
    assert is_correct("<think>hmm \\boxed{3}</think> The answer is \\boxed{34}.", "34")
    assert not is_correct("<think>maybe \\boxed{34}</think> I am not sure.", "34")


def test_is_correct_rejects_truncated_responses():
    """A response cut off before its answer is wrong."""
    assert not is_correct("<think>let me check this very long", "34")
