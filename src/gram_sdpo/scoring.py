# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
r"""Answer checking and the epistemic token count.

Correctness follows the setup Kim et al. (2026) describe: take the final
``\boxed{}`` expression, try an exact match, then a mathematical-equivalence
check. The epistemic token count is their count of ten uncertainty markers.
"""

from __future__ import annotations

import re

from gram_sdpo.logging_utils import get_logger

logger = get_logger(__name__)

EPISTEMIC_MARKERS: tuple[str, ...] = (
    "wait",
    "hmm",
    "perhaps",
    "maybe",
    "actually",
    "alternatively",
    "seems",
    "might",
    "likely",
    "check",
)
_MARKER_PATTERN = re.compile(r"\b(" + "|".join(EPISTEMIC_MARKERS) + r")\b", re.IGNORECASE)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def epistemic_token_count(text: str) -> int:
    """Count the ten uncertainty markers of Kim et al. (2026) in ``text``.

    Matching is case-insensitive and on whole words, so "checked" is not
    counted as "check".

    Args:
        text: A model response, thinking included.

    Returns:
        The total number of marker occurrences.
    """
    return len(_MARKER_PATTERN.findall(text))


def strip_thinking(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from a response.

    Args:
        text: A model response.

    Returns:
        The response with every thinking block removed and edges trimmed.
    """
    return _THINK_PATTERN.sub("", text).strip()


def extract_boxed(text: str) -> str | None:
    r"""Return the contents of the last ``\boxed{...}`` in ``text``.

    Braces are matched, so ``\boxed{\frac{1}{2}}`` returns ``\frac{1}{2}``.

    Args:
        text: A model response.

    Returns:
        The boxed contents, or ``None`` when there is no complete box.
    """
    start = text.rfind("\\boxed")
    while start != -1:
        open_brace = text.find("{", start)
        if open_brace != -1 and text[start + len("\\boxed") : open_brace].strip() == "":
            depth = 0
            for index in range(open_brace, len(text)):
                if text[index] == "{":
                    depth += 1
                elif text[index] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[open_brace + 1 : index].strip()
        start = text.rfind("\\boxed", 0, start)
    return None


def normalize_answer(answer: str) -> str:
    r"""Put an answer string in a simple canonical form for exact matching.

    Args:
        answer: A raw answer such as ``" $1,000$ "`` or ``"\text{42}"``.

    Returns:
        The answer without dollar signs, spacing, thousands separators,
        ``\text{}`` wrappers, a trailing period, or a trailing ``.0``.
    """
    result = answer.strip().strip("$").strip()
    result = re.sub(r"\\text\{([^}]*)\}", r"\1", result)
    result = result.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    result = result.replace("dfrac", "frac").replace("tfrac", "frac")
    result = re.sub(r"(?<=\d),(?=\d{3}\b)", "", result)
    result = re.sub(r"\s+", "", result)
    result = result.rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", result):
        result = result.split(".")[0]
    return result


def _as_float(value: str) -> float | None:
    """Parse a plain decimal number, returning ``None`` for anything else."""
    try:
        return float(value)
    except ValueError:
        return None


def answers_match(predicted: str, truth: str) -> bool:
    """Decide whether two answer strings denote the same value.

    Tries an exact match on normalized strings, then a numeric comparison,
    then ``math_verify`` when that package is installed.

    Args:
        predicted: The model's answer.
        truth: The reference answer.

    Returns:
        ``True`` when the answers agree.
    """
    left, right = normalize_answer(predicted), normalize_answer(truth)
    if left == right:
        return True
    left_number, right_number = _as_float(left), _as_float(right)
    if left_number is not None and right_number is not None:
        return abs(left_number - right_number) <= 1e-6 * max(1.0, abs(right_number))
    try:
        from math_verify import parse, verify
    except ImportError:
        return False
    try:
        return bool(verify(parse(f"${truth}$"), parse(f"${predicted}$")))
    except Exception as error:  # noqa: BLE001 - a malformed answer must never crash a run.
        logger.debug("math_verify could not compare %r and %r: %s", predicted, truth, error)
        return False


def is_correct(response: str, truth: str) -> bool:
    r"""Score one response against the reference answer.

    Args:
        response: The full model response. Only text after the last thinking
            block is searched, so a ``\boxed{}`` inside the reasoning does not
            count as the final answer.
        truth: The reference answer.

    Returns:
        ``True`` when the final boxed answer matches ``truth``. A response
        with no complete box, such as one cut off by the token limit, is wrong.
    """
    final_part = response.rsplit("</think>", 1)[-1]
    predicted = extract_boxed(final_part)
    if predicted is None:
        return False
    return answers_match(predicted, truth)
