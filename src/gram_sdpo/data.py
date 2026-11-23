# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
r"""Datasets and prompts.

Training uses DAPO-Math-17k. Out-of-distribution tests use AIME24 and AMC23.
Nothing is stored in the repository: files are downloaded from the Hugging
Face Hub the first time they are needed and cached by ``huggingface_hub``.

The prompts follow Kim et al. (2026), who reuse the self-teacher template of
Hübotter et al. (2026). The exact line breaks between the parts are not given
in either paper, so blank lines are used here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from gram_sdpo.logging_utils import get_logger

logger = get_logger(__name__)

ANSWER_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be "
    "of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem."
    "\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'

HUB_SOURCES: dict[str, tuple[str, str]] = {
    "dapo-math-17k": ("BytedTsinghua-SIA/DAPO-Math-17k", "data/dapo-math-17k.parquet"),
    "aime24": ("BytedTsinghua-SIA/AIME-2024", "data/aime-2024.parquet"),
    "amc23": ("zwhe99/amc23", "data/test-00000-of-00001.parquet"),
}


@dataclass(frozen=True)
class Problem:
    """One math problem with its reference answer."""

    problem_id: str
    question: str
    answer: str


def strip_dapo_wrapper(prompt: str) -> str:
    """Remove the instruction text DAPO wraps around each question.

    Args:
        prompt: The user message stored in the DAPO-format datasets.

    Returns:
        The bare question. Text without the wrapper is returned unchanged.
    """
    text = prompt
    if text.startswith(DAPO_PREFIX):
        text = text[len(DAPO_PREFIX) :]
    if text.endswith(DAPO_SUFFIX):
        text = text[: -len(DAPO_SUFFIX)]
    return text.strip()


def format_reference_answer(value: Any) -> str:
    """Turn a stored answer into a string, writing ``27.0`` as ``"27"``.

    Args:
        value: The answer as stored in the dataset.

    Returns:
        The answer as text.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def problems_from_dapo_rows(rows: list[dict[str, Any]]) -> list[Problem]:
    """Convert DAPO-format rows to problems, dropping repeated questions.

    The Hub files repeat each question many times (DAPO-Math-17k holds about
    1.8 million rows for about 17 thousand questions).

    Args:
        rows: Rows with ``prompt`` (a list holding one user message) and
            ``reward_model`` (a mapping with ``ground_truth``).

    Returns:
        One problem per distinct question, in first-seen order.
    """
    seen: set[str] = set()
    problems: list[Problem] = []
    for row in rows:
        question = strip_dapo_wrapper(str(row["prompt"][0]["content"]))
        if question in seen:
            continue
        seen.add(question)
        answer = format_reference_answer(row["reward_model"]["ground_truth"])
        problems.append(Problem(f"q{len(problems):05d}", question, answer))
    return problems


def toy_problems(count: int = 64) -> list[Problem]:
    """Build small addition problems for tests and smoke runs. No download.

    Args:
        count: Number of problems.

    Returns:
        Problems such as "What is 3 + 4?" with answer "7".
    """
    problems = []
    for index in range(count):
        left, right = index % 9 + 1, (index * 7) % 9 + 1
        problems.append(Problem(f"toy{index:03d}", f"What is {left} + {right}?", str(left + right)))
    return problems


def load_problems(name: str) -> list[Problem]:
    """Load a dataset by its short name.

    Args:
        name: ``"dapo-math-17k"``, ``"aime24"``, ``"amc23"``, or ``"toy"``.

    Returns:
        The problems, with repeated questions removed.
    """
    if name == "toy":
        return toy_problems()
    if name not in HUB_SOURCES:
        raise ValueError(f"Unknown dataset {name!r}. Choose from {sorted(HUB_SOURCES)} or 'toy'.")
    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo, filename = HUB_SOURCES[name]
    logger.info("Loading %s from %s", name, repo)
    path = hf_hub_download(repo, filename, repo_type="dataset")
    if name == "amc23":
        frame = pd.read_parquet(path)
        problems = [
            Problem(f"amc{int(row.id):03d}", str(row.question), format_reference_answer(row.answer))
            for row in frame.itertuples()
        ]
    else:
        frame = pd.read_parquet(path, columns=["prompt", "reward_model"])
        frame = frame.assign(_text=frame["prompt"].map(lambda p: p[0]["content"]))
        frame = frame.drop_duplicates("_text")
        problems = problems_from_dapo_rows(frame[["prompt", "reward_model"]].to_dict("records"))
    logger.info("Loaded %d distinct problems from %s", len(problems), name)
    return problems


def split_train_heldout(
    problems: list[Problem], heldout_size: int, seed: int, max_train: int | None = None
) -> tuple[list[Problem], list[Problem]]:
    """Split the training set into a training part and an in-domain test part.

    The split depends only on ``seed``, so training and evaluation agree on
    which problems were held out.

    Args:
        problems: All problems of the training dataset.
        heldout_size: Number of problems to hold out.
        seed: Seed for the shuffle.
        max_train: Optional cap on the training part.

    Returns:
        ``(train, heldout)``.
    """
    if heldout_size >= len(problems):
        raise ValueError(f"heldout_size {heldout_size} leaves no training problems")
    shuffled = list(problems)
    random.Random(seed).shuffle(shuffled)
    heldout, train = shuffled[:heldout_size], shuffled[heldout_size:]
    if max_train is not None:
        train = train[:max_train]
    return train, heldout


def student_messages(question: str) -> list[dict[str, str]]:
    """Chat messages for an unguided attempt at ``question``."""
    return [{"role": "user", "content": f"{question}\n\n{ANSWER_INSTRUCTION}"}]


def teacher_messages(question: str, solution: str) -> list[dict[str, str]]:
    """Chat messages for the self-teacher, which also sees a correct solution.

    Args:
        question: The problem text.
        solution: A correct rollout, with or without its reasoning.

    Returns:
        One user message: question, answer instruction, the solution, and the
        closing line from the self-teacher template.
    """
    content = (
        f"{question}\n\n{ANSWER_INSTRUCTION}\n\n"
        f"Correct solution:\n{solution}\n\n"
        "Correctly solve the original question."
    )
    return [{"role": "user", "content": content}]


def render_prompt(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool) -> str:
    """Apply the model's chat template and open the assistant turn.

    Args:
        tokenizer: A Hugging Face tokenizer with a chat template.
        messages: The chat messages.
        enable_thinking: Passed to the template. Qwen3 uses it to switch
            thinking mode. Templates that do not know the flag ignore it.

    Returns:
        The prompt text, ending where the model's response begins.
    """
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
