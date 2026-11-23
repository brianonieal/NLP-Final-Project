# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""Tests for configuration loading, datasets, and prompts."""

import pytest

from gram_sdpo.config import Config, apply_overrides, load_config
from gram_sdpo.data import (
    ANSWER_INSTRUCTION,
    DAPO_PREFIX,
    DAPO_SUFFIX,
    format_reference_answer,
    load_problems,
    problems_from_dapo_rows,
    render_prompt,
    split_train_heldout,
    strip_dapo_wrapper,
    student_messages,
    teacher_messages,
    toy_problems,
)


def test_load_config_reads_values_and_keeps_defaults(tmp_path):
    """Listed keys are applied and the rest keep their defaults."""
    path = tmp_path / "c.yaml"
    path.write_text("loss:\n  gram_weight: 0.5\ntrain:\n  steps: 7\n", encoding="utf-8")
    config = load_config(path)
    assert config.loss.gram_weight == 0.5
    assert config.train.steps == 7
    assert config.model.name == Config().model.name


def test_load_config_rejects_unknown_keys(tmp_path):
    """A typo in a key is an error, not a silent default."""
    path = tmp_path / "c.yaml"
    path.write_text("loss:\n  gram_wieght: 0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="gram_wieght"):
        load_config(path)


def test_overrides_are_parsed_as_yaml(tmp_path):
    """Overrides can set numbers, booleans, and lists."""
    path = tmp_path / "c.yaml"
    path.write_text("{}", encoding="utf-8")
    config = load_config(
        path, ["loss.gram_weight=0.1", "model.enable_thinking=false", "loss.gram_layers=[-1, 14]"]
    )
    assert config.loss.gram_weight == 0.1
    assert config.model.enable_thinking is False
    assert config.loss.gram_layers == [-1, 14]


def test_override_without_equals_is_rejected():
    """An override must name a key and a value."""
    with pytest.raises(ValueError, match="section.key=value"):
        apply_overrides({}, ["loss.gram_weight"])


@pytest.mark.parametrize(
    "override", ["loss.divergence=mse", "loss.gram_target=student", "loss.gram_weight=-1"]
)
def test_invalid_settings_are_rejected(tmp_path, override):
    """Settings that cannot be run fail at load time."""
    path = tmp_path / "c.yaml"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path, [override])


def test_strip_dapo_wrapper():
    """The DAPO instruction text is removed, other text is left alone."""
    assert strip_dapo_wrapper(f"{DAPO_PREFIX}What is 1 + 1?{DAPO_SUFFIX}") == "What is 1 + 1?"
    assert strip_dapo_wrapper("What is 1 + 1?") == "What is 1 + 1?"


def test_format_reference_answer():
    """Whole floats lose their decimal part."""
    assert format_reference_answer(27.0) == "27"
    assert format_reference_answer(2.5) == "2.5"
    assert format_reference_answer(" 34 ") == "34"


def test_problems_from_dapo_rows_drops_repeats():
    """The Hub files repeat each question, so repeats are removed."""
    row = {
        "prompt": [{"content": f"{DAPO_PREFIX}What is 1 + 1?{DAPO_SUFFIX}", "role": "user"}],
        "reward_model": {"ground_truth": "2"},
    }
    problems = problems_from_dapo_rows([row, dict(row)])
    assert len(problems) == 1
    assert problems[0].question == "What is 1 + 1?"
    assert problems[0].answer == "2"


def test_toy_problems_have_correct_answers():
    """Each toy answer is the sum in its question."""
    for problem in toy_problems(10):
        left, right = problem.question.removeprefix("What is ").removesuffix("?").split(" + ")
        assert int(left) + int(right) == int(problem.answer)


def test_load_problems_rejects_unknown_names():
    """Only the known datasets load."""
    assert len(load_problems("toy")) == 64
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_problems("imagenet")


def test_split_is_disjoint_and_repeatable():
    """Training and held-out parts share nothing and depend only on the seed."""
    problems = toy_problems(20)
    train, heldout = split_train_heldout(problems, 5, seed=3)
    again_train, again_heldout = split_train_heldout(problems, 5, seed=3)
    assert len(heldout) == 5 and len(train) == 15
    assert not {p.problem_id for p in train} & {p.problem_id for p in heldout}
    assert heldout == again_heldout and train == again_train


def test_split_caps_training_size_and_rejects_empty_training():
    """``max_train`` trims the training part. Holding out everything fails."""
    train, _ = split_train_heldout(toy_problems(20), 5, seed=0, max_train=3)
    assert len(train) == 3
    with pytest.raises(ValueError, match="no training problems"):
        split_train_heldout(toy_problems(5), 5, seed=0)


def test_teacher_prompt_extends_the_student_prompt_with_the_solution():
    """The self-teacher sees the same question plus a correct solution."""
    student = student_messages("What is 1 + 1?")[0]["content"]
    teacher = teacher_messages("What is 1 + 1?", "It is \\boxed{2}.")[0]["content"]
    assert student.endswith(ANSWER_INSTRUCTION)
    assert teacher.startswith(student)
    assert "Correct solution:\nIt is \\boxed{2}." in teacher
    assert teacher.endswith("Correctly solve the original question.")


def test_render_prompt_uses_the_chat_template(tiny_tokenizer):
    """Rendering returns text that contains the user message."""
    text = render_prompt(tiny_tokenizer, student_messages("What is 1 + 1?"), enable_thinking=False)
    assert "What is 1 + 1?" in text
