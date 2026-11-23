# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""The per-example computation: SDPO loss, Gram anchoring, and the KL baseline.

For one question ``x``, one sampled response ``y``, and one correct rollout
``s``, three forward passes share the same weights:

1. Student (adapters on) reads ``x, y``. This pass carries gradients.
2. Self-teacher (adapters off) reads ``x, s, y``. It gives the SDPO target.
3. Base model (adapters off) reads ``x, y``. It gives the Gram anchor and the
   KL-baseline target. This pass runs only when one of those terms is on.
"""

from __future__ import annotations

import random
from typing import Any

import torch

from gram_sdpo.config import Config
from gram_sdpo.data import render_prompt, student_messages, teacher_messages
from gram_sdpo.logging_utils import get_logger
from gram_sdpo.losses import gram_loss, sample_token_indices, token_divergence
from gram_sdpo.modeling import base_model
from gram_sdpo.rollout import Rollout
from gram_sdpo.scoring import strip_thinking

logger = get_logger(__name__)


def choose_solution(
    group: list[Rollout], correct: list[bool], index: int, rng: random.Random
) -> str | None:
    """Pick the correct rollout the self-teacher will see for one attempt.

    This follows the self-teacher template of Hübotter et al. (2026): a
    successful attempt is its own solution, and a failed attempt gets a
    successful rollout from the same group when there is one.

    Args:
        group: All rollouts sampled for one question.
        correct: Whether each rollout was correct.
        index: Which rollout the solution is for.
        rng: Random source for choosing among several correct rollouts.

    Returns:
        The solution text, or ``None`` when no rollout in the group was correct.
    """
    if correct[index]:
        return group[index].text
    candidates = [rollout.text for rollout, ok in zip(group, correct, strict=True) if ok]
    return rng.choice(candidates) if candidates else None


def context_limit(model: Any) -> int:
    """Return the longest input the model accepts, in tokens."""
    for name in ("max_position_embeddings", "n_positions"):
        value = getattr(model.config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return 1_000_000


def encode_prompt(
    tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool
) -> list[int]:
    """Render chat messages and return the prompt's token ids."""
    text = render_prompt(tokenizer, messages, enable_thinking)
    return tokenizer(text, add_special_tokens=False).input_ids


def forward_response(
    model: Any, prompt_ids: list[int], response_ids: list[int], hidden_layers: list[int] | None
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run the model on ``prompt + response`` and slice out the response part.

    Args:
        model: A causal language model.
        prompt_ids: Tokens of the rendered prompt.
        response_ids: Tokens of the response.
        hidden_layers: Indices into the model's hidden states to return, or
            ``None`` to skip hidden states. Index ``-1`` is the final layer.

    Returns:
        ``(logits, hidden)``. ``logits`` has one row per response token: row
        ``t`` is the distribution that predicts response token ``t``.
        ``hidden`` holds, for each requested layer, the representations at
        the response token positions.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids + response_ids], device=device)
    output = model(input_ids=input_ids, output_hidden_states=hidden_layers is not None)
    start, count = len(prompt_ids), len(response_ids)
    logits = output.logits[0, start - 1 : start + count - 1]
    hidden = []
    if hidden_layers is not None:
        hidden = [output.hidden_states[layer][0, start : start + count] for layer in hidden_layers]
    return logits, hidden


def example_losses(
    model: Any,
    tokenizer: Any,
    config: Config,
    question: str,
    response_ids: list[int],
    solution: str,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor] | None:
    """Compute every loss term for one question, response, and solution.

    Args:
        model: The student: a PEFT model whose adapters can be turned off.
        tokenizer: Its tokenizer.
        config: The run configuration.
        question: The problem text ``x``.
        response_ids: Tokens of the student's sampled response ``y``.
        solution: Text of a correct rollout ``s``.
        generator: Random generator for the Gram token sample.

    Returns:
        A dictionary with ``"total"`` (carries gradients), ``"sdpo"``, and,
        when switched on, ``"gram"`` and ``"kl"``. ``None`` when the example
        cannot be used: the response is too short or the self-teacher's input
        would not fit in the model's context.
    """
    loss_config = config.loss
    thinking = config.model.enable_thinking
    if not loss_config.solution_includes_thinking:
        solution = strip_thinking(solution)
    response = response_ids[: loss_config.train_max_tokens]
    if len(response) < 2:
        return None

    student_prompt = encode_prompt(tokenizer, student_messages(question), thinking)
    teacher_prompt = encode_prompt(tokenizer, teacher_messages(question, solution), thinking)
    if len(teacher_prompt) + len(response) > context_limit(model):
        logger.debug("Skipping an example: self-teacher input exceeds the context limit")
        return None

    use_gram = loss_config.gram_weight > 0
    use_kl = loss_config.kl_weight > 0
    layers = list(loss_config.gram_layers) if use_gram else None
    anchor_on_teacher = use_gram and loss_config.gram_target == "teacher"
    need_base_pass = use_kl or (use_gram and not anchor_on_teacher)

    student_logits, student_hidden = forward_response(model, student_prompt, response, layers)
    with base_model(model):
        teacher_logits, teacher_hidden = forward_response(
            model, teacher_prompt, response, layers if anchor_on_teacher else None
        )
        base_logits, base_hidden = None, []
        if need_base_pass:
            base_logits, base_hidden = forward_response(
                model, student_prompt, response, layers if not anchor_on_teacher else None
            )

    losses = {
        "sdpo": token_divergence(
            student_logits, teacher_logits, loss_config.divergence, loss_config.top_k_logits
        )
    }
    total = losses["sdpo"]
    if use_gram:
        anchor_hidden = teacher_hidden if anchor_on_teacher else base_hidden
        index = sample_token_indices(len(response), loss_config.gram_tokens, generator)
        index = index.to(student_logits.device)
        per_layer = [
            gram_loss(student[index], anchor[index])
            for student, anchor in zip(student_hidden, anchor_hidden, strict=True)
        ]
        losses["gram"] = torch.stack(per_layer).mean()
        total = total + loss_config.gram_weight * losses["gram"]
    if use_kl:
        losses["kl"] = token_divergence(student_logits, base_logits, "kl", loss_config.top_k_logits)
        total = total + loss_config.kl_weight * losses["kl"]
    losses["total"] = total
    return losses


@torch.no_grad()
def gram_distance(
    model: Any, tokenizer: Any, config: Config, question: str, response_ids: list[int]
) -> float:
    """Measure how far the student's Gram matrix is from the base model's.

    This is the drift measure of Experiment 1. It uses the same layers and
    token sample size as the Gram loss, with a fixed token sample, so values
    are comparable across checkpoints.

    Args:
        model: The student: a PEFT model whose adapters can be turned off.
        tokenizer: Its tokenizer.
        config: The run configuration.
        question: The problem text.
        response_ids: Tokens of a fixed probe response.

    Returns:
        The Gram loss between student and base representations, averaged over
        the configured layers. Zero means no drift.
    """
    layers = list(config.loss.gram_layers)
    response = response_ids[: config.loss.train_max_tokens]
    prompt = encode_prompt(tokenizer, student_messages(question), config.model.enable_thinking)
    _, student_hidden = forward_response(model, prompt, response, layers)
    with base_model(model):
        _, base_hidden = forward_response(model, prompt, response, layers)
    generator = torch.Generator().manual_seed(0)
    index = sample_token_indices(len(response), config.loss.gram_tokens, generator)
    index = index.to(student_hidden[0].device)
    values = [
        gram_loss(student[index], base[index])
        for student, base in zip(student_hidden, base_hidden, strict=True)
    ]
    return float(torch.stack(values).mean())
