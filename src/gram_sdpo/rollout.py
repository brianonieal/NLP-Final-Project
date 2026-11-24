# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""Sampling responses from the model, in batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from gram_sdpo.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class Rollout:
    """One sampled response.

    Attributes:
        text: The decoded response.
        token_ids: The generated tokens, ending with the end-of-turn token
            when the model produced one.
        finished: ``False`` when generation hit the token limit first.
    """

    text: str
    token_ids: list[int]
    finished: bool


def end_token_ids(model: Any, tokenizer: Any) -> set[int]:
    """Collect every token id that ends a response for this model."""
    ids: set[int] = set()
    for source in (tokenizer.eos_token_id, getattr(model.generation_config, "eos_token_id", None)):
        if isinstance(source, int):
            ids.add(source)
        elif source is not None:
            ids.update(int(i) for i in source)
    return ids


def trim_at_end_token(token_ids: list[int], end_ids: set[int]) -> tuple[list[int], bool]:
    """Cut generated tokens after the first end token.

    Args:
        token_ids: Generated tokens, possibly followed by padding.
        end_ids: Token ids that end a response.

    Returns:
        ``(tokens, finished)``. The end token is kept. ``finished`` is
        ``False`` when no end token was found.
    """
    for index, token in enumerate(token_ids):
        if token in end_ids:
            return token_ids[: index + 1], True
    return token_ids, False


@torch.no_grad()
def sample_responses(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    n_per_prompt: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    batch_size: int = 8,
) -> list[list[Rollout]]:
    """Sample ``n_per_prompt`` responses for each prompt.

    Args:
        model: A causal language model.
        tokenizer: Its tokenizer, padding on the left.
        prompts: Prompt texts that already include the chat template.
        n_per_prompt: Responses per prompt.
        max_new_tokens: Token limit for each response.
        temperature: Sampling temperature. Zero or less means greedy decoding.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling cutoff. Zero turns it off.
        batch_size: Prompts per generation call. If the GPU runs out of
            memory, the batch is halved and the same prompts are tried again.

    Returns:
        One list of rollouts per prompt, in the order given.
    """
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    end_ids = end_token_ids(model, tokenizer)
    sampling: dict[str, Any] = {"do_sample": False}
    if temperature > 0:
        sampling = {
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k > 0 else None,
        }

    results: list[list[Rollout]] = []
    start = 0
    while start < len(prompts):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        try:
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n_per_prompt,
                pad_token_id=tokenizer.pad_token_id,
                **sampling,
            )
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            del encoded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.warning(
                "Out of GPU memory while sampling. Retrying with batch size %d", batch_size
            )
            continue
        new_tokens = output[:, encoded["input_ids"].shape[1] :].tolist()
        del output
        for prompt_index in range(len(batch)):
            group = []
            for row in new_tokens[prompt_index * n_per_prompt : (prompt_index + 1) * n_per_prompt]:
                token_ids, finished = trim_at_end_token(row, end_ids)
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                group.append(Rollout(text=text, token_ids=token_ids, finished=finished))
            results.append(group)
        start += len(batch)
        logger.debug("Sampled %d of %d prompts", len(results), len(prompts))
    if was_training:
        model.train()
    return results
