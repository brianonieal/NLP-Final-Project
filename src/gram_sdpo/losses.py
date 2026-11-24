# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""The three loss terms: SDPO divergence, Gram anchoring, and the KL baseline.

All functions work on plain tensors, so they are tested without loading a
language model.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _top_k_log_probs(
    student_logits: torch.Tensor, other_logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce two distributions to the student's top-K tokens plus a tail bin.

    This is the top-K distillation of Hübotter et al. (2026): keep the K
    tokens the student rates highest and fold everything else into one bin.

    Args:
        student_logits: Tensor of shape ``(tokens, vocab)``.
        other_logits: Tensor of the same shape, from the teacher or base model.
        top_k: Number of tokens to keep.

    Returns:
        Two tensors of shape ``(tokens, top_k + 1)`` holding log-probabilities.
    """
    # Work from the raw logits: subtracting the log-sum-exp of each row gives
    # the same log-probabilities as log_softmax without building a second
    # full (tokens x vocab) tensor. With a 150k-token vocabulary that copy is
    # over 1 GB per call.
    student_top_logits, index = student_logits.topk(top_k, dim=-1)
    student_top = student_top_logits - torch.logsumexp(student_logits, dim=-1, keepdim=True)
    other_top = other_logits.gather(-1, index) - torch.logsumexp(other_logits, dim=-1, keepdim=True)

    def with_tail(top: torch.Tensor) -> torch.Tensor:
        """Append the log-probability of all tokens outside the top K."""
        mass = top.exp().sum(dim=-1, keepdim=True).clamp(max=1.0 - 1e-6)
        return torch.cat([top, torch.log1p(-mass)], dim=-1)

    return with_tail(student_top), with_tail(other_top)


def token_divergence(
    student_logits: torch.Tensor,
    target_logits: torch.Tensor,
    kind: str = "kl",
    top_k: int | None = None,
) -> torch.Tensor:
    """Average divergence between student and target next-token distributions.

    With ``kind="kl"`` this is the proposal's SDPO loss, KL(student || teacher)
    at every response token. With ``kind="jsd"`` it is the Jensen-Shannon
    divergence that Kim et al. (2026) used. The target is treated as a
    constant, so no gradient flows into it.

    Args:
        student_logits: Tensor of shape ``(tokens, vocab)`` with gradients.
        target_logits: Tensor of the same shape from the teacher or base model.
        kind: ``"kl"`` or ``"jsd"``.
        top_k: When set and smaller than the vocabulary, compare only the
            student's top-K tokens plus a tail bin.

    Returns:
        A scalar tensor: the divergence averaged over tokens.
    """
    student_logits = student_logits.float()
    target_logits = target_logits.detach().float()
    if top_k is not None and top_k < student_logits.shape[-1]:
        student, target = _top_k_log_probs(student_logits, target_logits, top_k)
    else:
        student = F.log_softmax(student_logits, dim=-1)
        target = F.log_softmax(target_logits, dim=-1)

    if kind == "kl":
        per_token = (student.exp() * (student - target)).sum(dim=-1)
    elif kind == "jsd":
        mixture = torch.logsumexp(torch.stack([student, target]), dim=0) - math.log(2.0)
        per_token = 0.5 * (student.exp() * (student - mixture)).sum(dim=-1) + 0.5 * (
            target.exp() * (target - mixture)
        ).sum(dim=-1)
    else:
        raise ValueError(f"kind must be 'kl' or 'jsd', got {kind!r}")
    return per_token.mean()


def gram_matrix(hidden: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarities between token representations.

    Args:
        hidden: Tensor of shape ``(tokens, width)``.

    Returns:
        Tensor of shape ``(tokens, tokens)``: the Gram matrix of the
        L2-normalized rows.
    """
    normalized = F.normalize(hidden.float(), dim=-1)
    return normalized @ normalized.T


def gram_loss(student_hidden: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
    """Gram anchoring loss, adapted from DINOv3 to token representations.

    The loss is the squared Frobenius norm of the difference between the two
    Gram matrices, divided by the number of entries. Dividing keeps its scale
    the same whatever number of tokens is sampled. Features may move freely as
    long as the similarities between tokens stay the same, so rotating every
    student representation by the same orthogonal matrix costs nothing.

    Args:
        student_hidden: Tensor of shape ``(tokens, width)`` with gradients.
        target_hidden: Tensor of the same shape from the anchor model.

    Returns:
        A scalar tensor.
    """
    difference = gram_matrix(student_hidden) - gram_matrix(target_hidden.detach())
    return difference.pow(2).mean()


def sample_token_indices(
    n_tokens: int, n_sample: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Choose which response tokens enter the Gram matrix.

    The Gram matrix grows with the square of the response length, so the loss
    is computed on a random sample of tokens.

    Args:
        n_tokens: Number of response tokens available.
        n_sample: Number to keep. All tokens are kept when there are fewer.
        generator: Optional random generator, for reproducible samples.

    Returns:
        A sorted 1-D tensor of indices.
    """
    if n_tokens <= n_sample:
        return torch.arange(n_tokens)
    return torch.randperm(n_tokens, generator=generator)[:n_sample].sort().values
