# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""Tests for the SDPO divergence, the Gram loss, and token sampling."""

import pytest
import torch

from gram_sdpo.losses import gram_loss, gram_matrix, sample_token_indices, token_divergence


@pytest.fixture
def logits():
    """Two random sets of next-token logits: 5 tokens, vocabulary of 50."""
    generator = torch.Generator().manual_seed(0)
    return torch.randn(5, 50, generator=generator), torch.randn(5, 50, generator=generator)


@pytest.mark.parametrize("kind", ["kl", "jsd"])
@pytest.mark.parametrize("top_k", [None, 10])
def test_divergence_is_zero_for_identical_distributions(logits, kind, top_k):
    """A student that already matches its target has nothing to learn."""
    student, _ = logits
    assert token_divergence(student, student.clone(), kind, top_k).item() == pytest.approx(
        0.0, abs=1e-6
    )


@pytest.mark.parametrize("kind", ["kl", "jsd"])
@pytest.mark.parametrize("top_k", [None, 10])
def test_divergence_is_positive_for_different_distributions(logits, kind, top_k):
    """Different distributions give a positive loss."""
    student, target = logits
    assert token_divergence(student, target, kind, top_k).item() > 0


def test_kl_matches_a_hand_computed_value():
    """KL(student || target) for two-token distributions [0.5, 0.5] and [0.9, 0.1]."""
    student = torch.log(torch.tensor([[0.5, 0.5]]))
    target = torch.log(torch.tensor([[0.9, 0.1]]))
    expected = 0.5 * torch.log(torch.tensor(0.5 / 0.9)) + 0.5 * torch.log(torch.tensor(0.5 / 0.1))
    assert token_divergence(student, target, "kl").item() == pytest.approx(
        expected.item(), rel=1e-5
    )


def test_jsd_is_symmetric_and_bounded(logits):
    """Jensen-Shannon divergence is symmetric and at most ln 2."""
    student, target = logits
    forward = token_divergence(student, target, "jsd").item()
    backward = token_divergence(target, student, "jsd").item()
    assert forward == pytest.approx(backward, rel=1e-5)
    assert forward <= torch.log(torch.tensor(2.0)).item()


def test_top_k_never_exceeds_the_full_kl(logits):
    """Merging tail tokens into one bin can only lose information."""
    student, target = logits
    full = token_divergence(student, target, "kl").item()
    reduced = token_divergence(student, target, "kl", top_k=10).item()
    assert reduced <= full + 1e-6


def test_no_gradient_reaches_the_target(logits):
    """The self-teacher is a constant: only the student gets gradients."""
    student, target = logits
    student.requires_grad_(True)
    target.requires_grad_(True)
    token_divergence(student, target, "kl").backward()
    assert student.grad is not None
    assert target.grad is None


def test_unknown_divergence_is_rejected(logits):
    """A misspelt divergence name raises instead of training on the wrong loss."""
    with pytest.raises(ValueError, match="kind"):
        token_divergence(*logits, kind="mse")


def test_gram_matrix_has_unit_diagonal():
    """Every token has cosine similarity 1 with itself."""
    hidden = torch.randn(6, 16)
    assert torch.allclose(torch.diagonal(gram_matrix(hidden)), torch.ones(6), atol=1e-5)


def test_gram_loss_ignores_a_shared_rotation_and_scaling():
    """Rotating or rescaling all features together keeps every similarity."""
    torch.manual_seed(0)
    hidden = torch.randn(8, 16)
    rotation, _ = torch.linalg.qr(torch.randn(16, 16))
    assert gram_loss(hidden @ rotation, hidden).item() == pytest.approx(0.0, abs=1e-6)
    assert gram_loss(3.0 * hidden, hidden).item() == pytest.approx(0.0, abs=1e-6)


def test_gram_loss_is_positive_when_token_relations_change():
    """Moving tokens independently changes the similarities."""
    torch.manual_seed(0)
    hidden = torch.randn(8, 16)
    assert gram_loss(hidden + torch.randn(8, 16), hidden).item() > 0


def test_gram_loss_sends_gradients_to_the_student_only():
    """The anchor is a constant."""
    student = torch.randn(4, 8, requires_grad=True)
    anchor = torch.randn(4, 8, requires_grad=True)
    gram_loss(student, anchor).backward()
    assert student.grad is not None
    assert anchor.grad is None


def test_sample_token_indices_keeps_everything_when_short():
    """Fewer tokens than the sample size means no sampling."""
    assert sample_token_indices(5, 10).tolist() == [0, 1, 2, 3, 4]


def test_sample_token_indices_is_sorted_unique_and_reproducible():
    """The sample is a sorted subset and repeats under the same seed."""
    first = sample_token_indices(100, 10, torch.Generator().manual_seed(1))
    second = sample_token_indices(100, 10, torch.Generator().manual_seed(1))
    assert first.tolist() == second.tolist()
    assert first.tolist() == sorted(set(first.tolist()))
    assert len(first) == 10
