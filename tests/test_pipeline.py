# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""End-to-end tests on a tiny random model: sampling, losses, training, evaluation."""

import json
import random

import pytest
import torch

from gram_sdpo.data import toy_problems
from gram_sdpo.evaluate import checkpoint_folders, evaluate_checkpoint, load_or_build_probes
from gram_sdpo.modeling import base_model, load_adapter_weights, resolve_dtype
from gram_sdpo.report import collect_rows, to_markdown, write_report
from gram_sdpo.rollout import Rollout, sample_responses, trim_at_end_token
from gram_sdpo.sdpo import choose_solution, example_losses, forward_response, gram_distance
from gram_sdpo.train import train


def always_correct(response: str, truth: str) -> bool:
    """Stand-in scorer so the tiny random model produces usable rollouts."""
    return True


def never_correct(response: str, truth: str) -> bool:
    """Stand-in scorer for the case where no rollout is correct."""
    return False


def perturb_adapters(model) -> None:
    """Give the LoRA B matrices nonzero values so the student differs from the base."""
    torch.manual_seed(1)
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.5)


def test_trim_at_end_token():
    """Tokens after the first end token are dropped and the end token is kept."""
    assert trim_at_end_token([5, 6, 1, 0, 0], {1}) == ([5, 6, 1], True)
    assert trim_at_end_token([5, 6, 7], {1}) == ([5, 6, 7], False)


def test_resolve_dtype():
    """'auto' means float32 on the CPU, and unknown names fail."""
    assert resolve_dtype("auto", torch.device("cpu")) == torch.float32
    assert resolve_dtype("bfloat16", torch.device("cpu")) == torch.bfloat16
    with pytest.raises(ValueError, match="Unknown dtype"):
        resolve_dtype("float8", torch.device("cpu"))


def test_sample_responses_shapes_and_limits(tiny_model, tiny_tokenizer):
    """Each prompt gets the requested number of rollouts within the token limit."""
    groups = sample_responses(
        tiny_model,
        tiny_tokenizer,
        ["ab", "abcdef", "a"],
        n_per_prompt=3,
        max_new_tokens=5,
        batch_size=2,
    )
    assert [len(group) for group in groups] == [3, 3, 3]
    for group in groups:
        for rollout in group:
            assert 1 <= len(rollout.token_ids) <= 5
            assert isinstance(rollout.text, str)


def test_choose_solution_follows_the_self_teacher_template():
    """A correct attempt is its own solution. A failed one borrows from its group."""
    group = [Rollout("wrong", [3], True), Rollout("right", [4], True)]
    rng = random.Random(0)
    assert choose_solution(group, [False, True], 1, rng) == "right"
    assert choose_solution(group, [False, True], 0, rng) == "right"
    assert choose_solution(group, [False, False], 0, rng) is None


def test_forward_response_aligns_logits_with_response_tokens(tiny_model):
    """Row t of the sliced logits is the prediction for response token t."""
    prompt, response = [5, 6, 7], [8, 9, 10, 11]
    logits, hidden = forward_response(tiny_model, prompt, response, [-1])
    full = tiny_model(input_ids=torch.tensor([prompt + response])).logits[0]
    assert logits.shape[0] == len(response)
    assert torch.allclose(logits[0], full[len(prompt) - 1])
    assert torch.allclose(logits[-1], full[len(prompt) + len(response) - 2])
    assert hidden[0].shape == (len(response), 32)


def test_regularizers_are_zero_before_training(tiny_model, tiny_tokenizer, tiny_config):
    """Fresh adapters are a no-op, so the student equals the base model."""
    tiny_config.loss.gram_weight = 1.0
    tiny_config.loss.kl_weight = 1.0
    losses = example_losses(
        tiny_model, tiny_tokenizer, tiny_config, "What is 1 + 2?", [10, 11, 12, 13], "abc"
    )
    assert losses["gram"].item() == pytest.approx(0.0, abs=1e-6)
    assert losses["kl"].item() == pytest.approx(0.0, abs=1e-6)
    assert losses["sdpo"].item() > 0


@pytest.mark.parametrize("gram_target", ["base", "teacher"])
def test_losses_are_finite_and_train_only_the_adapters(
    tiny_model, tiny_tokenizer, tiny_config, gram_target
):
    """After the student moves, every term is positive and only LoRA weights get gradients."""
    perturb_adapters(tiny_model)
    tiny_config.loss.gram_weight = 1.0
    tiny_config.loss.kl_weight = 1.0
    tiny_config.loss.gram_target = gram_target
    tiny_config.loss.gram_layers = [-1, 1]
    losses = example_losses(
        tiny_model, tiny_tokenizer, tiny_config, "What is 1 + 2?", [10, 11, 12, 13, 14], "abc"
    )
    assert all(torch.isfinite(value) for value in losses.values())
    assert losses["gram"].item() > 0 and losses["kl"].item() > 0
    expected = losses["sdpo"] + losses["gram"] + losses["kl"]
    assert losses["total"].item() == pytest.approx(expected.item(), rel=1e-5)
    losses["total"].backward()
    with_gradient = {n for n, p in tiny_model.named_parameters() if p.grad is not None}
    assert with_gradient and all("lora_" in name for name in with_gradient)


def test_example_losses_skips_unusable_examples(tiny_model, tiny_tokenizer, tiny_config):
    """A one-token response or an input past the context limit returns None."""
    assert example_losses(tiny_model, tiny_tokenizer, tiny_config, "q", [10], "abc") is None
    too_long = "a" * 600
    assert example_losses(tiny_model, tiny_tokenizer, tiny_config, "q", [10, 11], too_long) is None


def test_gram_distance_is_zero_for_the_base_model_and_positive_after_drift(
    tiny_model, tiny_tokenizer, tiny_config
):
    """Drift from the base model shows up in the Gram distance."""
    args = (tiny_model, tiny_tokenizer, tiny_config, "What is 1 + 2?", [10, 11, 12, 13, 14])
    assert gram_distance(*args) == pytest.approx(0.0, abs=1e-6)
    perturb_adapters(tiny_model)
    assert gram_distance(*args) > 0


def test_base_model_context_turns_adapters_off_and_back_on(tiny_model):
    """Inside the block the outputs are the base model's. Outside they are the student's."""
    ids = torch.tensor([[5, 6, 7, 8]])
    before = tiny_model(input_ids=ids).logits
    perturb_adapters(tiny_model)
    after = tiny_model(input_ids=ids).logits
    with base_model(tiny_model):
        inside = tiny_model(input_ids=ids).logits
    assert not torch.allclose(before, after)
    assert torch.allclose(before, inside, atol=1e-6)
    assert torch.allclose(after, tiny_model(input_ids=ids).logits)


def test_training_updates_adapters_and_writes_run_files(tiny_model, tiny_tokenizer, tiny_config):
    """Two SDPO steps change the adapter weights and leave a log and checkpoints."""
    tiny_config.loss.gram_weight = 1.0
    start = {n: p.detach().clone() for n, p in tiny_model.named_parameters() if "lora_" in n}
    run_dir = train(tiny_config, always_correct, tiny_model, tiny_tokenizer, toy_problems(8))
    changed = [
        n for n, p in tiny_model.named_parameters() if "lora_" in n and not torch.equal(p, start[n])
    ]
    assert changed
    records = [json.loads(line) for line in (run_dir / "train_log.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2]
    assert records[0]["attempts_trained"] == 4
    assert "loss_gram" in records[0]
    assert (run_dir / "config.yaml").exists()
    assert [p.name for p in checkpoint_folders(run_dir, ["all"])] == ["step-00001", "step-00002"]
    assert checkpoint_folders(run_dir, ["base", "final"])[1].name == "step-00002"


def test_training_skips_questions_with_no_correct_rollout(tiny_model, tiny_tokenizer, tiny_config):
    """With no correct rollout there is no solution, so nothing is trained."""
    start = {n: p.detach().clone() for n, p in tiny_model.named_parameters() if "lora_" in n}
    run_dir = train(tiny_config, never_correct, tiny_model, tiny_tokenizer, toy_problems(8))
    assert all(torch.equal(p, start[n]) for n, p in tiny_model.named_parameters() if "lora_" in n)
    record = json.loads((run_dir / "train_log.jsonl").read_text().splitlines()[0])
    assert record["attempts_trained"] == 0
    assert record["questions_without_solution"] == 2


def test_evaluation_report_and_checkpoint_reload(tiny_model, tiny_tokenizer, tiny_config, tmp_path):
    """A trained checkpoint can be reloaded, evaluated, and reported."""
    run_dir = train(tiny_config, always_correct, tiny_model, tiny_tokenizer, toy_problems(8))
    heldout = toy_problems(12)[8:]
    probes = load_or_build_probes(tiny_model, tiny_tokenizer, tiny_config, heldout)
    assert load_or_build_probes(tiny_model, tiny_tokenizer, tiny_config, heldout) == probes

    trained = {n: p.detach().clone() for n, p in tiny_model.named_parameters() if "lora_" in n}
    perturb_adapters(tiny_model)
    load_adapter_weights(tiny_model, checkpoint_folders(run_dir, ["final"])[0])
    assert all(torch.equal(p, trained[n]) for n, p in tiny_model.named_parameters() if "lora_" in n)

    results = evaluate_checkpoint(
        tiny_model, tiny_tokenizer, tiny_config, {"heldout": heldout}, probes, always_correct
    )
    metrics = results["benchmarks"]["heldout"]
    assert metrics["acc_at_k"] == 1.0 and metrics["pass_at_k"] == 1.0
    assert metrics["problems"] == 4 and metrics["k"] == 2
    assert results["gram_distance"] >= 0

    results.update(
        run="run", checkpoint="step-00002", step=2, seed=0, gram_weight=0.0, kl_weight=0.0
    )
    eval_dir = run_dir / "eval"
    eval_dir.mkdir()
    (eval_dir / "step-00002.json").write_text(json.dumps(results), encoding="utf-8")
    rows = collect_rows(run_dir.parent)
    assert rows[0]["heldout_acc_at_k"] == 1.0
    assert "heldout_acc_at_k" in to_markdown(rows)
    csv_path, md_path = write_report(run_dir.parent)
    assert csv_path.exists() and md_path.exists()


def test_report_handles_a_folder_with_no_results(tmp_path):
    """An empty runs folder gives a note instead of an error."""
    assert collect_rows(tmp_path) == []
    assert "No evaluation files" in to_markdown([])
