# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""End-to-end tests on a tiny random model: sampling, losses, training, evaluation."""

import json
import random

import pytest
import torch

from gram_sdpo import train as train_module
from gram_sdpo.data import toy_problems
from gram_sdpo.evaluate import checkpoint_folders, evaluate_checkpoint, load_or_build_probes
from gram_sdpo.modeling import base_model, load_adapter_weights, resolve_dtype
from gram_sdpo.report import collect_rows, to_markdown, write_report
from gram_sdpo.rollout import Rollout, sample_responses, trim_at_end_token
from gram_sdpo.sdpo import choose_solution, example_losses, forward_response, gram_distance
from gram_sdpo.timing import format_summary, project_hours, read_timing_run
from gram_sdpo.train import latest_checkpoint, train


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


def test_sampling_halves_the_batch_when_the_gpu_runs_out_of_memory(
    tiny_model, tiny_tokenizer, monkeypatch
):
    """An out-of-memory error on a batch is retried with half as many prompts."""
    real_generate = tiny_model.generate
    batch_sizes = []

    def generate_with_small_memory(**kwargs):
        """Fail on any batch of more than two prompts, as a full GPU would."""
        batch_sizes.append(kwargs["input_ids"].shape[0])
        if kwargs["input_ids"].shape[0] > 2:
            raise torch.OutOfMemoryError("simulated")
        return real_generate(**kwargs)

    monkeypatch.setattr(tiny_model, "generate", generate_with_small_memory)
    groups = sample_responses(
        tiny_model, tiny_tokenizer, ["ab"] * 5, n_per_prompt=2, max_new_tokens=4, batch_size=4
    )
    assert [len(group) for group in groups] == [2] * 5
    assert batch_sizes == [4, 2, 2, 1]


def test_sampling_gives_up_when_a_single_prompt_does_not_fit(
    tiny_model, tiny_tokenizer, monkeypatch
):
    """With a batch of one there is nothing left to halve, so the error is raised."""

    def always_out_of_memory(**kwargs):
        """Fail every call."""
        raise torch.OutOfMemoryError("simulated")

    monkeypatch.setattr(tiny_model, "generate", always_out_of_memory)
    with pytest.raises(torch.OutOfMemoryError):
        sample_responses(tiny_model, tiny_tokenizer, ["ab"], 1, max_new_tokens=4, batch_size=1)


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


def test_training_resumes_from_the_newest_checkpoint(tiny_model, tiny_tokenizer, tiny_config):
    """A run cut off after two steps continues at step three with its weights restored."""
    problems = toy_problems(8)
    run_dir = train(tiny_config, always_correct, tiny_model, tiny_tokenizer, problems)
    after_two = {n: p.detach().clone() for n, p in tiny_model.named_parameters() if "lora_" in n}
    assert latest_checkpoint(run_dir).name == "step-00002"

    perturb_adapters(tiny_model)
    tiny_config.train.steps = 3
    seen: dict[str, bool] = {}

    def record_restored_weights(response: str, truth: str) -> bool:
        """Note, at the first scoring call, whether the weights were restored."""
        if not seen:
            seen["restored"] = all(
                torch.equal(p, after_two[n])
                for n, p in tiny_model.named_parameters()
                if "lora_" in n
            )
        return True

    train(tiny_config, record_restored_weights, tiny_model, tiny_tokenizer, problems, resume=True)
    assert seen["restored"]
    steps = [json.loads(x)["step"] for x in (run_dir / "train_log.jsonl").read_text().splitlines()]
    assert steps == [1, 2, 3]
    assert latest_checkpoint(run_dir).name == "step-00003"


def test_resume_with_no_checkpoint_starts_from_step_one(tiny_model, tiny_tokenizer, tiny_config):
    """Asking to resume an empty run folder is the same as a fresh start."""
    run_dir = train(
        tiny_config, always_correct, tiny_model, tiny_tokenizer, toy_problems(8), resume=True
    )
    first = json.loads((run_dir / "train_log.jsonl").read_text().splitlines()[0])
    assert first["step"] == 1
    assert first["sampled_tokens_per_second"] > 0


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


def test_train_command_line_accepts_several_overrides(monkeypatch, tmp_path):
    """One --override flag can carry several settings, and --resume is passed on."""
    config_file = tmp_path / "c.yaml"
    config_file.write_text("{}", encoding="utf-8")
    captured = {}

    def fake_train(config, resume=False):
        """Record what the command line asked for instead of training."""
        captured.update(config=config, resume=resume)

    monkeypatch.setattr(train_module, "train", fake_train)
    train_module.main(
        [
            "--config",
            str(config_file),
            "--resume",
            "--override",
            "loss.gram_weight=0.1",
            f"train.output_dir={tmp_path / 'run'}",
            "--override",
            "train.steps=3",
        ]
    )
    assert captured["resume"] is True
    assert captured["config"].loss.gram_weight == 0.1
    assert captured["config"].train.steps == 3
    assert captured["config"].train.output_dir == str(tmp_path / "run")


def test_timing_summary_scales_one_step_to_the_plan(tiny_model, tiny_tokenizer, tiny_config):
    """A measured step is scaled by question visits for training and by samples for evaluation."""
    run_dir = train(tiny_config, always_correct, tiny_model, tiny_tokenizer, toy_problems(8))
    metrics = read_timing_run(run_dir)
    assert metrics["questions"] == 2 and metrics["attempts"] == 4

    metrics.update(seconds=900.0, sampling_seconds=580.0)
    plan = tiny_config
    plan.train.steps, plan.train.questions_per_step = 100, 16
    plan.data.heldout_size, plan.eval.max_problems = 200, 100
    plan.eval.benchmarks, plan.eval.samples_per_problem = ["heldout", "aime24", "amc23"], 16
    hours = project_hours(metrics, plan)
    assert hours["train_hours"] == pytest.approx(100 * 16 * (900 / 2) / 3600)
    assert hours["eval_hours"] == pytest.approx(170 * 16 * (580 / 4) / 3600)
    summary = format_summary("timing_x", metrics, hours)
    assert "projected: one training run" in summary
    assert f"~{4 * hours['train_hours'] + 9 * hours['eval_hours']:.0f} hours" in summary


def test_timing_summary_handles_a_run_with_no_log(tmp_path):
    """A run folder that has not logged a step yet is reported as missing, not as an error."""
    assert read_timing_run(tmp_path) is None
