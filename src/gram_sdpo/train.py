# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""The SDPO training loop.

Each step samples questions, lets the student attempt each one several times,
checks the answers, and trains on every attempt whose group has a correct
rollout. Run it with ``python -m gram_sdpo.train --config configs/<file>.yaml``.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import yaml

from gram_sdpo.config import Config, load_config
from gram_sdpo.data import (
    Problem,
    load_problems,
    render_prompt,
    split_train_heldout,
    student_messages,
)
from gram_sdpo.logging_utils import configure_logging, get_logger
from gram_sdpo.modeling import load_adapter_weights, load_student
from gram_sdpo.rollout import sample_responses
from gram_sdpo.scoring import epistemic_token_count, is_correct
from gram_sdpo.sdpo import choose_solution, example_losses

logger = get_logger(__name__)

ScoreFn = Callable[[str, str], bool]


def _mean(values: list[float]) -> float:
    """Average a list, returning 0.0 for an empty one."""
    return sum(values) / len(values) if values else 0.0


def train_step(
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    config: Config,
    questions: list[Problem],
    score_fn: ScoreFn,
    rng: random.Random,
    generator: torch.Generator,
) -> dict[str, float]:
    """Run one SDPO step: sample, score, and update on the usable attempts.

    Args:
        model: The student.
        tokenizer: Its tokenizer.
        optimizer: Optimizer over the adapter weights.
        config: The run configuration.
        questions: The questions for this step.
        score_fn: ``(response, reference answer) -> correct``.
        rng: Random source for solution choice.
        generator: Random generator for the Gram token sample.

    Returns:
        Metrics for the step: accuracy, response length, uncertainty-word
        count, how many attempts were trained on, sampling speed, peak GPU
        memory, and the mean of each loss.
    """
    prompts = [
        render_prompt(tokenizer, student_messages(q.question), config.model.enable_thinking)
        for q in questions
    ]
    sampling_started = time.time()
    groups = sample_responses(
        model,
        tokenizer,
        prompts,
        n_per_prompt=config.rollout.rollouts_per_question,
        max_new_tokens=config.rollout.max_new_tokens,
        temperature=config.rollout.temperature,
        top_p=config.rollout.top_p,
        top_k=config.rollout.top_k,
        batch_size=config.rollout.batch_size,
    )
    sampling_seconds = time.time() - sampling_started

    examples: list[tuple[Problem, list[int], str]] = []
    scores: list[float] = []
    lengths: list[float] = []
    markers: list[float] = []
    unfinished = 0
    questions_without_solution = 0
    for problem, group in zip(questions, groups, strict=True):
        correct = [score_fn(rollout.text, problem.answer) for rollout in group]
        scores.extend(float(ok) for ok in correct)
        lengths.extend(float(len(rollout.token_ids)) for rollout in group)
        markers.extend(float(epistemic_token_count(rollout.text)) for rollout in group)
        unfinished += sum(not rollout.finished for rollout in group)
        if not any(correct):
            questions_without_solution += 1
            continue
        for index, rollout in enumerate(group):
            solution = choose_solution(group, correct, index, rng)
            examples.append((problem, rollout.token_ids, solution))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {}
    used = 0
    for problem, response_ids, solution in examples:
        losses = example_losses(
            model, tokenizer, config, problem.question, response_ids, solution, generator
        )
        if losses is None:
            continue
        (losses["total"] / len(examples)).backward()
        used += 1
        for name, value in losses.items():
            sums[name] = sums.get(name, 0.0) + float(value.detach())
    if used:
        trainable = [p for p in model.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, config.train.grad_clip)
        optimizer.step()

    metrics = {
        "train_accuracy": _mean(scores),
        "mean_response_tokens": _mean(lengths),
        "mean_epistemic_tokens": _mean(markers),
        "unfinished_fraction": unfinished / max(1, len(scores)),
        "questions_without_solution": float(questions_without_solution),
        "attempts_trained": float(used),
        "sampling_seconds": round(sampling_seconds, 2),
        "sampled_tokens_per_second": sum(lengths) / max(sampling_seconds, 1e-9),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9
        if torch.cuda.is_available()
        else 0.0,
    }
    metrics.update({f"loss_{name}": total / used for name, total in sums.items()} if used else {})
    return metrics


TRAINER_STATE_FILE = "trainer_state.pt"


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Return the newest checkpoint folder that can be resumed from.

    Args:
        run_dir: The run folder.

    Returns:
        The highest-numbered ``step-*`` folder that holds a trainer state
        file, or ``None`` when there is none.
    """
    candidates = [
        folder
        for folder in sorted((run_dir / "checkpoints").glob("step-*"))
        if (folder / TRAINER_STATE_FILE).exists()
    ]
    return candidates[-1] if candidates else None


def train(
    config: Config,
    score_fn: ScoreFn = is_correct,
    model: Any | None = None,
    tokenizer: Any | None = None,
    problems: list[Problem] | None = None,
    resume: bool = False,
) -> Path:
    """Train the student with SDPO and save adapter checkpoints.

    Args:
        config: The run configuration.
        score_fn: Answer checker. Tests pass a stand-in.
        model: Optional preloaded student, for tests.
        tokenizer: Optional preloaded tokenizer, for tests.
        problems: Optional preloaded training problems, for tests.
        resume: Continue from the newest checkpoint in the run folder when
            there is one. The adapter, optimizer, and schedule are restored.
            Random draws restart from a seed derived from the step, so a
            resumed run is repeatable but not identical to an unbroken one.

    Returns:
        The run folder. It holds ``config.yaml``, ``train_log.jsonl``, and a
        ``checkpoints`` folder with one adapter per saved step.
    """
    run_dir = Path(config.train.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")

    torch.manual_seed(config.train.seed)
    rng = random.Random(config.train.seed)
    generator = torch.Generator().manual_seed(config.train.seed)

    if model is None or tokenizer is None:
        model, tokenizer, _ = load_student(config.model)
    if problems is None:
        all_problems = load_problems(config.data.train_dataset)
        problems, _ = split_train_heldout(
            all_problems,
            config.data.heldout_size,
            config.data.seed,
            config.data.max_train_questions,
        )
    logger.info("Training on %d questions for %d steps", len(problems), config.train.steps)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )
    warmup = max(1, config.train.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / warmup))

    first_step = 1
    checkpoint = latest_checkpoint(run_dir) if resume else None
    if checkpoint is not None:
        load_adapter_weights(model, checkpoint)
        state = torch.load(checkpoint / TRAINER_STATE_FILE, weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        first_step = int(state["step"]) + 1
        rng = random.Random(config.train.seed + first_step)
        generator = torch.Generator().manual_seed(config.train.seed + first_step)
        torch.manual_seed(config.train.seed + first_step)
        logger.info("Resuming from %s at step %d", checkpoint, first_step)

    log_path = run_dir / "train_log.jsonl"
    with log_path.open("a", encoding="utf-8") as log_file:
        for step in range(first_step, config.train.steps + 1):
            started = time.time()
            batch = rng.sample(problems, min(config.train.questions_per_step, len(problems)))
            metrics = train_step(
                model, tokenizer, optimizer, config, batch, score_fn, rng, generator
            )
            if metrics["attempts_trained"]:
                scheduler.step()
            metrics.update(
                step=step, lr=scheduler.get_last_lr()[0], seconds=round(time.time() - started, 2)
            )
            log_file.write(json.dumps(metrics) + "\n")
            log_file.flush()
            logger.info(
                "step %d | acc %.3f | tokens %.0f | trained %d | loss %.4f",
                step,
                metrics["train_accuracy"],
                metrics["mean_response_tokens"],
                int(metrics["attempts_trained"]),
                metrics.get("loss_total", float("nan")),
            )
            if step % config.train.save_every == 0 or step == config.train.steps:
                checkpoint = run_dir / "checkpoints" / f"step-{step:05d}"
                model.save_pretrained(str(checkpoint))
                torch.save(
                    {
                        "step": step,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    },
                    checkpoint / TRAINER_STATE_FILE,
                )
                logger.info("Saved adapter and trainer state to %s", checkpoint)
    return run_dir


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for training."""
    parser = argparse.ArgumentParser(description="Train a student with SDPO.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--override",
        nargs="+",
        action="extend",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Change settings. Takes one or more values and may be repeated.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the newest checkpoint in the run folder, if any.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config, args.override)
    configure_logging(log_file=Path(config.train.output_dir) / "train.log")
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
