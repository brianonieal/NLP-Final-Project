# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""Summarize timing-test runs and project how long a full plan would take.

Run it with ``python -m gram_sdpo.timing --runs runs --plan configs/sdpo.yaml``
after one or more one-step timing runs. It reads the last line of each
``timing_*/train_log.jsonl`` and scales it to the plan's settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from gram_sdpo.config import Config, load_config
from gram_sdpo.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)

OOD_PROBLEMS = {"aime24": 30, "amc23": 40}
# Experiments 1 and 2 as scripted: SDPO, SDPO + KL, and Gram anchoring at two
# weights, with six evaluations of the SDPO run and one of each other run.
PLAN_TRAINING_RUNS = 4
PLAN_EVALUATIONS = 9


def read_timing_run(run_dir: Path) -> dict[str, Any] | None:
    """Read the last logged step of a timing run.

    Args:
        run_dir: A run folder holding ``train_log.jsonl`` and ``config.yaml``.

    Returns:
        The step's metrics plus ``questions`` and ``attempts`` (how many
        questions and sampled attempts the step used), or ``None`` when the
        run has no log yet.
    """
    log = run_dir / "train_log.jsonl"
    if not log.exists() or not log.read_text(encoding="utf-8").strip():
        return None
    metrics = json.loads(log.read_text(encoding="utf-8").strip().splitlines()[-1])
    saved = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    metrics["questions"] = saved["train"]["questions_per_step"]
    metrics["attempts"] = metrics["questions"] * saved["rollout"]["rollouts_per_question"]
    return metrics


def project_hours(metrics: dict[str, Any], plan: Config) -> dict[str, float]:
    """Scale one measured step to the plan's training and evaluation sizes.

    Training time scales with the number of question visits. Evaluation time
    scales with the number of sampled responses, at the measured seconds per
    sampled attempt.

    Args:
        metrics: Output of ``read_timing_run``.
        plan: The configuration of the run being planned.

    Returns:
        ``train_hours`` for one training run and ``eval_hours`` for one
        evaluation of one checkpoint.
    """
    seconds_per_question = metrics["seconds"] / metrics["questions"]
    seconds_per_attempt = metrics["sampling_seconds"] / metrics["attempts"]
    question_visits = plan.train.steps * plan.train.questions_per_step
    heldout = min(plan.eval.max_problems or plan.data.heldout_size, plan.data.heldout_size)
    problems = sum(
        heldout if name == "heldout" else OOD_PROBLEMS.get(name, 0) for name in plan.eval.benchmarks
    )
    samples = problems * plan.eval.samples_per_problem
    return {
        "train_hours": question_visits * seconds_per_question / 3600,
        "eval_hours": samples * seconds_per_attempt / 3600,
    }


def plan_hours(hours: dict[str, float]) -> float:
    """Total GPU hours for Experiments 1 and 2 as scripted.

    Args:
        hours: Output of ``project_hours``.

    Returns:
        Hours for four training runs plus nine evaluations.
    """
    return PLAN_TRAINING_RUNS * hours["train_hours"] + PLAN_EVALUATIONS * hours["eval_hours"]


def format_summary(name: str, metrics: dict[str, Any], hours: dict[str, float]) -> str:
    """Render one timing run as a short text block.

    Args:
        name: The run folder's name.
        metrics: Output of ``read_timing_run``.
        hours: Output of ``project_hours``.

    Returns:
        A multi-line summary ready to print.
    """
    attempts = metrics["attempts"]
    lines = [
        f"== {name}",
        f"   accuracy on {attempts} attempts:        {metrics['train_accuracy']:.3f}",
        f"   attempts cut off by the limit:  {metrics['unfinished_fraction']:.0%}",
        f"   mean response tokens:           {metrics['mean_response_tokens']:.0f}",
        f"   attempts trained on:            {metrics['attempts_trained']:.0f} of {attempts}",
        f"   sampling speed (tokens/sec):    {metrics['sampled_tokens_per_second']:.0f}",
        f"   seconds for the step:           {metrics['seconds']:.0f}",
        f"   peak GPU memory (GB):           {metrics['peak_gpu_gb']:.1f}",
        f"   projected: one training run     ~{hours['train_hours']:.1f} hours",
        f"   projected: one evaluation       ~{hours['eval_hours']:.1f} hours",
        f"   projected: whole plan           ~{plan_hours(hours):.0f} hours"
        f" ({PLAN_TRAINING_RUNS} training runs, {PLAN_EVALUATIONS} evaluations)",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for the timing summary."""
    parser = argparse.ArgumentParser(description="Summarize timing runs and project a plan.")
    parser.add_argument("--runs", default="runs", help="Folder that holds the timing_* folders.")
    parser.add_argument("--plan", default="configs/sdpo.yaml", help="Config of the planned run.")
    args = parser.parse_args(argv)
    configure_logging()
    plan = load_config(args.plan)
    found = False
    for run_dir in sorted(Path(args.runs).glob("timing_*")):
        metrics = read_timing_run(run_dir)
        if metrics is None:
            print(f"== {run_dir.name}: no log yet")
            continue
        found = True
        print(format_summary(run_dir.name, metrics, project_hours(metrics, plan)))
    if not found:
        print(f"No finished timing runs under {args.runs}.")
    print(f"Projections use the settings in {args.plan}.")


if __name__ == "__main__":
    main()
