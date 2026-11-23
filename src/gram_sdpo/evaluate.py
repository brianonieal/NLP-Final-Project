# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""Evaluation: Acc@k, response length, uncertainty words, and Gram distance.

Run it with ``python -m gram_sdpo.evaluate --config configs/<file>.yaml``.
Each checkpoint gets one JSON file under ``<run folder>/eval``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from gram_sdpo.config import Config, load_config
from gram_sdpo.data import (
    Problem,
    load_problems,
    render_prompt,
    split_train_heldout,
    student_messages,
)
from gram_sdpo.logging_utils import configure_logging, get_logger
from gram_sdpo.modeling import base_model, load_adapter_weights, load_student
from gram_sdpo.rollout import sample_responses
from gram_sdpo.scoring import epistemic_token_count, is_correct
from gram_sdpo.sdpo import gram_distance
from gram_sdpo.train import ScoreFn, _mean

logger = get_logger(__name__)


def benchmark_problems(config: Config) -> tuple[dict[str, list[Problem]], list[Problem]]:
    """Load the problems for every benchmark in the config.

    Args:
        config: The run configuration.

    Returns:
        ``(benchmarks, heldout)``. ``benchmarks`` maps each name to its
        problems, cut to ``eval.max_problems`` when that is set. ``heldout``
        is the full in-domain test split, which also supplies the probes.
    """
    all_train = load_problems(config.data.train_dataset)
    _, heldout = split_train_heldout(all_train, config.data.heldout_size, config.data.seed)
    benchmarks: dict[str, list[Problem]] = {}
    for name in config.eval.benchmarks:
        problems = heldout if name == "heldout" else load_problems(name)
        benchmarks[name] = problems[: config.eval.max_problems]
    return benchmarks, heldout


def evaluate_benchmark(
    model: Any, tokenizer: Any, config: Config, problems: list[Problem], score_fn: ScoreFn
) -> dict[str, float]:
    """Score the model on one benchmark.

    Args:
        model: The model to evaluate.
        tokenizer: Its tokenizer.
        config: The run configuration.
        problems: The benchmark's problems.
        score_fn: ``(response, reference answer) -> correct``.

    Returns:
        ``acc_at_k`` (accuracy averaged over the k samples of each problem,
        then over problems), ``pass_at_k`` (share of problems with at least
        one correct sample), mean response length in tokens, mean
        uncertainty-word count, and the share of responses cut off by the
        token limit.
    """
    k = config.eval.samples_per_problem
    prompts = [
        render_prompt(tokenizer, student_messages(p.question), config.model.enable_thinking)
        for p in problems
    ]
    groups = sample_responses(
        model,
        tokenizer,
        prompts,
        n_per_prompt=k,
        max_new_tokens=config.eval.max_new_tokens,
        temperature=config.eval.temperature,
        top_p=config.eval.top_p,
        top_k=config.eval.top_k,
        batch_size=config.eval.batch_size,
    )
    per_problem_accuracy, solved, lengths, markers, unfinished = [], [], [], [], []
    for problem, group in zip(problems, groups, strict=True):
        correct = [score_fn(rollout.text, problem.answer) for rollout in group]
        per_problem_accuracy.append(_mean([float(ok) for ok in correct]))
        solved.append(float(any(correct)))
        lengths.extend(float(len(rollout.token_ids)) for rollout in group)
        markers.extend(float(epistemic_token_count(rollout.text)) for rollout in group)
        unfinished.extend(float(not rollout.finished) for rollout in group)
    return {
        "k": float(k),
        "problems": float(len(problems)),
        "acc_at_k": _mean(per_problem_accuracy),
        "pass_at_k": _mean(solved),
        "mean_response_tokens": _mean(lengths),
        "mean_epistemic_tokens": _mean(markers),
        "unfinished_fraction": _mean(unfinished),
    }


def load_or_build_probes(
    model: Any, tokenizer: Any, config: Config, heldout: list[Problem]
) -> list[dict[str, Any]]:
    """Return the fixed responses used to measure Gram distance.

    The probes are base-model responses to held-out questions. They are
    sampled once and saved, so every checkpoint of every run is measured on
    the same text.

    Args:
        model: A PEFT model. Probes are sampled with its adapters off.
        tokenizer: Its tokenizer.
        config: The run configuration.
        heldout: The in-domain test problems.

    Returns:
        A list of ``{"question": str, "response_ids": list[int]}``.
    """
    path = Path(config.eval.probe_file)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("model") == config.model.name:
            logger.info("Using %d saved probes from %s", len(saved["probes"]), path)
            return saved["probes"]
        logger.warning("Probe file %s is for another model. Rebuilding it.", path)
    questions = heldout[: config.eval.probe_size]
    prompts = [
        render_prompt(tokenizer, student_messages(p.question), config.model.enable_thinking)
        for p in questions
    ]
    torch.manual_seed(config.data.seed)
    with base_model(model):
        groups = sample_responses(
            model,
            tokenizer,
            prompts,
            n_per_prompt=1,
            max_new_tokens=min(config.eval.max_new_tokens, config.loss.train_max_tokens),
            temperature=config.eval.temperature,
            top_p=config.eval.top_p,
            top_k=config.eval.top_k,
            batch_size=config.eval.batch_size,
        )
    probes = [
        {"question": p.question, "response_ids": group[0].token_ids}
        for p, group in zip(questions, groups, strict=True)
        if len(group[0].token_ids) >= 2
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": config.model.name, "probes": probes}), encoding="utf-8")
    logger.info("Saved %d probes to %s", len(probes), path)
    return probes


def evaluate_checkpoint(
    model: Any,
    tokenizer: Any,
    config: Config,
    benchmarks: dict[str, list[Problem]],
    probes: list[dict[str, Any]],
    score_fn: ScoreFn = is_correct,
) -> dict[str, Any]:
    """Evaluate the model as it stands on every benchmark and on the probes.

    Args:
        model: The model, with the checkpoint's adapter weights loaded.
        tokenizer: Its tokenizer.
        config: The run configuration.
        benchmarks: Problems for each benchmark name.
        probes: Fixed responses for the Gram-distance measure.
        score_fn: Answer checker. Tests pass a stand-in.

    Returns:
        ``{"benchmarks": {name: metrics}, "gram_distance": float}``.
    """
    torch.manual_seed(config.train.seed)
    results: dict[str, Any] = {"benchmarks": {}}
    for name, problems in benchmarks.items():
        results["benchmarks"][name] = evaluate_benchmark(
            model, tokenizer, config, problems, score_fn
        )
        logger.info("%s: Acc@k %.3f", name, results["benchmarks"][name]["acc_at_k"])
    distances = [
        gram_distance(model, tokenizer, config, probe["question"], probe["response_ids"])
        for probe in probes
    ]
    results["gram_distance"] = _mean(distances)
    logger.info("Gram distance from the base model: %.6f", results["gram_distance"])
    return results


def checkpoint_folders(run_dir: Path, names: list[str]) -> list[Path | None]:
    """Resolve checkpoint names to folders.

    Args:
        run_dir: The run folder.
        names: ``"base"`` (no adapter), ``"all"`` (every saved step),
            ``"final"`` (the last saved step), or folder names such as
            ``"step-00020"``.

    Returns:
        Folders in the order requested. ``None`` stands for the base model.
    """
    saved = sorted((run_dir / "checkpoints").glob("step-*"))
    resolved: list[Path | None] = []
    for name in names:
        if name == "base":
            resolved.append(None)
        elif name == "all":
            resolved.extend(saved)
        elif name == "final":
            if not saved:
                raise FileNotFoundError(f"No checkpoints under {run_dir / 'checkpoints'}")
            resolved.append(saved[-1])
        else:
            folder = run_dir / "checkpoints" / name
            if not folder.is_dir():
                raise FileNotFoundError(f"No checkpoint at {folder}")
            resolved.append(folder)
    return resolved


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate checkpoints of a run.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["all"],
        help="'base', 'all', 'final', or step folder names. 'base' must come first.",
    )
    parser.add_argument("--override", action="append", default=[], metavar="SECTION.KEY=VALUE")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Evaluate again even when a checkpoint already has a result file.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config, args.override)
    run_dir = Path(config.train.output_dir)
    configure_logging(log_file=run_dir / "eval.log")

    folders = checkpoint_folders(run_dir, args.checkpoints)
    if None in folders[1:]:
        raise ValueError("'base' must be the first checkpoint, before any adapter is loaded.")
    model, tokenizer, _ = load_student(config.model)
    benchmarks, heldout = benchmark_problems(config)
    probes = load_or_build_probes(model, tokenizer, config, heldout)

    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    for folder in folders:
        name = "base" if folder is None else folder.name
        out_file = out_dir / f"{name}.json"
        if out_file.exists() and not args.force:
            logger.info("Skipping %s: %s already exists", name, out_file)
            continue
        if folder is not None:
            load_adapter_weights(model, folder)
        logger.info("Evaluating %s", name)
        results = evaluate_checkpoint(model, tokenizer, config, benchmarks, probes)
        results.update(
            run=run_dir.name,
            checkpoint=name,
            step=0 if folder is None else int(name.split("-")[-1]),
            model=config.model.name,
            gram_weight=config.loss.gram_weight,
            kl_weight=config.loss.kl_weight,
            seed=config.train.seed,
        )
        out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Wrote %s", out_file)


if __name__ == "__main__":
    main()
