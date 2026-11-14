<!-- GenAI disclosure: Claude (Anthropic) generated this file as part of the
     initial repo skeleton. Sections marked TODO are not written yet. -->

# NLP Final Project: Gram Loss for Self-Distillation

EN.705.641.81 Natural Language Processing: Self-Supervised Models
Johns Hopkins University Engineering for Professionals, Fall 2026

## Team

- Brian W. Onieal
- Morgan Zinn
- Rajiv Khandpur

## Aim

This package tests whether adding a Gram loss regularizer to self-distillation
reduces the drop in out-of-distribution reasoning that self-distillation can
cause in a reasoning LLM. Experiments use Qwen3-1.7B.

TODO (team): expand with the research question, the papers this builds on,
the datasets, and the evaluation.

## Status

Skeleton only. The package installs, logging is wired, and the test suite
runs. No training or evaluation code yet.

## Setup

```bash
conda env create -f environment.yml
conda activate nlp-final-project
pytest
```

## Usage

TODO (team): a short demonstration of the key features, including the
commands that reproduce each experiment.

## Architecture

TODO (team): diagram showing how the pieces of the package connect
(draw.io or similar), saved to `docs/architecture.png` and embedded here.

## Data

No data is stored in this repository. TODO (team): the download step and
where the data lands locally. The `data/` folder is git-ignored.

## Development

```bash
ruff format .
ruff check .
pytest
```

Every function, method, and class gets a docstring. Ruff enforces this
through the `D` rules in `pyproject.toml`.
