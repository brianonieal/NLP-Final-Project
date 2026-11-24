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

The experiment code is written and tested on CPU with small models. It has
not been run on a GPU with Qwen3-1.7B yet, and there are no results yet.

## Setup

```bash
conda env create -f environment.yml
conda activate nlp-final-project
pytest
```

## Usage

Each experiment is one YAML file in `configs/`. A run has three steps.

```bash
# 1. Train. Writes runs/<name>/ with a log and adapter checkpoints.
python -m gram_sdpo.train --config configs/sdpo.yaml

# 2. Evaluate. "base" scores the untrained model, "all" scores every checkpoint.
python -m gram_sdpo.evaluate --config configs/sdpo.yaml --checkpoints base all

# 3. Collect every run into runs/results.csv and runs/results.md.
python -m gram_sdpo.report --runs runs
```

The configs map to the proposal's experiments:

| Config | What it runs |
| --- | --- |
| `configs/smoke.yaml` | A few-minute CPU check of the whole pipeline. Numbers mean nothing. |
| `configs/sdpo.yaml` | SDPO with no regularizer (Experiment 1, and the SDPO row of Experiment 2) |
| `configs/sdpo_kl.yaml` | SDPO plus the KL penalty baseline (Experiment 2) |
| `configs/sdpo_gram.yaml` | SDPO plus Gram anchoring (Experiment 2) |

Change any setting from the command line with `--override`:

```bash
# Sweep the Gram weight.
python -m gram_sdpo.train --config configs/sdpo_gram.yaml \
    --override loss.gram_weight=0.1 train.output_dir=runs/sdpo_gram_w0.1

# Experiment 3: anchor to the self-teacher instead of the base model.
python -m gram_sdpo.train --config configs/sdpo_gram.yaml \
    --override loss.gram_target=teacher train.output_dir=runs/sdpo_gram_teacher

# Experiment 4: anchor a middle layer and the final layer.
python -m gram_sdpo.train --config configs/sdpo_gram.yaml \
    --override "loss.gram_layers=[14, -1]" train.output_dir=runs/sdpo_gram_layers
```

The same pieces are available from Python:

```python
from gram_sdpo import load_config
from gram_sdpo.train import train

config = load_config("configs/smoke.yaml", ["train.steps=1"])
run_folder = train(config)
```

### Running on Google Colab

`notebooks/colab_run.ipynb` installs the code on a Colab GPU, saves runs to
Google Drive, runs a short timing test, and then runs the experiments. A run
cut off by a dropped session continues with `--resume`, and evaluation skips
checkpoints that already have a result file.

### Running on a rented GPU

`scripts/run_plan.sh` runs one piece of the plan at a time on any single-GPU
Linux machine: `timing`, `sdpo`, `sdpo_kl`, `sdpo_gram <weight>`, and `report`.
Every piece can be rerun after an interruption.

## How the pieces connect

| Module | Job |
| --- | --- |
| `config.py` | Loads a YAML file into typed settings |
| `data.py` | Downloads the datasets and builds the student and self-teacher prompts |
| `modeling.py` | Loads the model, attaches LoRA adapters, and turns them off to get the base model |
| `rollout.py` | Samples responses in batches |
| `scoring.py` | Checks boxed answers and counts uncertainty words |
| `losses.py` | SDPO divergence, Gram loss, token sampling |
| `sdpo.py` | Combines the three forward passes into the loss for one example |
| `train.py` | The training loop |
| `evaluate.py` | Acc@k, response length, uncertainty words, Gram distance |
| `report.py` | Collects results into one table |

TODO (team): diagram showing how the pieces of the package connect
(draw.io or similar), saved to `docs/architecture.png` and embedded here.

## Data

No data is stored in this repository. The code downloads each dataset from
the Hugging Face Hub the first time it is needed:

| Name | Source | Use |
| --- | --- | --- |
| `dapo-math-17k` | `BytedTsinghua-SIA/DAPO-Math-17k` | Training, plus a held-out in-domain test split |
| `aime24` | `BytedTsinghua-SIA/AIME-2024` | Out-of-distribution test |
| `amc23` | `zwhe99/amc23` | Out-of-distribution test |

TODO (team): confirm these are the same AIME24 and AMC23 files Kim et al.
(2026) used. Their paper does not name the source.

## Development

```bash
ruff format .
ruff check .
pytest
```

Every function, method, and class gets a docstring. Ruff enforces this
through the `D` rules in `pyproject.toml`.
