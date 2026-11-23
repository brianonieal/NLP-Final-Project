# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""Experiment configuration, loaded from a YAML file into typed dataclasses.

Every experiment in the proposal is one YAML file under ``configs/``. Unknown
keys raise an error so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Which model to load and how LoRA adapters are attached to it."""

    name: str = "Qwen/Qwen3-1.7B"
    dtype: str = "auto"
    enable_thinking: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


@dataclass
class DataConfig:
    """Training data and the in-domain held-out split."""

    train_dataset: str = "dapo-math-17k"
    heldout_size: int = 200
    max_train_questions: int | None = None
    seed: int = 0


@dataclass
class RolloutConfig:
    """How the student samples responses during training."""

    rollouts_per_question: int = 8
    max_new_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    batch_size: int = 8


@dataclass
class LossConfig:
    """The SDPO loss and the two regularizers compared in the proposal."""

    divergence: str = "kl"
    top_k_logits: int | None = 100
    train_max_tokens: int = 2048
    solution_includes_thinking: bool = True
    gram_weight: float = 0.0
    gram_layers: list[int] = field(default_factory=lambda: [-1])
    gram_tokens: int = 256
    gram_target: str = "base"
    kl_weight: float = 0.0


@dataclass
class TrainConfig:
    """Optimization settings and where a run writes its files."""

    steps: int = 100
    questions_per_step: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10
    grad_clip: float = 1.0
    save_every: int = 20
    output_dir: str = "runs/sdpo"
    seed: int = 0


@dataclass
class EvalConfig:
    """Benchmarks, sampling settings, and the Gram-distance probe set."""

    benchmarks: list[str] = field(default_factory=lambda: ["heldout", "aime24", "amc23"])
    samples_per_problem: int = 16
    max_new_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    batch_size: int = 8
    max_problems: int | None = None
    probe_size: int = 32
    probe_file: str = "runs/probes.json"


@dataclass
class Config:
    """Everything one run needs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return the configuration as plain nested dictionaries."""
        return asdict(self)

    def validate(self) -> None:
        """Raise ``ValueError`` when a setting cannot be run."""
        if self.loss.divergence not in {"kl", "jsd"}:
            raise ValueError(f"loss.divergence must be 'kl' or 'jsd', got {self.loss.divergence!r}")
        if self.loss.gram_target not in {"base", "teacher"}:
            raise ValueError(
                f"loss.gram_target must be 'base' or 'teacher', got {self.loss.gram_target!r}"
            )
        if self.loss.gram_weight < 0 or self.loss.kl_weight < 0:
            raise ValueError("loss.gram_weight and loss.kl_weight must be zero or positive")
        if self.rollout.rollouts_per_question < 1:
            raise ValueError("rollout.rollouts_per_question must be at least 1")
        if not self.loss.gram_layers:
            raise ValueError("loss.gram_layers needs at least one layer index")


def _build(cls: type, values: dict[str, Any], path: str) -> Any:
    """Build dataclass ``cls`` from ``values`` and reject unknown keys.

    Args:
        cls: The dataclass to build.
        values: Keys and values read from YAML for this section.
        path: Dotted location of this section, used in error messages.

    Returns:
        An instance of ``cls``.
    """
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"Unknown config key(s) under '{path or 'root'}': {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value or {}, f"{path}.{name}".strip("."))
        else:
            kwargs[name] = value
    return cls(**kwargs)


def apply_overrides(values: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides to a nested dictionary.

    Args:
        values: Nested dictionary read from YAML. It is changed in place.
        overrides: Strings such as ``"loss.gram_weight=0.1"``. Values are parsed
            as YAML, so numbers, booleans, and lists work.

    Returns:
        The same dictionary, for chaining.
    """
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.split(".")
        node = values
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(raw)
    return values


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load, override, and validate a configuration file.

    Args:
        path: Path to a YAML file.
        overrides: Optional ``section.key=value`` strings applied on top.

    Returns:
        The validated configuration.
    """
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    apply_overrides(values, overrides or [])
    config = _build(Config, values, "")
    config.validate()
    return config
