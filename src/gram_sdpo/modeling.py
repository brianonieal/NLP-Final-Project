# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""Model loading, LoRA adapters, and access to the frozen base model.

The student is the base model plus LoRA adapters. Turning the adapters off
gives back the initial model, so one set of weights serves as student,
self-teacher, and Gram anchor.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from gram_sdpo.config import ModelConfig
from gram_sdpo.logging_utils import get_logger

logger = get_logger(__name__)


def pick_device() -> torch.device:
    """Return the GPU when one is available, otherwise the CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Translate the config's dtype name.

    Args:
        name: ``"auto"``, ``"bfloat16"``, ``"float16"``, or ``"float32"``.
        device: Where the model will run. ``"auto"`` means bfloat16 on a GPU
            and float32 on the CPU.

    Returns:
        The torch dtype.
    """
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    try:
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[name]
    except KeyError:
        raise ValueError(f"Unknown dtype {name!r}") from None


def prepare_tokenizer(tokenizer: Any) -> Any:
    """Set the padding a decoder-only model needs for batched generation.

    Args:
        tokenizer: A Hugging Face tokenizer.

    Returns:
        The same tokenizer, padding on the left with a pad token defined.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def attach_lora(model: Any, config: ModelConfig) -> Any:
    """Wrap ``model`` with fresh LoRA adapters.

    A fresh adapter starts as a no-op, so the wrapped model's outputs equal
    the base model's until training changes it.

    Args:
        model: A causal language model.
        config: LoRA rank, scaling, dropout, and target module names.

    Returns:
        The PEFT model. Only adapter weights require gradients.
    """
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_target_modules),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def load_student(config: ModelConfig) -> tuple[Any, Any, torch.device]:
    """Load the base model, attach fresh adapters, and prepare the tokenizer.

    Args:
        config: Model settings.

    Returns:
        ``(model, tokenizer, device)``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    dtype = resolve_dtype(config.dtype, device)
    logger.info("Loading %s on %s as %s", config.name, device, dtype)
    tokenizer = prepare_tokenizer(AutoTokenizer.from_pretrained(config.name))
    base = AutoModelForCausalLM.from_pretrained(config.name, dtype=dtype).to(device)
    model = attach_lora(base, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %d of %d (%.3f%%)", trainable, total, 100 * trainable / total
    )
    return model, tokenizer, device


def load_adapter_weights(model: Any, adapter_dir: str | Path) -> None:
    """Replace the model's adapter weights with a saved checkpoint.

    Args:
        model: A PEFT model with the same adapter layout as the checkpoint.
        adapter_dir: Folder written by ``model.save_pretrained``.
    """
    from peft.utils import load_peft_weights, set_peft_model_state_dict

    weights = load_peft_weights(str(adapter_dir), device=str(next(model.parameters()).device))
    set_peft_model_state_dict(model, weights)
    logger.info("Loaded adapter weights from %s", adapter_dir)


@contextmanager
def base_model(model: Any) -> Iterator[Any]:
    """Run the enclosed code on the initial model: adapters off, no gradients.

    The model is also put in evaluation mode inside the block, so dropout
    never adds noise to the self-teacher or the Gram anchor. The previous
    mode is restored on exit.

    Args:
        model: A PEFT model.

    Yields:
        The same model object, behaving as the base model inside the block.
    """
    was_training = model.training
    model.eval()
    try:
        with model.disable_adapter(), torch.no_grad():
            yield model
    finally:
        model.train(was_training)


def release_gpu_cache() -> None:
    """Hand cached GPU memory back so the next large allocation can use it.

    Sampling leaves large cached blocks behind. Without this call the training
    passes that follow can hit out-of-memory warnings on a 24 GB card even
    though the memory is no longer in use. Does nothing on the CPU.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
