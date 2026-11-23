# GenAI disclosure: Claude (Anthropic) generated this file. Brian Onieal reviewed it.
"""Shared fixtures: a tiny random model and a character tokenizer.

Nothing here downloads anything, so the suite runs offline in seconds.
"""

import pytest
import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from gram_sdpo.config import Config
from gram_sdpo.modeling import attach_lora, prepare_tokenizer

CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 +-=?.,:{}\\\n"


@pytest.fixture
def tiny_tokenizer():
    """A tokenizer with one token per character and a pass-through chat template."""
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    for character in CHARACTERS:
        vocab.setdefault(character, len(vocab))
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Split(Regex("[\\s\\S]"), "isolated")
    core.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}\n"
    return prepare_tokenizer(tokenizer)


@pytest.fixture
def tiny_config(tmp_path):
    """A configuration sized for the tiny model, writing under ``tmp_path``."""
    config = Config()
    config.model.name = "tiny-random"
    config.model.enable_thinking = False
    config.model.lora_r = 2
    config.model.lora_alpha = 4
    config.model.lora_target_modules = ["c_attn"]
    config.data.train_dataset = "toy"
    config.data.heldout_size = 4
    config.rollout.rollouts_per_question = 2
    config.rollout.max_new_tokens = 8
    config.rollout.batch_size = 2
    config.loss.train_max_tokens = 8
    config.loss.gram_tokens = 6
    config.loss.top_k_logits = 10
    config.train.steps = 2
    config.train.questions_per_step = 2
    config.train.save_every = 1
    config.train.warmup_steps = 1
    config.train.learning_rate = 1e-2
    config.train.output_dir = str(tmp_path / "run")
    config.eval.benchmarks = ["heldout"]
    config.eval.samples_per_problem = 2
    config.eval.max_new_tokens = 8
    config.eval.batch_size = 2
    config.eval.probe_size = 2
    config.eval.probe_file = str(tmp_path / "probes.json")
    return config


@pytest.fixture
def tiny_model(tiny_tokenizer, tiny_config):
    """A two-layer random GPT-2 with LoRA adapters attached."""
    torch.manual_seed(0)
    gpt_config = GPT2Config(
        vocab_size=len(tiny_tokenizer),
        n_positions=512,
        n_embd=32,
        n_layer=2,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    return attach_lora(GPT2LMHeadModel(gpt_config), tiny_config.model)
