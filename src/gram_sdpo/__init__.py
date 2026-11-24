# GenAI disclosure: Claude (Anthropic) generated this file as part of the
# initial repo skeleton.
"""Gram loss regularization for self-distillation of a reasoning LLM."""

from gram_sdpo.config import Config, load_config
from gram_sdpo.logging_utils import configure_logging, get_logger

__version__ = "0.1.0"

__all__ = ["Config", "__version__", "configure_logging", "get_logger", "load_config"]
