# GenAI disclosure: Claude (Anthropic) generated this file as part of the
# initial repo skeleton.
"""Tests for the shared logging setup."""

import logging

import pytest

from gram_sdpo.logging_utils import PACKAGE_LOGGER_NAME, configure_logging, get_logger


@pytest.fixture(autouse=True)
def reset_package_logger():
    """Leave the package logger with no handlers after each test."""
    yield
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_get_logger_without_name_returns_package_logger():
    """No name means the package logger itself."""
    assert get_logger().name == PACKAGE_LOGGER_NAME


def test_get_logger_keeps_names_already_inside_the_package():
    """A module's ``__name__`` is used as is."""
    assert get_logger("gram_sdpo.losses").name == "gram_sdpo.losses"


def test_get_logger_nests_outside_names_under_the_package():
    """A script name such as ``__main__`` is nested under the package."""
    assert get_logger("__main__").name == "gram_sdpo.__main__"


def test_configure_logging_sets_the_level():
    """The level argument is applied, by constant or by name."""
    assert configure_logging(level=logging.WARNING).level == logging.WARNING
    assert configure_logging(level="DEBUG").level == logging.DEBUG


def test_configure_logging_twice_does_not_duplicate_handlers():
    """A second call replaces handlers instead of stacking them."""
    configure_logging()
    logger = configure_logging()
    assert len(logger.handlers) == 1


def test_configure_logging_writes_to_file(tmp_path):
    """Messages reach the log file, and its parent folder is created."""
    log_file = tmp_path / "logs" / "run.log"
    configure_logging(log_file=log_file)
    get_logger("gram_sdpo.train").info("step %d done", 3)
    for handler in logging.getLogger(PACKAGE_LOGGER_NAME).handlers:
        handler.flush()
    text = log_file.read_text(encoding="utf-8")
    assert "step 3 done" in text
    assert "gram_sdpo.train" in text


def test_child_messages_below_the_level_are_dropped(tmp_path):
    """A DEBUG message is not written when the level is INFO."""
    log_file = tmp_path / "run.log"
    configure_logging(level=logging.INFO, log_file=log_file)
    get_logger("gram_sdpo.eval").debug("hidden")
    for handler in logging.getLogger(PACKAGE_LOGGER_NAME).handlers:
        handler.flush()
    assert "hidden" not in log_file.read_text(encoding="utf-8")
