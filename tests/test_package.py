# GenAI disclosure: Claude (Anthropic) generated this file as part of the
# initial repo skeleton.
"""Checks that the package installs and exposes its public names."""

import gram_sdpo


def test_version_is_a_nonempty_string():
    """The package reports a version."""
    assert isinstance(gram_sdpo.__version__, str)
    assert gram_sdpo.__version__


def test_public_api_is_importable():
    """Every name in ``__all__`` exists on the package."""
    for name in gram_sdpo.__all__:
        assert hasattr(gram_sdpo, name)
