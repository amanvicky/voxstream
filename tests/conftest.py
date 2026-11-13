"""pytest configuration for the VoxStream verification suite.

The suite is dual-mode: it runs as a plain script (colored section report,
exit code = failures) and each ``test_*`` function doubles as a pytest test.
Pytest-only niceties live here.
"""

import warnings

import pytest


def pytest_configure(config):
    # Section functions return human-readable detail strings; pytest >= 8
    # warns about non-None returns. The values are the report, not a mistake.
    warnings.filterwarnings("ignore", category=pytest.PytestReturnNotNoneWarning)
