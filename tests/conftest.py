import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.fst_analyzer import FSTAnalyzer  # noqa: E402
from src.resources import load_resources  # noqa: E402
from src.rule_based_analyzer import RuleBasedAnalyzer  # noqa: E402


@pytest.fixture(scope="session")
def resources():
    return load_resources()


@pytest.fixture(scope="session")
def rb(resources):
    return RuleBasedAnalyzer(resources, enable_guesser=False)


@pytest.fixture(scope="session")
def rb_guess(resources):
    return RuleBasedAnalyzer(resources, enable_guesser=True)


@pytest.fixture(scope="session")
def fst(resources):
    return FSTAnalyzer(resources)


def shorts(result):
    """Set of compact analysis strings, e.g. {'लड़का NOUN MASC PL OBL'}."""
    return {a.short() for a in result.analyses}
