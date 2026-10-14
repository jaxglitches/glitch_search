import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # small arrays; CPU keeps numbers identical run to run
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from glitchsearch import Grid  # noqa: E402


@pytest.fixture(scope="session")
def grid():
    return Grid()
