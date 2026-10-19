"""Smoke test: verify core data dependencies import correctly."""

import numpy
import pandas
import pyarrow


def test_pandas_imports():
    assert pandas.__version__


def test_numpy_imports():
    assert numpy.__version__


def test_pyarrow_imports():
    assert pyarrow.__version__
