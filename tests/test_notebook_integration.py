"""Execute notebook_integration.py in a real Jupyter kernel.

Any unhandled exception in a notebook cell fails the test.
The DOCUMENTATION=1 environment variable is set so that JSVisualization
does not attempt to start a local HTTP server.
"""

from pathlib import Path

import pytest

NOTEBOOK_PY = Path(__file__).parent / "notebook_integration.py"


def test_notebook_integration(monkeypatch):
    jupytext = pytest.importorskip("jupytext")
    pytest.importorskip("nbconvert")
    from nbconvert.preprocessors import ExecutePreprocessor

    monkeypatch.setenv("DOCUMENTATION", "1")

    nb = jupytext.read(str(NOTEBOOK_PY))
    ep = ExecutePreprocessor(timeout=120, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(NOTEBOOK_PY.parent)}})
