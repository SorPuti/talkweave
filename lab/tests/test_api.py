"""Verify public streaming contracts and documentation conventions."""

import ast
from pathlib import Path
import numpy as np
import pytest
import torch
from api import Frame
from core import Codec, Duplex
from infer import Session


def test_frame_contract_and_replay(tmp_path):
    """Reject malformed or replayed frames at the engine boundary."""
    torch.save(Codec().state_dict(), tmp_path / "codec.pt")
    torch.save(Duplex().state_dict(), tmp_path / "duplex.pt")
    session = Session(tmp_path)
    result = session.process(Frame(np.zeros(320, dtype="float32"), seq=0))
    assert result.audio.shape == (320,)
    assert np.isfinite(result.audio).all() and result.elapsed_ms > 0
    assert set(result.events) == {"speech", "laughter", "noise", "cough", "music"}
    with pytest.raises(ValueError):
        session.process(Frame(np.zeros(320, dtype="float32"), seq=0))
    with pytest.raises(ValueError):
        session.process(Frame(np.zeros(320, dtype="float32"), seq=1, rate=48000))
    with pytest.raises(ValueError):
        session.process(Frame(np.full(320, np.nan, dtype="float32"), seq=1))


def test_single_line_docstrings():
    """Require concise ASCII documentation on implementation boundaries."""
    root = Path(__file__).resolve().parents[1]
    files = list((root / "src").glob("*.py")) + [root / "run.py"]
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node)
                assert doc and len(doc.splitlines()) == 1, (path.name, node.name)
                assert doc.isascii(), (path.name, node.name)
