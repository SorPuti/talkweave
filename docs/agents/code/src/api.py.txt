"""Implement the api component of the CPU duplex tutorial."""

from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class Frame:
    """Define the Frame component contract."""

    samples: np.ndarray
    seq: int = 0
    rate: int = 16000


@dataclass(frozen=True)
class Result:
    """Define the Result component contract."""

    audio: np.ndarray
    user: str
    agent: str
    events: dict[str, float]
    elapsed_ms: float


class Engine(Protocol):
    """Define the Engine component contract."""

    def process(self, frame: Frame) -> Result:
        """Validate and process one frame through the engine contract."""
        ...
