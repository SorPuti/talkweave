"""Implement the fixture component of the CPU duplex tutorial."""

import json
from pathlib import Path
import numpy as np
import soundfile as sf
from core import SR


def main():
    """Run the command-line workflow and report its outcome."""
    p = Path("data")
    p.mkdir(exist_ok=True)
    t = np.arange(SR * 2) / SR
    for i, split in enumerate(["train", "validation"]):
        u = (0.15 * np.sin(2 * np.pi * (220 + i * 30) * t) * (t < 1)).astype("float32")
        a = (0.15 * np.sin(2 * np.pi * (440 + i * 30) * t) * (t > 0.7)).astype(
            "float32"
        )
        sf.write(p / f"{split}-user.wav", u, SR)
        sf.write(p / f"{split}-agent.wav", a, SR)
        row = {
            "session": split,
            "speakers": [f"fixture-{i}"],
            "language": "synthetic",
            "user": f"{split}-user.wav",
            "agent": f"{split}-agent.wav",
            "user_words": [{"start": 0.1, "end": 0.5, "text": "tom "}],
            "agent_words": [{"start": 0.8, "end": 1.3, "text": "eco "}],
            "events": [{"start": 0, "end": 1, "label": "music"}],
        }
        (p / f"{split}.jsonl").write_text(json.dumps(row) + "\n")
    print(
        "Fixture gerado. As palavras tom/eco são rótulos artificiais de teste, não transcrição real."
    )


if __name__ == "__main__":
    main()
