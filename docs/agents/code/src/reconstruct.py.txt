"""Implement the reconstruct component of the CPU duplex tutorial."""

import argparse
from pathlib import Path
import torch
import soundfile as sf
from core import Codec, frames, read_audio, SR


def main():
    """Run the command-line workflow and report its outcome."""
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--run", default="runs")
    p.add_argument("--output", default="reconstructed.wav")
    args = p.parse_args()
    torch.set_num_threads(2)
    codec = Codec().eval()
    codec.load_state_dict(
        torch.load(Path(args.run) / "codec.pt", weights_only=True, map_location="cpu")
    )
    x = read_audio(args.input)
    with torch.inference_mode():
        codes = codec.encode(frames(x))
        y = codec.decode(codes).reshape(-1).numpy()[: len(x)]
    sf.write(args.output, y, SR)
    print(
        f"{len(codes.unique())}/256 códigos usados neste arquivo; ouça a reconstrução antes de treinar o diálogo."
    )


if __name__ == "__main__":
    main()
