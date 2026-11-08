"""Implement the record component of the CPU duplex tutorial."""

import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from core import SR


def main():
    """Run the command-line workflow and report its outcome."""
    p = argparse.ArgumentParser()
    p.add_argument("--input", help="WAV estéreo já sincronizado, 16 kHz")
    p.add_argument("--seconds", type=float, default=6)
    p.add_argument("--session", required=True)
    p.add_argument("--speakers", nargs=2, required=True)
    p.add_argument("--split", choices=["train", "validation", "test"], default="train")
    args = p.parse_args()
    if not args.session.replace("-", "").replace("_", "").isalnum():
        p.error("ID deve conter letras/números, - ou _")
    if not 0 < args.seconds <= 6:
        p.error("Grave exemplos de até seis segundos nesta versão")
    root = Path("data")
    root.mkdir(exist_ok=True)
    up, ap = root / f"{args.session}-user.wav", root / f"{args.session}-agent.wav"
    if up.exists() or ap.exists():
        raise FileExistsError("Sessão já existe; escolha outro ID")
    if args.input:
        x, sr = sf.read(args.input, dtype="float32", always_2d=True)
        if sr != SR or x.shape[1] != 2:
            raise ValueError("Esperado estéreo 16 kHz")
        if len(x) > 6 * SR:
            raise ValueError("Recorte em sessões de até seis segundos")
    else:
        import sounddevice as sd

        print("Dois canais: entrada 1 usuário, entrada 2 agente. Gravando...")
        x = sd.rec(int(args.seconds * SR), samplerate=SR, channels=2, dtype="float32")
        sd.wait()
    if not np.isfinite(x).all():
        raise ValueError("Áudio não finito")
    sf.write(up, x[:, 0], SR)
    sf.write(ap, x[:, 1], SR)
    row = {
        "session": args.session,
        "speakers": args.speakers,
        "language": "pt",
        "user": up.name,
        "agent": ap.name,
        "user_words": [],
        "agent_words": [],
        "events": [],
    }
    with (root / f"{args.split}.jsonl").open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        "Agora anote user_words, agent_words e events no manifesto. Não treine com rótulos vazios por engano."
    )


if __name__ == "__main__":
    main()
