"""Implement the train component of the CPU duplex tutorial."""

import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from core import Codec, Duplex, CHARS, load_rows, shifted, seed


def main():
    """Run the command-line workflow and report its outcome."""
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["codec", "duplex"])
    p.add_argument("--manifest", required=True)
    p.add_argument("--validation")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--output", default="runs")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Duplex activation dropout; 0 disables it",
    )
    args = p.parse_args()
    if args.steps < 1:
        p.error("steps precisa ser positivo")
    if not 0.0 <= args.dropout < 1.0:
        p.error("dropout precisa estar no intervalo [0, 1)")
    if args.stage == "codec" and args.dropout != 0:
        p.error("--dropout se aplica ao estágio duplex, não ao codec")
    torch.set_num_threads(args.threads)
    seed()
    root = Path(args.output)
    root.mkdir(exist_ok=True, parents=True)
    rows = load_rows(args.manifest)
    validation = load_rows(args.validation) if args.validation else []

    # Evita vazamento por sessão/locutor. Se houver dois locutores, liste ambos em speakers.
    def identities(items):
        """Collect speaker identifiers for split leakage checks."""
        return {
            s
            for r in items
            for s in r["meta"].get("speakers", [r["meta"].get("speaker", "")])
            if s
        }

    if validation and identities(rows) & identities(validation):
        raise ValueError("Locutor compartilhado entre treino e validação")
    device = torch.device(args.device)
    codec = Codec().to(device)
    if args.stage == "codec":
        train_frames = torch.cat([r[k] for r in rows for k in ("user", "agent")]).to(
            device
        )
        val_frames = (
            torch.cat([r[k] for r in validation for k in ("user", "agent")]).to(device)
            if validation
            else None
        )
        model = codec

        def objective(is_val=False):
            """Compute the current stage loss on a sampled example."""
            source = val_frames if is_val else train_frames
            ids = torch.randint(len(source), (min(256, len(source)),), device=device)
            return model(source[ids])[1]
    else:
        codec.load_state_dict(
            torch.load(root / "codec.pt", map_location=device, weights_only=True)
        )
        codec.eval().requires_grad_(False)
        with torch.no_grad():
            silence = int(codec.encode(torch.zeros(1, 320, device=device)).item())
            for row in rows + validation:
                row["u"] = codec.encode(row["user"].to(device))
                row["a"] = codec.encode(row["agent"].to(device))
        model = Duplex(dropout=args.dropout).to(device)
        text_weights = torch.ones(len(CHARS), device=device)
        text_weights[0] = 0.05
        audio_weights = torch.ones(256, device=device)
        audio_weights[silence] = 0.2

        def objective(is_val=False):
            """Compute the current stage loss on a sampled example."""
            source = validation if is_val else rows
            r = source[int(torch.randint(len(source), ()).item())]
            # Prefixo aquece o estado: começar no meio sem contexto prejudica a causalidade aprendida.
            n = min(len(r["u"]), 300)
            u, a = r["u"][:n], r["a"][:n]
            ut, at, ev = (
                r[k][:n].to(device) for k in ("user_text", "agent_text", "events")
            )
            al, ul, tl, el, _ = model(
                u[None],
                shifted(a, silence)[None],
                shifted(at, 0)[None],
                teacher_text=at[None],
            )
            return (
                F.cross_entropy(al[0], a, weight=audio_weights)
                + 0.5 * F.cross_entropy(ul[0], ut, weight=text_weights)
                + 0.5 * F.cross_entropy(tl[0], at, weight=text_weights)
                + 0.2 * F.binary_cross_entropy_with_logits(el[0], ev)
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    log = []
    for step in range(1, args.steps + 1):
        model.train()
        loss = objective()
        if not torch.isfinite(loss):
            raise RuntimeError("Loss não finita: confira dados e learning rate")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 25 == 0 or step == args.steps:
            record = {"step": step, "loss": float(loss.detach())}
            if validation:
                model.eval()
                with torch.no_grad():
                    record["validation_loss"] = float(objective(True))
            log.append(record)
            print(json.dumps(record), flush=True)
    torch.save(model.cpu().state_dict(), root / f"{args.stage}.pt")
    (root / f"{args.stage}-metrics.json").write_text(json.dumps(log, indent=2))
    (root / "environment.txt").write_text(f"torch={torch.__version__}\nseed=7\n")
    (root / f"{args.stage}-config.json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
