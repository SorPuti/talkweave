"""Implement the evaluate component of the CPU duplex tutorial."""

import argparse
import json
import numpy as np
from core import EVENTS, load_rows, token_text, wer


def main():
    """Run the command-line workflow and report its outcome."""
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args()
    rows = load_rows(args.manifest)
    if len(rows) != 1:
        raise ValueError("Avalie uma sessão por relatório")
    r = rows[0]
    report = json.load(open(args.report))
    prediction = np.asarray(report["event_probabilities"]) >= 0.5
    target = r["events"].numpy().astype(bool)
    if prediction.shape != target.shape:
        raise ValueError("Duração não coincide")
    result = {
        "user_wer": wer(token_text(r["user_text"]), report["user_text"]),
        "agent_wer": wer(token_text(r["agent_text"]), report["agent_text"]),
    }
    for j, label in enumerate(EVENTS):
        tp = int((prediction[:, j] & target[:, j]).sum())
        fp = int((prediction[:, j] & ~target[:, j]).sum())
        fn = int((~prediction[:, j] & target[:, j]).sum())
        result[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        }
    print(json.dumps(result, indent=2))
    print(
        "F1 por quadro, não por evento. Ausência de positivos não comprova qualidade."
    )


if __name__ == "__main__":
    main()
