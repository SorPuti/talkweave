"""Implement the baseline component of the CPU duplex tutorial."""

import numpy as np


def features(x):
    """Extract frame energy and zero-crossing rate."""
    return np.array(
        [
            np.sqrt(np.mean(x * x) + 1e-12),
            np.mean(np.signbit(x[1:]) != np.signbit(x[:-1])),
        ]
    )


def main():
    """Run the command-line workflow and report its outcome."""
    rng = np.random.default_rng(7)
    data = []
    labels = []
    for i in range(400):
        t = np.arange(320) / 16000
        y = i % 2
        x = (
            rng.normal(0, 0.12, 320)
            if y
            else 0.12 * np.sin(2 * np.pi * rng.uniform(100, 450) * t)
        )
        data.append(features(x))
        labels.append(y)
    x = np.array(data)
    y = np.array(labels)
    idx = rng.permutation(len(y))
    tr, te = idx[:300], idx[300:]
    mean = x[tr].mean(0)
    std = x[tr].std(0) + 1e-8
    z = (x - mean) / std
    w = np.zeros(2)
    b = 0.0
    for step in range(300):
        prob = 1 / (1 + np.exp(-np.clip(z[tr] @ w + b, -30, 30)))
        error = prob - y[tr]
        w -= 0.05 * (z[tr].T @ error / len(tr) + 0.01 * w)
        b -= 0.05 * error.mean()
    accuracy = ((z[te] @ w + b > 0) == y[te]).mean()
    print(
        f"Teste sintético tom/ruído: acurácia={accuracy:.3f}; não é detector de fala humana."
    )
    np.savez("baseline.npz", w=w, b=b, mean=mean, std=std)


if __name__ == "__main__":
    main()
