"""Implement the core component of the CPU duplex tutorial."""

import json
import math
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

SR = 16000
FRAME = 320  # 20 ms, 50 quadros/s
EVENTS = ["speech", "laughter", "noise", "cough", "music"]
VOCAB = "<blank>| abcdefghijklmnopqrstuvwxyzáàâãéêíóôõúüç0123456789.,?!:;-'/"
CHARS = [""] + list(VOCAB.split("|", 1)[1])
CHAR_ID = {c: i for i, c in enumerate(CHARS) if c}


def normalize(text):
    """Normalize text to lowercase NFC Unicode."""
    return unicodedata.normalize("NFC", text.lower())


def read_audio(path):
    """Read and validate a mono 16 kHz audio file."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR or x.shape[1] != 1:
        raise ValueError(f"{path}: esperado WAV mono {SR} Hz; recebido {sr}, {x.shape}")
    if not np.isfinite(x).all() or np.max(np.abs(x), initial=0) > 1:
        raise ValueError("Áudio fora da faixa [-1, 1] ou não finito")
    return x[:, 0]


def frames(x):
    """Pad and reshape samples into fixed audio frames."""
    return torch.from_numpy(np.pad(x, (0, (-len(x)) % FRAME))).view(-1, FRAME)


def aligned_chars(words, count):
    """Place annotated characters on the frame clock without overlap."""
    y = torch.zeros(count, dtype=torch.long)
    for word in words:
        text = normalize(word["text"])
        if not text:
            continue
        start, end = float(word["start"]), float(word["end"])
        if not (0 <= start < end <= count * FRAME / SR):
            raise ValueError("Intervalo de palavra inválido")
        lo, hi = int(start * SR / FRAME), min(count, int(end * SR / FRAME))
        if hi - lo < len(text):
            raise ValueError(
                "Mais caracteres que quadros: use tokens subword ou corrija o alinhamento"
            )
        positions = np.floor(np.linspace(lo, hi, len(text), endpoint=False)).astype(int)
        for t, char in zip(positions, text):
            if char not in CHAR_ID:
                raise ValueError(f"Caractere fora do vocabulário: {char!r}")
            if y[t] != 0:
                raise ValueError("Palavras se sobrepõem no mesmo canal")
            y[t] = CHAR_ID[char]
    return y


def load_rows(manifest):
    """Load synchronized audio, text, and event labels from JSONL."""
    p = Path(manifest).resolve()
    rows = []
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        u, a = read_audio(p.parent / row["user"]), read_audio(p.parent / row["agent"])
        if len(u) != len(a) or len(u) == 0:
            raise ValueError("Canais precisam da mesma duração e origem temporal")
        uf, af = frames(u), frames(a)
        n = len(uf)
        ev = torch.zeros(n, len(EVENTS))
        for e in row.get("events", []):
            if (
                e["label"] not in EVENTS
                or not 0 <= e["start"] < e["end"] <= len(u) / SR
            ):
                raise ValueError("Evento ou intervalo inválido")
            ev[
                int(e["start"] * 50) : math.ceil(e["end"] * 50),
                EVENTS.index(e["label"]),
            ] = 1
        rows.append(
            dict(
                user=uf,
                agent=af,
                user_text=aligned_chars(row.get("user_words", []), n),
                agent_text=aligned_chars(row.get("agent_words", []), n),
                events=ev,
                meta=row,
            )
        )
    if not rows:
        raise ValueError("Manifesto vazio")
    return rows


class Codec(nn.Module):
    """Define the Codec component contract."""

    def __init__(self, codes=256, dim=32):
        """Initialize the component and its bounded state."""
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(FRAME, 128), nn.GELU(), nn.Linear(128, dim))
        self.book = nn.Embedding(codes, dim)
        nn.init.uniform_(self.book.weight, -0.1, 0.1)
        self.dec = nn.Sequential(
            nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, FRAME), nn.Tanh()
        )

    def nearest(self, z):
        """Find the closest codebook vector for each latent vector."""
        distance = (
            z.square().sum(-1, keepdim=True)
            + self.book.weight.square().sum(-1)
            - 2 * z @ self.book.weight.T
        )
        return distance.argmin(-1)

    def encode(self, x):
        """Map audio frames to discrete code indices."""
        return self.nearest(self.enc(x))

    def decode(self, code):
        """Reconstruct audio frames from discrete code indices."""
        return self.dec(self.book(code))

    def forward(self, x):
        """Compute causal predictions and return the updated state."""
        z = self.enc(x)
        ids = self.nearest(z)
        q = self.book(ids)
        straight = z + (q - z).detach()
        reconstruction = self.dec(straight)
        vq = F.mse_loss(q, z.detach()) + 0.25 * F.mse_loss(z, q.detach())
        # Penalização espectral simples; não substitui perdas perceptuais modernas.
        spec = F.l1_loss(torch.fft.rfft(reconstruction).abs(), torch.fft.rfft(x).abs())
        loss = F.l1_loss(reconstruction, x) + 0.01 * spec + vq
        return reconstruction, loss, ids


class Duplex(nn.Module):
    """Define the Duplex component contract."""

    def __init__(self, hidden=128, dropout=0.0):
        """Initialize the component and its bounded state."""
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1)")
        self.input_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.audio_embedding = nn.Embedding(256, 48)
        self.text_embedding = nn.Embedding(len(CHARS), 24)
        self.rnn = nn.GRU(48 + 48 + 24, hidden, batch_first=True)
        self.user_text = nn.Linear(hidden, len(CHARS))
        self.agent_text = nn.Linear(hidden, len(CHARS))
        self.event_head = nn.Linear(hidden, len(EVENTS))
        self.audio_head = nn.Linear(hidden + 24, 256)

    def forward(
        self, user, previous_audio, previous_text, hidden=None, teacher_text=None
    ):
        """Compute causal predictions and return the updated state."""
        inputs = torch.cat(
            [
                self.audio_embedding(user),
                self.audio_embedding(previous_audio),
                self.text_embedding(previous_text),
            ],
            -1,
        )
        z, hidden = self.rnn(self.input_dropout(inputs), hidden)
        z = self.output_dropout(z)
        ut, at = self.user_text(z), self.agent_text(z)
        text = at.argmax(-1) if teacher_text is None else teacher_text
        audio = self.audio_head(torch.cat([z, self.text_embedding(text)], -1))
        return audio, ut, at, self.event_head(z), hidden


def shifted(x, first):
    """Shift targets right and insert the initial token."""
    return torch.cat([x.new_tensor([first]), x[:-1]])


def token_text(tokens):
    # Não é CTC: blanks são retirados, repetições NÃO são colapsadas.
    """Decode characters while preserving legitimate repetitions."""
    return "".join(CHARS[int(i)] for i in tokens)


def edit_distance(a, b):
    """Compute sequence edit distance with bounded row storage."""
    old = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, old[j] + 1, old[j - 1] + (x != y)))
        old = new
    return old[-1]


def wer(reference, hypothesis):
    """Compute word error rate with a protected empty denominator."""
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    return edit_distance(ref, hyp) / max(1, len(ref))


def seed(value=7):
    """Set local random seeds for repeatable experiments."""
    torch.manual_seed(value)
    np.random.seed(value)
