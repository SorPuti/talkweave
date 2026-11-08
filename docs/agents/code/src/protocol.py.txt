"""Implement the protocol component of the CPU duplex tutorial."""

import struct
import numpy as np

HEADER = struct.Struct("<BQQH")
VERSION = 1


def pack(sequence, sample_clock, samples):
    """Serialize a validated PCM frame with its session clock."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 1 or len(x) > 65535 or not np.isfinite(x).all():
        raise ValueError("Bloco inválido")
    pcm = (np.clip(x, -1, 1) * 32767).round().astype("<i2")
    return HEADER.pack(VERSION, sequence, sample_clock, len(x)) + pcm.tobytes()


def unpack(packet):
    """Validate and decode one binary PCM frame."""
    if len(packet) < HEADER.size:
        raise ValueError("Cabeçalho truncado")
    version, seq, clock, n = HEADER.unpack_from(packet)
    if version != VERSION or len(packet) != HEADER.size + 2 * n:
        raise ValueError("Pacote inválido")
    return (
        seq,
        clock,
        np.frombuffer(packet, dtype="<i2", offset=HEADER.size).astype(np.float32)
        / 32767,
    )
