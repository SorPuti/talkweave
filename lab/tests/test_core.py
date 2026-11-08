import numpy as np
import pytest
import torch
from core import Codec, Duplex, CHARS, aligned_chars, wer, shifted
from protocol import pack, unpack


def test_protocol_and_corruption():
    """Verify protocol and corruption behavior."""
    x = np.linspace(-1, 1, 320, dtype="float32")
    seq, clock, y = unpack(pack(7, 960, x))
    assert (seq, clock) == (7, 960)
    assert np.max(abs(x - y)) < 4e-5
    with pytest.raises(ValueError):
        unpack(pack(0, 0, x)[:-1])


def test_alignment_preserves_double_letters():
    """Verify alignment preserves double letters behavior."""
    chars = aligned_chars([dict(start=0, end=0.4, text="rr")], 20)
    assert sum(chars > 0) == 2
    with pytest.raises(ValueError):
        aligned_chars([dict(start=0, end=0.02, text="olá")], 20)


def test_no_future_leakage():
    """Verify no future leakage behavior."""
    torch.manual_seed(7)
    model = Duplex().eval()
    u = torch.randint(256, (1, 20))
    a = torch.zeros_like(u)
    t = torch.zeros_like(u)
    before = model(u, a, t)[0]
    changed = u.clone()
    changed[:, 10:] = 255
    after = model(changed, a, t)[0]
    torch.testing.assert_close(before[:, :10], after[:, :10])


def test_streaming_equals_sequence():
    """Verify streaming equals sequence behavior."""
    model = Duplex().eval()
    u = torch.randint(256, (1, 10))
    a = torch.zeros_like(u)
    t = torch.zeros_like(u)
    full = model(u, a, t)[0]
    hidden = None
    out = []
    for i in range(10):
        z, *_, hidden = model(u[:, i : i + 1], a[:, i : i + 1], t[:, i : i + 1], hidden)
        out.append(z)
    torch.testing.assert_close(full, torch.cat(out, 1), atol=1e-6, rtol=1e-5)


def test_codec_gradient_and_shape():
    """Verify codec gradient and shape behavior."""
    model = Codec()
    x = torch.randn(8, 320) * 0.1
    y, loss, ids = model(x)
    loss.backward()
    assert y.shape == x.shape and ids.shape == (8,)
    assert torch.isfinite(loss) and model.enc[0].weight.grad.abs().sum() > 0


def test_wer_and_shift():
    """Verify wer and shift behavior."""
    assert wer("olá mundo", "olá") == 0.5
    assert shifted(torch.tensor([3, 4, 5]), 9).tolist() == [9, 3, 4]


def test_dropout_training_and_inference():
    """Keep dropout stochastic in training and transparent in inference."""
    torch.manual_seed(7)
    baseline = Duplex(dropout=0.0).eval()
    regularized = Duplex(dropout=0.3)
    regularized.load_state_dict(baseline.state_dict())
    user = torch.randint(256, (1, 30))
    previous = torch.zeros_like(user)
    regularized.train()
    first = regularized(user, previous, previous)[0]
    second = regularized(user, previous, previous)[0]
    assert not torch.equal(first, second)
    regularized.eval()
    torch.testing.assert_close(
        regularized(user, previous, previous)[0], baseline(user, previous, previous)[0]
    )
    torch.testing.assert_close(
        regularized(user, previous, previous)[0],
        regularized(user, previous, previous)[0],
    )
    with pytest.raises(ValueError):
        Duplex(dropout=1.0)
