"""Smoke tests for model forward passes — shape and no-crash checks on CPU."""

import pytest
import torch
import torch.nn as nn

# Fixtures
BATCH = 2
EMBED_DIM = 64  # Small to keep tests fast
ACOUSTIC_INPUT = (BATCH, 3, 32, 256)   # (B, C, pings, depth)
FLAT_ACOUSTIC = (BATCH, 32 * 256 * 3)  # Flattened: 24576
VISUAL_INPUT = (BATCH, 3, 224, 224)


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# ConvEncoder
# ---------------------------------------------------------------------------

class TestConvEncoder:
    def test_output_shape(self, device):
        from models.acoustic import ConvEncoder
        model = ConvEncoder(embed_dim=EMBED_DIM).to(device)
        x = torch.randn(*ACOUSTIC_INPUT)
        out = model(x)
        assert out.shape == (BATCH, EMBED_DIM)

    def test_no_nan_output(self, device):
        from models.acoustic import ConvEncoder
        model = ConvEncoder(embed_dim=EMBED_DIM).to(device)
        x = torch.randn(*ACOUSTIC_INPUT)
        out = model(x)
        assert not torch.isnan(out).any()

    def test_batch_size_one(self, device):
        from models.acoustic import ConvEncoder
        # eval mode avoids BatchNorm1d error with batch=1
        model = ConvEncoder(embed_dim=EMBED_DIM).eval().to(device)
        x = torch.randn(1, 3, 32, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, EMBED_DIM)


# ---------------------------------------------------------------------------
# Embedder (LeWM core)
# ---------------------------------------------------------------------------

class TestEmbedder:
    def test_output_shape(self, device):
        from models.lewm import Embedder
        model = Embedder(embed_dim=EMBED_DIM).to(device)
        x = torch.randn(*FLAT_ACOUSTIC)
        out = model(x)
        assert out.shape == (BATCH, 32, EMBED_DIM)

    def test_no_nan_output(self, device):
        from models.lewm import Embedder
        model = Embedder(embed_dim=EMBED_DIM).to(device)
        x = torch.randn(*FLAT_ACOUSTIC)
        out = model(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# SIGReg
# ---------------------------------------------------------------------------

class TestSIGReg:
    def test_returns_scalar_loss(self, device):
        from models.lewm import SIGReg
        model = SIGReg(embed_dim=EMBED_DIM).to(device)
        embeddings = torch.randn(BATCH, 32, EMBED_DIM)
        loss = model(embeddings)
        assert loss.ndim == 0  # scalar
        assert not torch.isnan(loss)

    def test_loss_non_negative(self, device):
        from models.lewm import SIGReg
        model = SIGReg(embed_dim=EMBED_DIM).to(device)
        embeddings = torch.randn(BATCH, 32, EMBED_DIM)
        loss = model(embeddings)
        assert loss.item() >= 0.0


# ---------------------------------------------------------------------------
# CrossModalJEPA
# ---------------------------------------------------------------------------

class TestCrossModalJEPA:
    def _make_model(self, task="presence"):
        from models.acoustic import ConvEncoder
        from models.jepa import CrossModalJEPA
        encoder = ConvEncoder(embed_dim=EMBED_DIM)
        return CrossModalJEPA(encoder, embed_dim=EMBED_DIM, task=task)

    def test_presence_output_shape(self, device):
        model = self._make_model("presence").to(device)
        model.eval()
        acoustic = torch.randn(*ACOUSTIC_INPUT)
        visual = torch.randn(*VISUAL_INPUT)
        with torch.no_grad():
            _, _, logits = model(visual, acoustic)
        assert logits.shape == (BATCH, 4)

    def test_counting_output_shape(self, device):
        model = self._make_model("counting").to(device)
        model.eval()
        acoustic = torch.randn(*ACOUSTIC_INPUT)
        visual = torch.randn(*VISUAL_INPUT)
        with torch.no_grad():
            _, _, logits = model(visual, acoustic)
        assert logits.shape == (BATCH, 4)

    def test_no_nan_output(self, device):
        model = self._make_model("presence").to(device)
        model.eval()
        acoustic = torch.randn(*ACOUSTIC_INPUT)
        visual = torch.randn(*VISUAL_INPUT)
        with torch.no_grad():
            _, _, logits = model(visual, acoustic)
        assert not torch.isnan(logits).any()


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class TestFocalLoss:
    def test_returns_scalar(self):
        from models.jepa import FocalLoss
        loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
        logits = torch.randn(4, 4)
        targets = torch.randint(0, 4, (4,))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_non_negative(self):
        from models.jepa import FocalLoss
        loss_fn = FocalLoss()
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0.0

    def test_sum_reduction(self):
        from models.jepa import FocalLoss
        loss_mean = FocalLoss(reduction="mean")(torch.randn(4, 4), torch.randint(0, 4, (4,)))
        loss_sum = FocalLoss(reduction="sum")(torch.randn(4, 4), torch.randint(0, 4, (4,)))
        # Sum should be >= mean for batch > 1 (sum = mean * batch_size roughly)
        assert loss_sum.item() >= 0.0
        assert loss_mean.item() >= 0.0

    def test_gamma_zero_equals_cross_entropy(self):
        from models.jepa import FocalLoss
        torch.manual_seed(0)
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        focal = FocalLoss(alpha=1.0, gamma=0.0)(logits, targets)
        ce = nn.CrossEntropyLoss()(logits, targets)
        assert abs(focal.item() - ce.item()) < 1e-4


# ---------------------------------------------------------------------------
# AcousticAST
# ---------------------------------------------------------------------------

class TestAcousticAST:
    def test_output_shape_flat_input(self, device):
        from models.ast import AcousticAST
        model = AcousticAST(embed_dim=EMBED_DIM, num_classes=4).to(device)
        model.eval()
        # Flat input: B × (3 × 32 × 256) = B × 24576
        x = torch.randn(BATCH, 3 * 32 * 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH, 4)

    def test_no_nan_output(self, device):
        from models.ast import AcousticAST
        model = AcousticAST(embed_dim=EMBED_DIM, num_classes=4).to(device)
        model.eval()
        x = torch.randn(BATCH, 3 * 32 * 256)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()
