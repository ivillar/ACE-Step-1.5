"""Smoke tests for the TopK Sparse Autoencoder."""

import torch
import pytest
from ace_step_sae.config import SAEConfig
from ace_step_sae.model import TopKSparseAutoencoder


def make_cfg(**overrides):
    defaults = dict(
        d_model=64, expansion_factor=4, k=8, device="cpu", dtype="float32"
    )
    defaults.update(overrides)
    return SAEConfig(**defaults)


class TestTopKSAE:
    def test_forward_shapes(self):
        cfg = make_cfg()
        sae = TopKSparseAutoencoder(cfg)
        x = torch.randn(16, 64)
        out = sae(x)
        assert out["x_hat"].shape == (16, 64)
        assert out["z"].shape == (16, 256)
        assert out["topk_indices"].shape == (16, 8)

    def test_sparsity(self):
        cfg = make_cfg(k=4)
        sae = TopKSparseAutoencoder(cfg)
        x = torch.randn(8, 64)
        out = sae(x)
        # Each row of z should have exactly k non-zero entries
        nonzero_per_row = (out["z"] != 0).sum(dim=-1)
        assert (nonzero_per_row == 4).all()

    def test_reconstruction_improves(self):
        """After a few training steps, MSE should decrease."""
        cfg = make_cfg()
        sae = TopKSparseAutoencoder(cfg)
        sae.train()
        opt = torch.optim.Adam(sae.parameters(), lr=1e-3)

        data = torch.randn(256, 64)
        initial_loss = sae(data)["mse_loss"].item()

        for _ in range(50):
            out = sae(data)
            out["loss"].backward()
            opt.step()
            opt.zero_grad()

        final_loss = sae(data)["mse_loss"].item()
        assert final_loss < initial_loss

    def test_save_load(self, tmp_path):
        cfg = make_cfg()
        sae = TopKSparseAutoencoder(cfg)
        x = torch.randn(4, 64)
        out1 = sae(x)

        path = str(tmp_path / "test_sae.pt")
        sae.save(path)
        sae2 = TopKSparseAutoencoder.load(path)
        out2 = sae2(x)

        assert torch.allclose(out1["x_hat"], out2["x_hat"], atol=1e-6)

    def test_feature_direction(self):
        cfg = make_cfg()
        sae = TopKSparseAutoencoder(cfg)
        direction = sae.get_feature_direction(0)
        assert direction.shape == (64,)

    def test_dead_feature_tracking(self):
        cfg = make_cfg(dead_feature_threshold=1)
        sae = TopKSparseAutoencoder(cfg)
        sae.train()

        x = torch.randn(32, 64)
        sae(x)

        # Some features should have fired
        assert sae.num_alive > 0

    def test_normalize_decoder(self):
        cfg = make_cfg()
        sae = TopKSparseAutoencoder(cfg)
        sae.normalize_decoder_weights()
        norms = sae.W_dec.data.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
