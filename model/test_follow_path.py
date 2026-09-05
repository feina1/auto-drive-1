"""Offline model checks: python -m unittest model.test_follow_path."""

import unittest

import torch
from transformers import Dinov2Config, Dinov2Model

from model.follow_path import Model


class ModelTest(unittest.TestCase):
    def test_forward_and_head_only_update(self):
        torch.manual_seed(0)
        # Real DINOv2 implementation, reduced and randomly initialized for testing.
        encoder = Dinov2Model(Dinov2Config(
            hidden_size=32, mlp_ratio=2, num_hidden_layers=1,
            num_attention_heads=2, patch_size=14,
        ))
        model = Model(num_waypoints=3, encoder=encoder).train()
        self.assertFalse(model.encoder.training)
        rgb = torch.rand(2, 3, 36, 64)
        speed = torch.tensor([[10.0], [20.0]], requires_grad=True)
        before = model.head[0].weight.detach().clone()
        optimizer = torch.optim.Adam(model.head.parameters(), lr=1e-3)
        prediction = model(rgb, speed)
        self.assertEqual(prediction.shape, (2, 3, 2))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.head.parameters()))
        self.assertGreater(speed.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.head[0].weight))
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(rgb, speed), model(rgb, speed))
        with self.assertRaises(ValueError):
            model(rgb, speed.squeeze(1))


if __name__ == "__main__":
    unittest.main()
