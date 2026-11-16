"""Numerical checks of fixed-factor InfLoRA updates."""
import copy
import importlib.util
import unittest
from pathlib import Path
import torch
from torch import nn

class InfLoRATests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('csp_sca.inflora'), 'InfLoRA implementation is required')
        from csp_sca import inflora
        self.m = inflora
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(13)

    def test_designed_down_is_fixed_complement_and_only_up_trains(self):
        d = 9
        old = torch.eye(d, dtype=torch.float64, device=self.device)[:, :2]
        x = torch.randn(50, d, dtype=torch.float64, device=self.device)
        covariance = x.T @ x / len(x)
        a, info = self.m.design_down(covariance, old, rank=3)
        torch.testing.assert_close(a @ old, torch.zeros(3, 2, dtype=a.dtype, device=a.device), atol=1e-12, rtol=0)
        torch.testing.assert_close(a @ a.T, torch.eye(3, dtype=a.dtype, device=a.device)/3, atol=1e-12, rtol=0)
        base = nn.Linear(d, 5).double().to(self.device)
        initial = copy.deepcopy(base)
        adapter = self.m.InfLoRALinear(base, a)
        torch.testing.assert_close(adapter(x), initial(x), atol=0, rtol=0)
        frozen_a = adapter.A.detach().clone()
        optimizer = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=.01, weight_decay=0)
        for _ in range(3):
            optimizer.zero_grad()
            adapter(x).square().mean().backward()
            optimizer.step()
        self.assertEqual([name for name,p in adapter.named_parameters() if p.requires_grad], ['B'])
        torch.testing.assert_close(adapter.A, frozen_a, atol=0, rtol=0)
        torch.testing.assert_close(adapter(old.T), initial(old.T), atol=1e-12, rtol=0)
        self.assertGreater(float(adapter.B.detach().norm()), 0)
        for name, value in initial.state_dict().items():
            torch.testing.assert_close(base.state_dict()[name], value, atol=0, rtol=0)

    def test_covariance_memory_preserves_old_even_without_current_energy(self):
        old = torch.eye(9, dtype=torch.float64, device=self.device)[:, :2]
        covariance = torch.zeros(9, 9, dtype=torch.float64, device=self.device)
        covariance[2,2], covariance[3,3] = 10, 1
        bank, info = self.m.update_memory(covariance, old, threshold=.95)
        self.assertEqual(info['added_rank'], 1)
        torch.testing.assert_close(bank @ bank.T @ old, old, atol=1e-12, rtol=0)
        self.assertGreaterEqual(info['retained_covariance_energy_fraction'], .95)
        first, stats = self.m.update_memory(covariance, old[:, :0], threshold=.95, initial=True)
        self.assertEqual(first.shape[1], 1)  # official max(count(cumsum<threshold),1)
        zero, stats = self.m.update_memory(torch.zeros_like(covariance), old, threshold=.95)
        torch.testing.assert_close(zero @ zero.T, old @ old.T, atol=1e-12, rtol=0)

    def test_all_tokens_covariance_keeps_variation_cancelled_by_class_means(self):
        x = torch.tensor([[1., 2.], [-1., -2.]], dtype=torch.float64, device=self.device)
        got = self.m.token_covariance([x[:1], x[1:]])
        torch.testing.assert_close(got, x.T @ x / 2)
        self.assertGreater(float(got.norm()), 0.)
        self.assertEqual(float(x.mean(0).norm()), 0.)

    def test_two_increment_checkpoint_reconstruction_and_frozen_history(self):
        initial = nn.Linear(9, 5).double().to(self.device)
        x = torch.randn(15, 9, dtype=torch.float64, device=self.device)
        designs = [torch.linalg.qr(torch.randn(9, 3, dtype=torch.float64, device=self.device)).Q.T / 3**.5 for _ in range(2)]
        model, saved = copy.deepcopy(initial), []
        for a in designs:
            model = self.m.InfLoRALinear(model, a)
            with torch.no_grad(): model.B.normal_()
            saved.append(model.B.detach().clone())
        replay = copy.deepcopy(initial)
        for a,b in zip(designs, saved):
            replay = self.m.InfLoRALinear(replay, a)
            with torch.no_grad(): replay.B.copy_(b)
        torch.testing.assert_close(model(x), replay(x), atol=0, rtol=0)
        self.assertFalse(model.base.B.requires_grad)
        self.assertTrue(model.B.requires_grad)

if __name__ == '__main__':
    print('Numerical test device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
    unittest.main()
