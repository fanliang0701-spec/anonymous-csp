"""Numerical checks of GPM and O-LoRA update rules."""
import copy
import io
import unittest
import torch
from torch import nn
from csp_sca.baselines import install_olora, olora_regularization, setup_gpm, update_gpm_basis


def tiny(device):
    model = nn.Module()
    model.backbone = nn.Module()
    model.backbone.layers = nn.ModuleList([nn.Module()])
    model.backbone.layers[0].self_attn = nn.Module()
    attention = model.backbone.layers[0].self_attn
    attention.q_proj = nn.Linear(7, 5)
    attention.v_proj = nn.Linear(7, 3)
    model.head = nn.Linear(5, 2)
    return model.double().to(device)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(13)

    def test_olora_frozen_history_penalty_and_sum(self):
        model = tiny(self.device)
        original = copy.deepcopy(model)
        x = torch.randn(11, 7, device=self.device, dtype=torch.float64)
        first = install_olora(model, rank=2, alpha=2, seed=42)
        self.assertEqual(float(olora_regularization(model)), 0.)
        first_q = first['backbone.layers.0.self_attn.q_proj']
        with torch.no_grad():
            first_q.B.normal_()
        first_a = first_q.A.detach().clone()
        first_b = first_q.B.detach().clone()
        second = install_olora(model, rank=2, alpha=2, seed=2026)
        new_q = second['backbone.layers.0.self_attn.q_proj']
        expected = sum((a.base.A @ a.A.T).abs().sum() for a in second.values()) * .5
        torch.testing.assert_close(olora_regularization(model), expected)
        olora_regularization(model).backward()
        self.assertIsNone(first_q.A.grad)
        self.assertIsNotNone(new_q.A.grad)
        self.assertTrue(all(not p.requires_grad for p in new_q.base.parameters()))
        with torch.no_grad():
            new_q.B.normal_()
        base = original.backbone.layers[0].self_attn.q_proj(x)
        wanted = base + x @ first_a.T @ first_b.T + x @ new_q.A.T @ new_q.B.T
        torch.testing.assert_close(new_q(x), wanted)
        # Selected checkpoints contain precisely current branch parameters; restore
        # leaves every older branch and original map byte-identical.
        selected = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0)
        optimizer.zero_grad()
        (new_q(x).square().mean() + olora_regularization(model)).backward()
        optimizer.step()
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(selected[n]) if n in selected else None
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, selected[n] if p.requires_grad else frozen[n], rtol=0, atol=0)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, weights_only=True))

    def test_gpm_actual_sgd_updates_preserve_responses_and_scope(self):
        model = tiny(self.device)
        basis = torch.linalg.qr(torch.randn(7, 3, device=self.device, dtype=torch.float64)).Q
        controller = setup_gpm(model, {'backbone.layers.0.self_attn': basis})
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        self.assertEqual(set(trainable), {'backbone.layers.0.self_attn.q_proj.weight',
                                         'backbone.layers.0.self_attn.v_proj.weight'})
        q = model.backbone.layers[0].self_attn.q_proj
        v = model.backbone.layers[0].self_attn.v_proj
        responses = [a(basis.T).detach().clone() for a in (q, v)]
        old_weights = [a.weight.detach().clone() for a in (q, v)]
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        for _ in range(5):
            optimizer.zero_grad()
            x = torch.randn(9, 7, device=self.device, dtype=torch.float64)
            (q(x).square().mean() + v(x).square().mean()).backward()
            controller.project_gradients()
            optimizer.step()
        for a, response, old in zip((q, v), responses, old_weights):
            torch.testing.assert_close(a(basis.T), response, rtol=0, atol=1e-13)
            self.assertGreater(float((a.weight - old).detach().norm()), 0)

    def test_cumulative_uncentered_memory_retains_old_basis(self):
        device = self.device
        previous = torch.eye(7, device=device, dtype=torch.float64)[:, :2]
        # Existing directions have zero new energy; must still be retained.
        x = torch.zeros(32, 7, device=device, dtype=torch.float64)
        x[:, 2] = torch.arange(32, device=device)
        x[:, 3] = 1
        basis, info = update_gpm_basis(x, previous, energy=.9)
        torch.testing.assert_close(basis @ basis.T @ previous, previous, rtol=0, atol=1e-13)
        self.assertEqual(info['added_rank'], 1)
        self.assertGreaterEqual(info['retained_energy_fraction'], .9)
        full, info = update_gpm_basis(x, torch.eye(7, device=device, dtype=torch.float64), .9)
        self.assertEqual(info['added_rank'], 0)
        self.assertEqual(full.shape, (7, 7))
        zero, info = update_gpm_basis(torch.zeros_like(x), previous, .9)
        self.assertEqual(zero.shape[1], 2)
        # Nonzero constant activation must survive: this is not centered PCA.
        constant = torch.ones_like(x)
        b, info = update_gpm_basis(constant, previous[:, :0], .9)
        self.assertEqual(b.shape[1], 1)
        torch.testing.assert_close(constant @ b @ b.T, constant)
        initial, info = update_gpm_basis(constant, previous[:, :0], .9, initial=True)
        self.assertEqual(initial.shape[1], 0)


if __name__ == '__main__':
    print('Numerical test device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
    unittest.main()
