"""Real small-matrix checks; no production model or dataset is loaded."""

import pytest
import torch
from torch import nn

from csp_sca.protected_lowrank import (
    ProtectedLowRankLinear,
    basis_from_class_token_means,
    install_qv_adapters,
    random_basis_like,
)


@pytest.mark.parametrize("optimizer_cls", [torch.optim.SGD, torch.optim.AdamW])
@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-10), (torch.float32, 1e-5)])
def test_training_learns_orthogonal_input_and_preserves_old_response(optimizer_cls, dtype, tolerance):
    base = nn.Linear(2, 1, bias=True, dtype=dtype)
    with torch.no_grad():
        base.weight.copy_(torch.tensor([[0.3, -0.2]], dtype=dtype))
        base.bias.fill_(0.1)
    originals = {name: p.detach().clone() for name, p in base.named_parameters()}
    basis = torch.tensor([[1.], [0.]], dtype=dtype)
    adapter = ProtectedLowRankLinear(base, basis, rank=2, alpha=2)
    inputs = torch.eye(2, dtype=dtype)
    before = base(inputs).detach().clone()
    assert torch.equal(adapter(inputs), before)
    optimizer = optimizer_cls(adapter.parameters(), lr=0.1, weight_decay=0.02)
    old_a = adapter.A.detach().clone()
    target = before + torch.tensor([[3.], [1.]], dtype=dtype)
    for step in range(120):
        optimizer.zero_grad()
        (adapter(inputs) - target).square().mean().backward()
        if step == 0:
            assert adapter.B.grad.abs().sum() > 0
            assert torch.count_nonzero(adapter.A.grad) == 0
        optimizer.step()
        delta = adapter.effective_delta()
        residual = torch.linalg.vector_norm(delta @ basis) / delta.norm().clamp_min(1e-15)
        assert residual < tolerance
    after = adapter(inputs).detach()
    torch.testing.assert_close(after[0], before[0], atol=tolerance, rtol=0)
    assert after[1].item() > before[1].item() + 0.8
    assert not torch.equal(adapter.A, old_a)
    for name, value in base.named_parameters():
        assert not value.requires_grad
        assert value.grad is None
        assert torch.equal(value, originals[name])
    torch.testing.assert_close(adapter.merged_linear()(inputs), after, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16])
def test_dtype_zero_residual_local_rng_and_ordinary_lowrank(dtype):
    base = nn.Linear(3, 2, dtype=dtype)
    rng = torch.random.get_rng_state().clone()
    adapter = ProtectedLowRankLinear(base, torch.empty(3, 0), rank=2, alpha=4, seed=24)
    assert torch.equal(torch.random.get_rng_state(), rng)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    assert adapter.A.dtype == adapter.B.dtype == adapter.basis.dtype == expected_dtype
    x = torch.tensor([[1., 2., -1.], [0.3, -1., 2.]], dtype=dtype)
    assert torch.equal(adapter(x), base(x))
    with torch.no_grad():
        adapter.A.copy_(torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=expected_dtype))
        adapter.B.copy_(torch.tensor([[1., 0.], [0., 1.]], dtype=expected_dtype))
    torch.testing.assert_close(adapter.effective_delta(), torch.tensor([[2., 0., 0.], [0., 2., 0.]], dtype=expected_dtype))
    expected = base(x) + (2 * x[:, :2]).to(dtype)
    torch.testing.assert_close(adapter(x), expected)
    if dtype in (torch.float64, torch.float32):
        torch.testing.assert_close(adapter.merged_linear()(x), adapter(x))


def test_autocast_keeps_adapter_projection_and_factors_in_float32():
    adapter = ProtectedLowRankLinear(nn.Linear(3, 2), torch.eye(3)[:, :1], rank=2)
    with torch.no_grad():
        adapter.B.fill_(0.25)
    x = torch.tensor([[1., 0.3, -0.8]])
    residual = ((x - x[:, :1] @ torch.tensor([[1., 0., 0.]])) @ adapter.A.T @ adapter.B.T) * 4
    with torch.autocast("cpu", dtype=torch.bfloat16):
        base_output = adapter.base(x)
        actual = adapter(x)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, base_output + residual.to(torch.bfloat16))


def test_full_basis_has_zero_effective_update():
    adapter = ProtectedLowRankLinear(nn.Linear(3, 2, dtype=torch.float64), torch.eye(3, dtype=torch.float64))
    with torch.no_grad():
        adapter.B.fill_(5)
    x = torch.tensor([[1., -2., 3.]], dtype=torch.float64)
    assert torch.equal(adapter(x), adapter.base(x))
    assert torch.count_nonzero(adapter.effective_delta()) == 0


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-10), (torch.float32, 1e-4)])
@pytest.mark.parametrize("optimizer_cls", [torch.optim.SGD, torch.optim.AdamW])
def test_rotated_basis_stays_protected_after_optimizer_updates(dtype, tolerance, optimizer_cls):
    basis = random_basis_like(torch.eye(6, dtype=dtype)[:, :3], seed=103)
    adapter = ProtectedLowRankLinear(nn.Linear(6, 4, dtype=dtype), basis, rank=3, alpha=3)
    generator = torch.Generator().manual_seed(104)
    x = torch.randn(2, 3, 6, dtype=dtype, generator=generator)
    target = torch.randn(2, 3, 4, dtype=dtype, generator=generator)
    old_x = basis.T
    old_response = adapter.base(old_x).detach()
    optimizer = optimizer_cls([adapter.A, adapter.B], lr=.05, weight_decay=.1)
    for _ in range(20):
        optimizer.zero_grad()
        (adapter(x) - target).square().mean().backward()
        optimizer.step()
        delta = adapter.effective_delta()
        assert float(((delta @ basis).norm() / delta.norm()).detach()) < tolerance
    torch.testing.assert_close(adapter(old_x), old_response, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(adapter.merged_linear()(x), adapter(x), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16])
def test_fitted_basis_preserves_dtype_and_token_mean(dtype):
    means = torch.tensor([[[1., 0., 3.]], [[-1., 0., 3.]]], dtype=dtype)
    basis, meta = basis_from_class_token_means(means)
    assert basis.dtype == dtype
    assert basis.device == means.device
    assert meta["rank"] == 2
    assert meta["remaining_dimension"] == 1
    assert meta["token_mean_residual"] < 1e-5


@pytest.mark.parametrize("basis", [torch.zeros(2, 1), torch.ones(2, 2), torch.eye(3),
    torch.tensor([[float("nan")], [0.]]), torch.tensor([[float("inf")], [0.]]),
    torch.ones(2), torch.empty(2, 3), torch.ones(2, 1, dtype=torch.long)])
def test_rejects_invalid_bases(basis):
    with pytest.raises(ValueError):
        ProtectedLowRankLinear(nn.Linear(2, 1), basis)


def test_basis_retains_squared_energy_and_global_token_mean_directions():
    means = torch.tensor([[[2., 0., 3., 0.], [2., 0., 0., 4.]],
                          [[-2., 0., 3., 0.], [-2., 0., 0., 4.]],
                          [[0., 1., 3., 0.], [0., 1., 0., 4.]],
                          [[0., -1., 3., 0.], [0., -1., 0., 4.]]], dtype=torch.float64)
    basis, meta = basis_from_class_token_means(means)
    # Discriminative energy is 16 + 4; 90% requires both axes, plus both token means.
    torch.testing.assert_close(basis @ basis.T, torch.eye(4, dtype=torch.float64))
    assert meta["discriminative_rank"] == 2
    assert meta["rank"] == 4
    assert meta["retained_energy"] == pytest.approx(1.)
    assert meta["centered_energy"] == pytest.approx(20.)
    assert not meta["centered_numerical_zero"]
    assert meta["orthogonality_residual"] < 1e-10
    one, one_meta = basis_from_class_token_means(means, energy=0.75)
    torch.testing.assert_close(one @ one.T, torch.diag(torch.tensor([1., 0., 1., 1.], dtype=torch.float64)))
    assert one_meta["discriminative_rank"] == 1
    assert one_meta["retained_energy"] == pytest.approx(0.8)


def test_zero_centered_energy_does_not_invent_discriminative_directions():
    means = torch.tensor([[[0.1, 0., 0.], [0., 0.3, 0.]]] * 3, dtype=torch.float64)
    basis, meta = basis_from_class_token_means(means)
    assert meta["centered_numerical_zero"]
    assert meta["discriminative_rank"] == 0
    assert meta["retained_energy"] == 0.
    torch.testing.assert_close(basis @ basis.T, torch.diag(torch.tensor([1., 1., 0.], dtype=torch.float64)))
    empty, zero = basis_from_class_token_means(torch.zeros(3, 2, 4))
    assert empty.shape == (4, 0)
    assert zero["centered_numerical_zero"]


@pytest.mark.parametrize("means,energy", [(torch.zeros(2, 3), .9), (torch.zeros(0, 2, 3), .9),
    (torch.ones(2, 3, 4, dtype=torch.long), .9), (torch.full((2, 3, 4), float("nan")), .9),
    (torch.zeros(2, 3, 4), 0), (torch.zeros(2, 3, 4), 1.1)])
def test_rejects_invalid_class_token_means(means, energy):
    with pytest.raises(ValueError):
        basis_from_class_token_means(means, energy)


def test_random_basis_preserves_rank_dtype_and_rng_without_reusing_real_axes():
    basis = torch.eye(5, dtype=torch.float64)[:, :2]
    rng = torch.random.get_rng_state().clone()
    random = random_basis_like(basis, 13013)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert random.dtype == basis.dtype
    assert random.shape == basis.shape
    torch.testing.assert_close(random.T @ random, torch.eye(2, dtype=torch.float64))
    assert torch.equal(random, random_basis_like(basis, 13013))
    assert not torch.allclose(random @ random.T, basis @ basis.T)
    assert not torch.equal(random, random_basis_like(basis, 13014))
    assert random_basis_like(torch.empty(5, 0), 13).shape == (5, 0)


def qwen_shaped_model():
    model = nn.Module()
    model.frontend = nn.Linear(4, 4)
    model.head = nn.Linear(4, 3)
    model.backbone = nn.Module()
    model.backbone.layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(layer.self_attn, name, nn.Linear(4, 4))
        layer.norm = nn.LayerNorm(4)
        model.backbone.layers.append(layer)
    return model


@pytest.mark.parametrize("mode", ["protected", "unconstrained", "random"])
def test_installer_replaces_only_qv_and_freezes_every_original_parameter(mode):
    model = qwen_shaped_model()
    original = dict(model.named_parameters())
    bases = {f"backbone.layers.{i}.self_attn": torch.eye(4)[:, :2] for i in range(2)}
    rng = torch.random.get_rng_state().clone()
    installed = install_qv_adapters(model, bases, mode)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert set(installed) == {f"{key}.{name}" for key in bases for name in ("q_proj", "v_proj")}
    assert all(not p.requires_grad for p in original.values())
    assert {name for name, p in model.named_parameters() if p.requires_grad} == {
        f"{name}.{factor}" for name in installed for factor in ("A", "B")}
    for i in range(2):
        name = f"backbone.layers.{i}.self_attn"
        q, v = installed[name + ".q_proj"], installed[name + ".v_proj"]
        assert torch.equal(q.A, v.A)
        assert torch.equal(q.basis, v.basis)
        if mode == "protected":
            torch.testing.assert_close(q.basis, bases[name])
        elif mode == "unconstrained":
            assert q.basis.shape == (4, 0)
        else:
            torch.testing.assert_close(q.basis, random_basis_like(bases[name], 13013 + i))


def test_installer_rejects_missing_basis_and_unknown_mode():
    model = qwen_shaped_model()
    with pytest.raises(ValueError):
        install_qv_adapters(model, {}, "protected")
    with pytest.raises(ValueError):
        install_qv_adapters(model, {}, "typo")
