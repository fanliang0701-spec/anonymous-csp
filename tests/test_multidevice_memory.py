"""CPU checks of real stacked residuals and cumulative protected subspaces."""

import copy
import json

import pytest
import torch
from torch import nn

from csp_sca.protected_lowrank import ProtectedLowRankLinear, install_qv_adapters
from csp_sca.multidevice_memory import union_basis, union_bank


@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-10), (torch.float32, 1e-5)])
def test_three_stages_learn_new_axes_and_freeze_the_entire_previous_chain(dtype, tolerance):
    model = nn.Linear(3, 1, dtype=dtype)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    inputs = torch.eye(3, dtype=dtype)
    for stage in range(3):
        previous = model
        old_state = copy.deepcopy(previous.state_dict())
        before = previous(inputs).detach().clone()
        model = ProtectedLowRankLinear(previous, inputs[:, :stage], rank=1, alpha=1, seed=13 + stage)
        assert torch.equal(model(inputs), before)
        assert {name for name, p in model.named_parameters() if p.requires_grad} == {"A", "B"}
        assert all(p.grad is None for p in previous.parameters())
        initial_a = model.A.detach().clone()
        target = before.clone()
        target[stage] += stage + 1
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        for _ in range(500):
            optimizer.zero_grad()
            (model(inputs) - target).square().mean().backward()
            optimizer.step()
        torch.testing.assert_close(model(inputs), target, atol=2e-4, rtol=0)
        torch.testing.assert_close(model(inputs)[:stage], before[:stage], atol=tolerance, rtol=0)
        assert not torch.equal(model.A, initial_a)
        for name, value in previous.state_dict().items():
            assert torch.equal(value, old_state[name]), name
        assert all(not p.requires_grad and p.grad is None for p in previous.parameters())
    torch.testing.assert_close(model(inputs).flatten(), inputs.new_tensor([1., 2., 3.]), atol=5e-4, rtol=0)
    # Restoring the same chain must retain branch execution order and values.
    restored = nn.Linear(3, 1, dtype=dtype)
    for stage in range(3):
        restored = ProtectedLowRankLinear(restored, inputs[:, :stage], rank=1, alpha=1, seed=99)
    restored.load_state_dict(model.state_dict())
    assert torch.equal(restored(inputs), model(inputs))
    state = copy.deepcopy(model.state_dict())
    merged = model.merged_linear()
    assert isinstance(merged, nn.Linear)
    assert all(not p.requires_grad for p in merged.parameters())
    torch.testing.assert_close(merged(inputs), model(inputs), atol=tolerance, rtol=tolerance)
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name]), name


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16, torch.float16])
def test_zero_initialized_nested_branch_preserves_previous_forward_exactly(dtype):
    base = nn.Linear(4, 2, dtype=dtype)
    old = ProtectedLowRankLinear(base, torch.empty(4, 0), rank=2, alpha=2)
    with torch.no_grad():
        old.B.fill_(.25)
    x = torch.tensor([[1., -.3, .5, -2.], [.1, 3., -.2, .8]], dtype=dtype)
    expected = old(x).detach()
    rng = torch.random.get_rng_state().clone()
    new = ProtectedLowRankLinear(old, torch.eye(4)[:, :1], rank=2, alpha=2, seed=1013)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert new.A.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert new.A.device == base.weight.device
    assert torch.equal(new(x), expected)
    if dtype == torch.float32:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert torch.equal(new(x), old(x))


class TinyQvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layers = nn.ModuleList([nn.Module()])
        attention = nn.Module()
        attention.q_proj = nn.Linear(4, 3)
        attention.v_proj = nn.Linear(4, 2)
        self.backbone.layers[0].self_attn = attention
        self.head = nn.Linear(3, 1)

    def forward(self, x):
        attention = self.backbone.layers[0].self_attn
        return attention.q_proj(x), attention.v_proj(x)


@pytest.mark.parametrize("mode", ["protected", "unconstrained", "random"])
def test_installer_stacks_only_qv_and_freezes_old_increments(mode):
    model = TinyQvModel()
    key = "backbone.layers.0.self_attn"
    bases = {key: torch.eye(4)[:, :1]}
    old = install_qv_adapters(model, bases, mode, rank=2, alpha=2)
    with torch.no_grad():
        for adapter in old.values():
            adapter.B.fill_(.3)
    x = torch.tensor([[1., .5, -.2, 2.]])
    before = tuple(value.detach().clone() for value in model(x))
    previous = copy.deepcopy(model.state_dict())
    original_parameters = list(model.parameters())
    installed = install_qv_adapters(model, bases, mode, rank=2, alpha=2, seed=1013)
    assert all(not p.requires_grad for p in original_parameters)
    assert {name for name, p in model.named_parameters() if p.requires_grad} == {
        f"{name}.{factor}" for name in installed for factor in ("A", "B")}
    assert all(torch.equal(value, expected) for value, expected in zip(model(x), before))
    for name, adapter in installed.items():
        assert adapter.base is old[name]
        assert adapter.B.shape == old[name].B.shape
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    optimizer.zero_grad()
    sum(value.square().mean() for value in model(x)).backward()
    optimizer.step()
    for name, value in previous.items():
        parts = name.split(".")
        if "q_proj" in parts or "v_proj" in parts:
            index = parts.index("q_proj") if "q_proj" in parts else parts.index("v_proj")
            parts.insert(index + 1, "base")
        assert torch.equal(model.state_dict()[".".join(parts)], value), name


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_union_preserves_old_axes_adds_only_new_information_and_is_json_safe(dtype):
    axes = torch.eye(4, dtype=dtype)
    old, new = axes[:, :1].clone(), axes[:, :2].clone()
    basis, metadata = union_basis(old, new)
    assert basis.dtype == dtype and basis.device == old.device
    torch.testing.assert_close(basis @ basis.T, torch.diag(axes.new_tensor([1., 1., 0., 0.])))
    assert metadata["old_rank"] == 1
    assert metadata["new_rank"] == 2
    assert metadata["added_rank"] == 1
    assert metadata["rank"] == 2
    assert metadata["remaining_dimension"] == 2
    assert not metadata["capacity_exhausted"]
    assert metadata["old_span_residual"] <= 1e-5
    assert torch.equal(old, axes[:, :1])
    assert torch.equal(new, axes[:, :2])
    json.dumps(metadata, allow_nan=False)
    # The result must pass the actual installer, not just a looser test check.
    model = TinyQvModel()
    install_qv_adapters(model, {"backbone.layers.0.self_attn": basis}, "protected")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("relative_size,expected_added", [(.25, 0), (4., 1)])
def test_union_uses_storage_precision_threshold_only_on_new_residual(dtype, relative_size, expected_added):
    old = torch.eye(8, dtype=dtype)[:, :1]
    # Unit new direction nearly coincides with the old first axis.
    tolerance = 8 * torch.finfo(dtype).eps
    new = old.clone()
    new[1] = tolerance * relative_size
    new /= new.norm()
    basis, metadata = union_basis(old, new)
    assert metadata["rank_tolerance"] == pytest.approx(tolerance, rel=1e-12)
    assert metadata["added_rank"] == expected_added
    expected = torch.zeros(8, dtype=dtype)
    expected[:1 + expected_added] = 1
    torch.testing.assert_close(basis @ basis.T, torch.diag(expected))
    assert metadata["old_span_residual"] <= 1e-5


@pytest.mark.parametrize("old_rank,new_rank,expected", [(0, 0, 0), (0, 2, 2), (2, 0, 2), (4, 1, 4), (4, 4, 4)])
def test_union_supports_empty_spaces_and_keeps_full_capacity(old_rank, new_rank, expected):
    axes = torch.eye(4)
    basis, metadata = union_basis(axes[:, :old_rank], axes[:, :new_rank])
    assert basis.shape == (4, expected)
    torch.testing.assert_close(basis @ basis.T, torch.diag(axes.new_tensor([1.] * expected + [0.] * (4 - expected))))
    assert metadata["rank"] >= old_rank
    assert metadata["remaining_dimension"] == 4 - expected
    assert metadata["capacity_exhausted"] == (expected == 4)
    assert metadata["old_span_residual"] <= 1e-5
    assert metadata["orthogonality_residual"] <= 1e-5


def test_union_has_no_artificial_rank_cap():
    axes = torch.eye(300)
    basis, metadata = union_basis(axes[:, :257], axes[:, [0, 258, 299]])
    assert basis.shape == (300, 259)
    assert metadata["added_rank"] == 2
    torch.testing.assert_close(basis @ basis.T @ axes[:, :257], axes[:, :257])
    torch.testing.assert_close(basis @ basis.T @ axes[:, [258, 299]], axes[:, [258, 299]])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotated_cumulative_union_never_loses_old_span_or_creates_duplicate_axes(dtype):
    generator = torch.Generator().manual_seed(13)
    axes, _ = torch.linalg.qr(torch.randn(12, 12, dtype=torch.float64, generator=generator))
    axes = axes.to(dtype)
    bank = axes[:, :0]
    for stage in range(6):
        old = bank.clone()
        new = axes[:, :2 * (stage + 1)]
        bank, metadata = union_basis(bank, new)
        assert bank.shape[1] == 2 * (stage + 1)
        assert metadata["added_rank"] == 2
        assert metadata["old_span_residual"] <= 1e-5
        torch.testing.assert_close(bank.double() @ bank.double().T @ old.double(), old.double(), atol=1e-6, rtol=0)
    again, metadata = union_basis(bank, axes)
    assert metadata["added_rank"] == 0
    assert metadata["capacity_exhausted"]
    torch.testing.assert_close(again @ again.T, torch.eye(12, dtype=dtype))


@pytest.mark.parametrize("bad", [
    None, torch.eye(3), torch.zeros(4, 1), torch.ones(4, 2), torch.ones(4),
    torch.empty(4, 5), torch.empty(0, 0), torch.eye(4, dtype=torch.float64),
    torch.eye(4, dtype=torch.float16), torch.eye(4, dtype=torch.bfloat16),
    torch.eye(4, dtype=torch.long), torch.full((4, 1), float("nan")),
    torch.full((4, 1), float("inf")),
])
@pytest.mark.parametrize("bad_is_old", [True, False])
def test_union_rejects_invalid_mismatched_or_nonorthonormal_inputs(bad, bad_is_old):
    valid = torch.eye(4)[:, :1]
    with pytest.raises(ValueError):
        union_basis(bad, valid) if bad_is_old else union_basis(valid, bad)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_union_rejects_low_precision_storage_even_when_dtypes_match(dtype):
    basis = torch.eye(4, dtype=dtype)
    with pytest.raises(ValueError, match="float32.*float64"):
        union_basis(basis, basis)


def test_union_rejects_device_mismatch_without_loading_gpu():
    with pytest.raises(ValueError, match="device"):
        union_basis(torch.eye(4), torch.empty(4, 0, device="meta"))


def test_union_bank_matches_layers_and_retains_per_layer_statistics():
    old = {"layer0": torch.eye(4)[:, :1], "layer1": torch.eye(3)}
    new = {"layer0": torch.eye(4)[:, 1:2], "layer1": torch.eye(3)[:, :1]}
    bases, stats = union_bank(old, new)
    assert set(bases) == set(stats) == {"layer0", "layer1"}
    assert bases["layer0"].shape == (4, 2)
    assert bases["layer1"].shape == (3, 3)
    assert stats["layer0"]["added_rank"] == 1
    assert stats["layer1"]["capacity_exhausted"]
    json.dumps(stats, allow_nan=False)
    assert union_bank({}, {}) == ({}, {})


@pytest.mark.parametrize("old,new", [({}, {"layer": torch.eye(2)}), ({"layer": torch.eye(2)}, {}), (None, {}), ({}, [])])
def test_union_bank_rejects_nonmatching_keys_or_non_dictionary_inputs(old, new):
    with pytest.raises(ValueError):
        union_bank(old, new)
