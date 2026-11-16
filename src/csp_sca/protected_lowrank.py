"""Frozen linear maps with input-subspace-protected low-rank residuals."""

import copy
import math
import re

import torch
from torch import nn
from torch.nn import functional as F


def _math_dtype(dtype):
    return torch.float64 if dtype == torch.float64 else torch.float32


def _validate_basis(basis, input_dim=None):
    if (not isinstance(basis, torch.Tensor) or basis.ndim != 2
            or not basis.is_floating_point() or not torch.isfinite(basis).all()
            or basis.shape[0] < 1 or basis.shape[1] > basis.shape[0]
            or (input_dim is not None and basis.shape[0] != input_dim)):
        raise ValueError('Expected finite floating basis[input_dim, k], with k <= input_dim')
    values = basis.detach().to(dtype=_math_dtype(basis.dtype))
    identity = torch.eye(values.shape[1], device=values.device, dtype=values.dtype)
    tolerance = 1e-10 if basis.dtype == torch.float64 else (
        5e-3 if basis.dtype in (torch.float16, torch.bfloat16) else 1e-5)
    if not torch.allclose(values.T @ values, identity, atol=tolerance, rtol=0):
        raise ValueError('Basis columns must be orthonormal')


def _perpendicular(values, basis):
    if basis.shape[1] == basis.shape[0]:
        return values * 0
    return values - (values @ basis) @ basis.T


def _leaf_linear(base):
    while isinstance(base, ProtectedLowRankLinear):
        base = base.base
    if not isinstance(base, nn.Linear) or not base.weight.is_floating_point():
        raise ValueError('base must be a floating nn.Linear or ProtectedLowRankLinear chain')
    return base


class ProtectedLowRankLinear(nn.Module):
    """Only A/B train; projection is part of the residual on every forward.

    Adapter arithmetic is float32 outside autocast, except float64 base weights
    retain float64 for numerical tests. The frozen base keeps its own dtype.
    """

    def __init__(self, base, basis, rank=8, alpha=8, seed=13):
        super().__init__()
        leaf = _leaf_linear(base)
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError('rank must be a positive integer')
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha must be finite and positive')
        _validate_basis(basis, leaf.in_features)
        self.base = base
        self.base.requires_grad_(False)
        self.base.zero_grad(set_to_none=True)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        dtype, device = _math_dtype(leaf.weight.dtype), leaf.weight.device
        self.register_buffer('basis', basis.detach().to(device=device, dtype=dtype).clone())
        generator = torch.Generator(device='cpu').manual_seed(seed)
        initial_a = torch.empty(rank, leaf.in_features, dtype=dtype)
        nn.init.kaiming_uniform_(initial_a, a=math.sqrt(5), generator=generator)
        self.A = nn.Parameter(initial_a.to(device))
        self.B = nn.Parameter(torch.zeros(leaf.out_features, rank, device=device, dtype=dtype))

    def forward(self, inputs):
        base_output = self.base(inputs)
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            x_perp = _perpendicular(inputs.to(self.A.dtype), self.basis)
            residual = F.linear(F.linear(x_perp, self.A), self.B) * self.scaling
        return base_output + residual.to(base_output.dtype)

    def effective_delta(self):
        """Return the actual scaled, projected additive weight matrix."""
        with torch.autocast(device_type=self.A.device.type, enabled=False):
            return _perpendicular((self.B @ self.A) * self.scaling, self.basis)

    @torch.no_grad()
    def merged_linear(self):
        """Recursively merge a frozen copy for FP32/FP64 numerical checks only."""
        merged = (self.base.merged_linear() if isinstance(self.base, ProtectedLowRankLinear)
                  else copy.deepcopy(self.base))
        merged.weight.add_(self.effective_delta().to(merged.weight.dtype))
        return merged


@torch.no_grad()
def basis_from_class_token_means(means, energy=.9):
    """Protect 90% centered class energy plus every global token mean direction.

    Numerical-zero class variation is measured against the uncentered input
    scale; no arbitrary SVD null vectors are used to invent class information.
    The returned basis matches the input device/dtype. Metadata is JSON-safe.
    """
    if (not isinstance(means, torch.Tensor) or means.ndim != 3
            or min(means.shape) < 1 or not means.is_floating_point()
            or not torch.isfinite(means).all()):
        raise ValueError('Expected finite floating means[C, T, D]')
    if not math.isfinite(energy) or not 0 < energy <= 1:
        raise ValueError('energy must be in (0, 1]')
    values = means.detach().to(dtype=_math_dtype(means.dtype))
    classes, tokens, dimension = values.shape
    token_means = values.mean(dim=0)
    centered = (values - token_means).reshape(-1, dimension)
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    eps = torch.finfo(values.dtype).eps
    zero_tolerance = max(centered.shape) * eps * values.norm()
    numerical_zero = bool(singular[0] <= zero_tolerance)
    centered_energy = float(singular.square().sum())
    if numerical_zero:
        discriminative_rank, retained_energy = 0, 0.
    else:
        fractions = (singular / singular[0]).square()
        fractions /= fractions.sum()
        cumulative = fractions.cumsum(0)
        discriminative_rank = min(int(torch.searchsorted(cumulative,
            cumulative.new_tensor(energy)).item()) + 1, len(singular))
        retained_energy = float(cumulative[discriminative_rank - 1])
    # Normalize token directions so their physical scale cannot erase an
    # already selected unit discriminative direction in the union SVD.
    norms = token_means.norm(dim=1)
    nonzero_tokens = norms > 0
    directions = token_means[nonzero_tokens] / norms[nonzero_tokens, None]
    candidates = torch.cat((vh[:discriminative_rank], directions), dim=0)
    if len(candidates):
        _, union_singular, union_vh = torch.linalg.svd(candidates, full_matrices=False)
        rank_tolerance = max(candidates.shape) * eps * union_singular[0]
        rank = int((union_singular > rank_tolerance).sum())
        basis = union_vh[:rank].T.contiguous()
    else:
        basis = values.new_empty(dimension, 0)
        rank = 0
    basis = basis.to(means.dtype)
    check = basis.to(values.dtype)
    identity = torch.eye(rank, dtype=values.dtype, device=values.device)
    orthogonality = float((check.T @ check - identity).norm())
    token_residual = float(_perpendicular(token_means, check).norm()
                           / token_means.norm().clamp_min(torch.finfo(values.dtype).tiny))
    return basis, {
        'class_count': classes, 'token_count': tokens, 'input_dim': dimension,
        'energy_threshold': energy, 'centered_energy': centered_energy,
        'centered_numerical_zero': numerical_zero,
        'centered_zero_tolerance': float(zero_tolerance),
        'singular_values': singular.cpu().tolist(),
        'discriminative_rank': discriminative_rank, 'retained_energy': retained_energy,
        'rank': rank, 'remaining_dimension': dimension - rank,
        'orthogonality_residual': orthogonality, 'token_mean_residual': token_residual,
    }


@torch.no_grad()
def random_basis_like(basis, seed):
    """Use an independent CPU generator, retaining device, dtype and rank."""
    _validate_basis(basis)
    if basis.shape[1] == 0:
        return torch.empty_like(basis)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    values = torch.randn(basis.shape, dtype=_math_dtype(basis.dtype), generator=generator)
    orthogonal, _ = torch.linalg.qr(values, mode='reduced')
    return orthogonal.to(device=basis.device, dtype=basis.dtype)


def install_qv_adapters(model, bases, mode, rank=8, alpha=8, seed=13):
    """Freeze the full previous chain, then wrap only backbone attention Q/V."""
    if mode not in {'protected', 'unconstrained', 'random'}:
        raise ValueError('mode must be protected, unconstrained, or random')
    attentions = [(name, module) for name, module in model.named_modules()
                  if re.fullmatch(r'backbone\.layers\.\d+\.self_attn', name)]
    if not attentions or set(bases) != {name for name, _ in attentions}:
        raise ValueError('Provide one basis for every backbone.layers.N.self_attn')
    # Validate before mutating any original modules.
    for name, attention in attentions:
        for projection in ('q_proj', 'v_proj'):
            base = getattr(attention, projection, None)
            leaf = _leaf_linear(base)
            _validate_basis(bases[name], leaf.in_features)
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    installed = {}
    for name, attention in attentions:
        basis = bases[name]
        if mode == 'unconstrained':
            basis = basis[:, :0]
        elif mode == 'random':
            layer = int(name.split('.')[2])
            basis = random_basis_like(basis, seed=13000 + seed + layer)
        for projection in ('q_proj', 'v_proj'):
            adapter = ProtectedLowRankLinear(getattr(attention, projection), basis,
                                             rank=rank, alpha=alpha, seed=seed)
            adapter.train(attention.training)
            setattr(attention, projection, adapter)
            installed[f'{name}.{projection}'] = adapter
    return installed
