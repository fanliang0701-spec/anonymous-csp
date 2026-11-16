"""Cumulative protected input spaces; old directions are never truncated."""

import torch

from .protected_lowrank import _validate_basis


@torch.no_grad()
def union_basis(old, new):
    """Return an orthonormal union and JSON-safe numerical diagnostics.

    Inputs must share dimension, device, and FP32/FP64 storage dtype. QR fixes
    the complete old space first; SVD sees only the new orthogonal residual.
    A new singular direction is redundant at or below
    ``max(D, K) * eps(storage_dtype) * max(1, smax(residual))`` where K is
    the number of new columns. There is no rank cap beyond ambient dimension.
    All decompositions use FP64, followed by checks on the stored output.
    """
    if not isinstance(old, torch.Tensor) or not isinstance(new, torch.Tensor):
        raise ValueError('old and new must be basis tensors')
    if old.dtype not in (torch.float32, torch.float64) or new.dtype not in (torch.float32, torch.float64):
        raise ValueError('Basis storage must be float32 or float64')
    if old.device != new.device:
        raise ValueError('Bases must share the same device')
    if old.dtype != new.dtype:
        raise ValueError('Bases must share the same storage dtype')
    with torch.autocast(device_type=old.device.type, enabled=False):
        _validate_basis(old)
        _validate_basis(new, old.shape[0])
        dimension, old_rank = old.shape
        fixed, _ = torch.linalg.qr(old.double(), mode='reduced')
        residual = new.double()
        if old_rank == dimension:
            residual = torch.zeros_like(residual)
        elif old_rank:
            # Reprojection removes roundoff left by the first subtraction.
            residual = residual - fixed @ (fixed.T @ residual)
            residual = residual - fixed @ (fixed.T @ residual)
        directions, singular, _ = torch.linalg.svd(residual, full_matrices=False)
        maximum = float(singular[0]) if singular.numel() else 0.
        tolerance = max(residual.shape) * torch.finfo(old.dtype).eps * max(1., maximum)
        additions = directions[:, singular > tolerance]
        candidates = torch.cat((fixed, additions), dim=1)
        orthogonal, _ = torch.linalg.qr(candidates, mode='reduced')
        basis = orthogonal.to(old.dtype).contiguous()
        _validate_basis(basis)
        stored = basis.double()
        identity = torch.eye(stored.shape[1], dtype=torch.float64, device=old.device)
        orthogonality = float((stored.T @ stored - identity).norm())
        old_values = old.double()
        old_residual = float((old_values - stored @ (stored.T @ old_values)).norm()
                             / old_values.norm().clamp_min(torch.finfo(torch.float64).tiny))
        if old_residual > 1e-5:
            raise ValueError('Stored cumulative basis does not preserve the old span within 1e-5')
    rank = basis.shape[1]
    return basis, {
        'input_dim': dimension, 'old_rank': old_rank, 'new_rank': new.shape[1],
        'added_rank': rank - old_rank, 'rank': rank,
        'remaining_dimension': dimension - rank, 'capacity_exhausted': rank == dimension,
        'storage_dtype': str(old.dtype), 'rank_tolerance': tolerance,
        'residual_singular_values': singular.cpu().tolist(),
        'orthogonality_residual': orthogonality, 'old_span_residual': old_residual,
    }


def union_bank(old_bases, new_bases):
    """Union matching layer dictionaries and return statistics keyed by layer."""
    if not isinstance(old_bases, dict) or not isinstance(new_bases, dict) or set(old_bases) != set(new_bases):
        raise ValueError('Basis banks must be dictionaries with matching layer keys')
    bases, statistics = {}, {}
    for name in old_bases:
        bases[name], statistics[name] = union_basis(old_bases[name], new_bases[name])
    return bases, statistics
