"""Shared-head Q/V adaptation of official InfLoRA_domain; protocol is explicit."""
from __future__ import annotations
import math
import hashlib
import time
import torch
from torch import nn
from torch.nn import functional as F


def validate_basis(basis, dimension):
    if basis.ndim != 2 or basis.shape[0] != dimension or basis.shape[1] > dimension:
        raise ValueError('Invalid memory shape')
    if not torch.isfinite(basis).all() or not torch.allclose(basis.T @ basis,
            torch.eye(basis.shape[1], dtype=basis.dtype, device=basis.device), atol=2e-5, rtol=0):
        raise ValueError('Memory must be finite and orthonormal')


def token_covariance(chunks):
    total, count = None, 0
    for chunk in chunks:
        x = chunk.detach().reshape(-1, chunk.shape[-1])
        if x.dtype != torch.float64: x = x.float()
        total = x.T @ x if total is None else total + x.T @ x
        count += len(x)
    if not count: raise ValueError('No training tokens')
    return total / count


@torch.no_grad()
def design_down(covariance, previous, rank=8):
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not torch.isfinite(covariance).all():
        raise ValueError('Require finite square covariance')
    previous = previous.to(covariance)
    validate_basis(previous, len(covariance))
    if len(covariance)-previous.shape[1] < rank:
        raise ValueError('InfLoRA complement cannot contain requested rank')
    residual = covariance - previous @ (previous.T @ covariance)
    u, singular, _ = torch.linalg.svd(residual, full_matrices=False)
    down = u[:, :rank]
    # Repair floating-point leakage without changing the designed residual span.
    for _ in range(2): down -= previous @ (previous.T @ down)
    down = torch.linalg.qr(down, mode='reduced').Q.T / math.sqrt(3)
    overlap = float((down @ previous).norm())
    if overlap > 2e-5: raise ValueError('Designed down factor escaped complement')
    return down, dict(rank=rank, memory_rank=previous.shape[1], complement_overlap=overlap,
        smallest_selected_singular_value=float(singular[rank-1]), scaling='1/sqrt(3)')


@torch.no_grad()
def update_memory(covariance, previous, threshold=.95, initial=False):
    if not 0 < threshold <= 1 or not torch.isfinite(covariance).all():
        raise ValueError('Invalid covariance or threshold')
    previous = previous.to(covariance)
    validate_basis(previous, len(covariance))
    if initial and previous.shape[1]: raise ValueError('Initial memory is empty')
    total = covariance.square().sum()
    residual = covariance - previous @ (previous.T @ covariance)
    u, singular, _ = torch.linalg.svd(residual, full_matrices=False)
    tolerance = len(covariance) * torch.finfo(covariance.dtype).eps * singular[0]
    available = min(int((singular > tolerance).sum()), len(covariance)-previous.shape[1])
    explained = (total-residual.square().sum()).clamp_min(0)
    needed = (threshold*total-explained).clamp_min(0)
    if total == 0 or available == 0: count = 0
    elif initial:
        count = min(max(int((singular.square().cumsum(0)/total < threshold).sum()), 1), available)
    elif needed == 0: count = 0
    else: count = min(int(torch.searchsorted(singular.square().cumsum(0), needed))+1, available)
    added = u[:, :count]
    for _ in range(2): added -= previous @ (previous.T @ added)
    candidate = torch.cat((previous, added), dim=1)
    basis = torch.linalg.qr(candidate, mode='reduced').Q if candidate.shape[1] else candidate
    validate_basis(basis, len(covariance))
    old_error = float((previous-basis@(basis.T@previous)).norm()/previous.norm().clamp_min(1e-30))
    if old_error > 2e-5: raise ValueError('Cumulative memory lost previous directions')
    return basis, dict(input_dim=len(covariance), previous_rank=previous.shape[1], added_rank=count,
        rank=basis.shape[1], threshold=threshold, covariance_energy=float(total),
        retained_covariance_energy_fraction=float((basis.T@covariance).square().sum()/total) if total>0 else None,
        old_span_residual=old_error)


class InfLoRALinear(nn.Module):
    def __init__(self, base, down):
        super().__init__()
        leaf = base
        while isinstance(leaf, InfLoRALinear): leaf = leaf.base
        if not isinstance(leaf, nn.Linear) or down.shape[1] != leaf.in_features:
            raise ValueError('Down factor input dimension mismatch')
        self.base = base.requires_grad_(False)
        self.base.zero_grad(set_to_none=True)
        dtype = torch.float64 if leaf.weight.dtype == torch.float64 else torch.float32
        self.A = nn.Parameter(down.detach().to(device=leaf.weight.device, dtype=dtype).clone(), requires_grad=False)
        self.B = nn.Parameter(torch.zeros(leaf.out_features, len(down), device=leaf.weight.device, dtype=dtype))

    def forward(self, x):
        y = self.base(x)
        with torch.autocast(x.device.type, enabled=False):
            delta = F.linear(F.linear(x.to(self.A.dtype), self.A), self.B)
        return y + delta.to(y.dtype)


def install_inflora(model, designs):
    from .baselines import attentions
    maps = attentions(model)
    if set(designs) != {name for name,_ in maps}: raise ValueError('Incomplete designed factors')
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    result = {}
    for name, module in maps:
        for projection in ('q_proj','v_proj'):
            adapter = InfLoRALinear(getattr(module, projection), designs[name])
            adapter.train(module.training)
            setattr(module, projection, adapter)
            result[name+'.'+projection] = adapter
    return result


@torch.no_grad()
def extract_covariances(model, current, device, per_class=16):
    from .baselines import attentions
    from .memory import balanced_anchor_indices
    from .engine import loader
    started = time.monotonic()
    indices = balanced_anchor_indices(current['train_labels'], per_class, 13)
    maps = attentions(model)
    sums, counts, handles = {}, {}, []
    def collect(name, inputs):
        x = inputs.detach().reshape(-1, inputs.shape[-1]).float()
        with torch.autocast(device.type, enabled=False):
            part = x.T @ x
            if name not in sums: sums[name], counts[name] = part, len(x)
            else: sums[name].add_(part); counts[name] += len(x)
    was_training = model.training
    model.eval()
    for name,module in maps:
        handles.append(module.q_proj.register_forward_pre_hook(lambda module,args,name=name:collect(name,args[0])))
    try:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'):
            for x,_ in loader(current['train_traces'][indices], current['train_labels'][indices],64): model(x.to(device))
    finally:
        for handle in handles: handle.remove()
        model.train(was_training)
    covariance = {name: (sums.pop(name)/counts[name]).cpu() for name,_ in maps}
    return dict(covariances=covariance, token_rows=counts,
        covariance_sha256={name:hashlib.sha256(c.numpy().tobytes()).hexdigest() for name,c in covariance.items()},
        anchors=dict(indices=indices.tolist(), samples=len(indices), seed=13, per_class=per_class),
        wall_seconds=time.monotonic()-started,
        algorithm='uncentered actual-token covariance X.T@X/N; never class means')


def memory_from_covariances(payload, previous, device, threshold, initial=False):
    bases, stats = {}, {}
    started = time.monotonic()
    for name,covariance in payload['covariances'].items():
        basis,stats[name] = update_memory(covariance.to(device), previous[name].to(device), threshold, initial)
        bases[name] = basis.cpu()
    return dict(bases=bases, statistics=stats, anchors=payload['anchors'], token_rows=payload['token_rows'],
        covariance_sha256=payload['covariance_sha256'],
        wall_seconds=payload['wall_seconds']+time.monotonic()-started,
        algorithm='official InfLoRA_domain covariance SVD squared singular energy; cumulative old span retained')
