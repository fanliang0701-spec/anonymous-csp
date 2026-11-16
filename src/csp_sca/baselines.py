"""Faithfully scoped O-LoRA and full-Q/V GPM adaptations; see BASELINE_PROTOCOL.md."""
from __future__ import annotations
import math
import re
import torch
from torch import nn
from torch.nn import functional as F


def attentions(model):
    result = [(n, m) for n, m in model.named_modules()
              if re.fullmatch(r'backbone\.layers\.\d+\.self_attn', n)]
    if not result:
        raise ValueError('No complete decoder attention maps found')
    return result


class OLoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=8, seed=13):
        super().__init__()
        leaf = base
        while isinstance(leaf, OLoRALinear):
            leaf = leaf.base
        if not isinstance(leaf, nn.Linear) or rank < 1 or alpha <= 0:
            raise ValueError('Expected linear map, positive rank/alpha')
        self.base = base.requires_grad_(False)
        self.base.zero_grad(set_to_none=True)
        self.scaling = alpha / rank
        dtype = torch.float64 if leaf.weight.dtype == torch.float64 else torch.float32
        generator = torch.Generator(device='cpu').manual_seed(seed)
        a = torch.empty(rank, leaf.in_features, dtype=dtype)
        nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=generator)
        self.A = nn.Parameter(a.to(leaf.weight.device))
        self.B = nn.Parameter(torch.zeros(leaf.out_features, rank, device=leaf.weight.device, dtype=dtype))

    def forward(self, x):
        y = self.base(x)
        with torch.autocast(x.device.type, enabled=False):
            delta = F.linear(F.linear(x.to(self.A.dtype), self.A), self.B) * self.scaling
        return y + delta.to(y.dtype)


def install_olora(model, rank=8, alpha=8, seed=13):
    maps = attentions(model)
    model.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    installed = {}
    for name, module in maps:
        for projection in ('q_proj', 'v_proj'):
            adapter = OLoRALinear(getattr(module, projection), rank, alpha, seed)
            adapter.train(module.training)
            setattr(module, projection, adapter)
            installed[name + '.' + projection] = adapter
    return installed


def olora_regularization(model, coefficient=.5):
    """Official unnormalized entrywise L1 overlap of old A rows with new A rows."""
    terms = []
    for _, module in attentions(model):
        for name in ('q_proj', 'v_proj'):
            current = getattr(module, name)
            if not isinstance(current, OLoRALinear):
                continue
            prior = current.base
            with torch.autocast(current.A.device.type, enabled=False):
                while isinstance(prior, OLoRALinear):
                    terms.append((prior.A.detach() @ current.A.T).abs().sum())
                    prior = prior.base
    if terms:
        return coefficient * torch.stack(terms).sum()
    return next(model.parameters()).new_zeros(())


def validate_basis(basis, dimension):
    if basis.ndim != 2 or basis.shape[0] != dimension or basis.shape[1] > dimension:
        raise ValueError('Wrong basis dimensions')
    b = basis.float() if basis.dtype != torch.float64 else basis
    if not torch.isfinite(b).all() or not torch.allclose(b.T @ b,
            torch.eye(b.shape[1], device=b.device, dtype=b.dtype), atol=2e-5, rtol=0):
        raise ValueError('Basis must be finite and orthonormal')


class GPMController:
    def __init__(self, model, bases):
        maps = attentions(model)
        if set(bases) != {name for name, _ in maps}:
            raise ValueError('Require one GPM input basis per attention')
        self.entries = []
        for name, module in maps:
            for projection in ('q_proj', 'v_proj'):
                linear = getattr(module, projection)
                if not isinstance(linear, nn.Linear):
                    raise ValueError('GPM requires full linear Q/V weights without adapters')
                validate_basis(bases[name], linear.in_features)
                self.entries.append((linear.weight, bases[name].detach().to(linear.weight).clone()))
        model.requires_grad_(False)
        model.zero_grad(set_to_none=True)
        for weight, _ in self.entries:
            weight.requires_grad_(True)

    @torch.no_grad()
    def project_gradients(self):
        """Call after backward/clipping, before zero-momentum, zero-decay SGD.step."""
        for weight, basis in self.entries:
            if weight.grad is not None:
                weight.grad.sub_((weight.grad @ basis) @ basis.T)


def setup_gpm(model, bases):
    return GPMController(model, bases)


@torch.no_grad()
def update_gpm_basis(activations, previous, energy=.9, initial=False):
    """Uncentered activation SVD, then original residual-energy cumulative update.

    activations[N,D] contains actual token activations, never class averages.
    Retains every previous direction even when the current task has no energy there.
    """
    if not 0 < energy <= 1 or activations.ndim != 2 or not torch.isfinite(activations).all():
        raise ValueError('Require finite activation matrix and energy in (0,1]')
    x = activations.detach()
    x = x.double() if x.dtype == torch.float64 else x.float()
    previous = previous.to(x)
    validate_basis(previous, x.shape[1])
    total = x.square().sum()
    residual = x - (x @ previous) @ previous.T
    _, s, vh = torch.linalg.svd(residual, full_matrices=False)
    explained = (total - residual.square().sum()).clamp_min(0)
    needed = (energy * total - explained).clamp_min(0)
    if initial and previous.shape[1]:
        raise ValueError("Initial GPM memory must be empty")
    if initial and total > 0:
        # Match official initial convention, including the missing +1.
        count = int((s.square().cumsum(0) / total < energy).sum())
        tolerance = max(residual.shape) * torch.finfo(x.dtype).eps * s[0]
        count = min(count, int((s > tolerance).sum()))
    elif total == 0 or needed == 0:
        count = 0
    else:
        tolerance = max(residual.shape) * torch.finfo(x.dtype).eps * s[0]
        available = int((s > tolerance).sum())
        count = min(int(torch.searchsorted(s.square().cumsum(0), needed).item()) + 1, available,
                    x.shape[1] - previous.shape[1])
    candidate = torch.cat((previous, vh[:count].T), dim=1)
    basis = torch.linalg.qr(candidate, mode='reduced').Q if candidate.shape[1] else candidate
    validate_basis(basis, x.shape[1])
    retained = float((x @ basis).square().sum() / total) if total > 0 else None
    return basis, dict(input_dim=x.shape[1], previous_rank=previous.shape[1],
        added_rank=count, rank=basis.shape[1], energy_threshold=energy,
        activation_energy=float(total), retained_energy_fraction=retained,
        activation_rows=x.shape[0])


@torch.no_grad()
def extract_gpm_memory(model, current, previous, device, per_class=16, energy=.9, initial=False):
    """Current-training anchors only, actual inputs to each Q map, shared with V."""
    from .memory import balanced_anchor_indices
    from .engine import loader
    indices = balanced_anchor_indices(current['train_labels'], per_class, 13)
    maps = attentions(model)
    if set(previous) != {name for name, _ in maps}:
        raise ValueError('Previous memory scope mismatch')
    samples, handles = {name: [] for name, _ in maps}, []
    def collect(name, inputs):
        samples[name].append(inputs.detach().reshape(-1, inputs.shape[-1]).float().cpu())
    was_training = model.training
    model.eval()
    for name, module in maps:
        handles.append(module.q_proj.register_forward_pre_hook(
            lambda module, args, name=name: collect(name, args[0])))
    try:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            for x, _ in loader(current['train_traces'][indices], current['train_labels'][indices], 64):
                model(x.to(device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    bases, statistics = {}, {}
    for name, _ in maps:
        x = torch.cat(samples.pop(name)).to(device)
        basis, statistics[name] = update_gpm_basis(x, previous[name], energy, initial=initial)
        bases[name] = basis.cpu()
        del x, basis
    return dict(bases=bases, statistics=statistics, anchors=dict(indices=indices.tolist(),
        samples=len(indices), seed=13, per_class=per_class),
        algorithm='GPM uncentered activation SVD and cumulative residual-energy memory')
