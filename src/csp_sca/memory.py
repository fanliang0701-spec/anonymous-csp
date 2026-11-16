"""Class/token means and cumulative memory used by CSP."""
import time
import numpy as np
import torch
from .protected_lowrank import basis_from_class_token_means, _validate_basis
from .multidevice_memory import union_bank
from .engine import loader

def layer_names(model):
    return [f'backbone.layers.{i}.self_attn' for i in range(len(model.backbone.layers))]

def balanced_anchor_indices(labels, per_class, seed, classes=256):
    labels=np.asarray(labels)
    if labels.ndim!=1 or per_class<1 or classes<1:
        raise ValueError('invalid anchor labels/count')
    rng=np.random.default_rng(seed)
    selected=[]
    for label in range(classes):
        rows=np.flatnonzero(labels==label)
        if len(rows)<per_class:
            raise ValueError(f'class {label} has too few training rows')
        selected.extend(rng.choice(rows,per_class,replace=False).tolist())
    return np.array(sorted(selected),dtype=np.int64)


def basis_for_storage(basis):
    """Remove SVD roundoff with rank-preserving QR before the FP32 boundary."""
    from .protected_lowrank import _validate_basis
    if (basis.ndim!=2 or basis.shape[1]>basis.shape[0]
            or not torch.isfinite(basis).all()):
        raise ValueError('invalid basis shape or values')
    if basis.shape[1]:
        q,r=torch.linalg.qr(basis.double(),mode='reduced')
        tolerance=max(basis.shape)*torch.finfo(torch.float64).eps*r.norm()
        if (r.diagonal().abs()<=tolerance).any():
            raise ValueError('basis rank would change during orthogonalization')
        result=q.float().cpu()
    else:
        result=basis.float().cpu()
    _validate_basis(result)
    return result


@torch.no_grad()
def class_means(model, current, device, per_class=16):
    indices = balanced_anchor_indices(current['train_labels'], per_class, 13)
    labels = current['train_labels'][indices]
    counts = np.bincount(labels, minlength=256)
    sums, context, handles = {}, {}, []

    def collect(name, hidden):
        if name not in sums:
            sums[name] = torch.zeros((256, hidden.shape[1], hidden.shape[2]), device=device)
        sums[name].index_add_(0, context['labels'], hidden.detach().float())

    model.eval()
    for name in layer_names(model):
        handles.append(model.get_submodule(name + '.q_proj').register_forward_pre_hook(
            lambda module, args, name=name: collect(name, args[0])))
    try:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            for x, y in loader(current['train_traces'][indices], labels, 64):
                context['labels'] = y.to(device)
                model(x.to(device))
    finally:
        for handle in handles:
            handle.remove()
    divisor = torch.as_tensor(counts, device=device, dtype=torch.float32)[:, None, None]
    return {k: v / divisor for k, v in sums.items()}, dict(
        indices=indices.tolist(), class_counts=counts.tolist(), samples=len(indices))


@torch.no_grad()
def memory_from_means(means, previous):
    if set(means) != set(previous):
        raise ValueError('Memory layer keys differ')
    current, statistics = {}, {}
    for name, values in means.items():
        basis, statistics[name] = basis_from_class_token_means(values, energy=.9)
        current[name] = basis_for_storage(basis).to(values.device)
    bases, unions = union_bank({k: v.to(current[k].device) for k, v in previous.items()}, current)
    return dict(bases={k: basis_for_storage(v) for k, v in bases.items()},
                current_bases={k: v.cpu() for k, v in current.items()},
                statistics=statistics, union_statistics=unions)


@torch.no_grad()
def signal_geometry(means, basis):
    """Fraction of current centered class/token energy available after projection."""
    _validate_basis(basis, means.shape[-1])
    values = means.float()
    centered = values - values.mean(dim=0, keepdim=True)
    flat = centered.reshape(-1, centered.shape[-1])
    u = basis.to(device=values.device, dtype=torch.float32)
    residual = flat * 0 if u.shape[1] == u.shape[0] else flat - (flat @ u) @ u.T
    total, remaining = float(flat.square().sum()), float(residual.square().sum())
    return dict(input_dim=basis.shape[0], protected_rank=basis.shape[1],
        remaining_dimension=basis.shape[0] - basis.shape[1],
        occupied_fraction=basis.shape[1] / basis.shape[0],
        centered_class_energy=total, remaining_class_energy=remaining,
        remaining_class_energy_fraction=min(1., max(0., remaining / total)) if total > 0 else None)


def anchors(current, cap=16):
    count = int(np.bincount(current['train_labels'], minlength=256).min())
    if count < 1:
        raise ValueError('Memory requires every class in training')
    return min(cap, count)


def empty_bank(model):
    return {n: torch.empty(model.backbone.config.hidden_size, 0) for n in layer_names(model)}


@torch.no_grad()
def task_directions(model, current, device):
    """Current-task CSP basis with unit total weight split between class-variation and token-mean energy."""
    means, _ = class_means(model, current, device, per_class=anchors(current))
    basis = memory_from_means(means, empty_bank(model))['current_bases']
    out = {}
    for n, b in basis.items():
        b = b.double().to(device)
        v = means[n].double()
        tok = v.mean(0)
        z = (v - tok).reshape(-1, v.shape[-1])
        t = tok / tok.norm(dim=1, keepdim=True).clamp_min(1e-30)
        e = (z @ b).square().sum(0) / z.square().sum().clamp_min(1e-30) + (t @ b).square().sum(0) / t.square().sum()
        e = e / e.sum()
        out[n] = (b * e.sqrt()).float().cpu()
    return out


@torch.no_grad()
def budget_union(old_bank, scaled, cap):
    """Frequent-directions style weighted union truncated to at most `cap` directions per layer."""
    bases, weights, stats = {}, {}, {}
    for n, new in scaled.items():
        old = old_bank['bases'][n].double() * old_bank['weights'][n].double()[None, :]
        sketch = torch.cat((old, new.double()), 1)
        u, sv, _ = torch.linalg.svd(sketch, full_matrices=False)
        keep = int((sv > max(sketch.shape) * torch.finfo(torch.float64).eps * sv[0]).sum())
        kept = min(keep, cap)
        bases[n] = basis_for_storage(u[:, :kept].contiguous()) if kept else torch.empty(new.shape[0], 0)
        weights[n] = sv[:kept].float()
        stats[n] = dict(union_rank=keep, kept=kept, dropped=keep - kept,
                        dropped_weight_fraction=float(sv[kept:].square().sum() / sv.square().sum()))
    return dict(bases=bases, weights=weights, statistics=stats, cap=cap)


def wide_maps(model):
    return [(f'fc{k}.{i}', getattr(model, f'fc{k}'), i) for i in range(model.blocks) for k in (1, 2)]


@torch.no_grad()
def wide_class_means(model, current, device, per_class):
    idx = balanced_anchor_indices(current['train_labels'], per_class, 13)
    labels = current['train_labels'][idx]
    sums, ctx, handles = {}, {}, []
    def collect(name, x):
        if name not in sums:
            sums[name] = torch.zeros(256, x.shape[-1], device=device)
        sums[name].index_add_(0, ctx['y'], x.detach().float())
    for name, container, i in wide_maps(model):
        handles.append(container[i].register_forward_pre_hook(lambda m, a, name=name: collect(name, a[0])))
    model.eval()
    try:
        for x, y in loader(current['train_traces'][idx], labels, 256):
            ctx['y'] = y.to(device)
            model(x.to(device))
    finally:
        for h in handles:
            h.remove()
    counts = torch.as_tensor(np.bincount(labels, minlength=256), device=device, dtype=torch.float32)[:, None]
    return {k: (v / counts)[:, None, :] for k, v in sums.items()}
