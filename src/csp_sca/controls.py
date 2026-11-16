"""Replay, Fisher regularization, and parameter-matched InfLoRA."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from .protected_lowrank import _leaf_linear
from .memory import layer_names, balanced_anchor_indices, anchors
from .baselines import attentions
from .inflora import InfLoRALinear

def quotas(domains, per_class):
    return [per_class // domains + int(d < per_class % domains) for d in range(domains)]


def refresh_buffer(old, domain, current, capacity, seed=13):
    x, y = current['train_traces'], current['train_labels']
    rng = np.random.default_rng(seed + 100000 * domain)
    q = quotas(domain + 1, capacity // 256)
    picks = []
    for label in range(256):
        for d, count in enumerate(q):
            if d == domain:
                cand = np.flatnonzero(y == label)
            else:
                cand = np.flatnonzero((old['labels'] == label) & (old['domains'] == d))
            take = min(count, len(cand))
            picks.append((d, label, np.sort(rng.choice(cand, take, replace=False))))
    total = sum(len(p[2]) for p in picks)
    out = dict(traces=np.empty((total, x.shape[1]), np.float32), labels=np.empty(total, np.uint16),
               domains=np.empty(total, np.uint8), rows=np.empty(total, np.uint32))
    pos = 0
    for d, label, sel in picks:
        sl = slice(pos, pos + len(sel))
        out['traces'][sl] = x[sel] if d == domain else old['traces'][sel]
        out['rows'][sl] = sel if d == domain else old['rows'][sel]
        out['labels'][sl], out['domains'][sl] = label, d
        pos += len(sel)
    return out


class ReplayBatches:
    def __init__(self, current, order, buffer, rng):
        self.current, self.order, self.buffer, self.rng = current, order, buffer, rng
        self.replay_rows = 0

    def __len__(self):
        return (len(self.order) + 63) // 64

    def __iter__(self):
        for start in range(0, len(self.order), 64):
            rows = self.order[start:start + 64]
            extra = self.rng.integers(0, len(self.buffer['labels']), size=len(rows) // 4)
            self.replay_rows += len(extra)
            xs = np.concatenate((self.current['train_traces'][rows], self.buffer['traces'][extra]))
            ys = np.concatenate((self.current['train_labels'][rows], self.buffer['labels'][extra])).astype(np.int64)
            yield torch.from_numpy(xs), torch.from_numpy(ys)


def qv_leaves(model):
    return {f'{n}.{p}': _leaf_linear(model.get_submodule(f'{n}.{p}')) for n in layer_names(model) for p in ('q_proj', 'v_proj')}


def estimate_fisher(model, current, device, per_class=8):
    idx = balanced_anchor_indices(current['train_labels'], anchors(current, per_class), 13)
    leaves = qv_leaves(model)
    flags = {n: p.requires_grad for n, p in model.named_parameters()}
    model.requires_grad_(False)
    weights = [leaf.weight for leaf in leaves.values()]
    for w in weights:
        w.requires_grad_(True)
    fisher = [torch.zeros_like(w) for w in weights]
    model.eval()
    started = time.monotonic()
    for i in idx:
        x = torch.from_numpy(current['train_traces'][i:i + 1]).to(device)
        y = torch.as_tensor(current['train_labels'][i:i + 1]).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(x)
        loss = F.cross_entropy(logits.float(), y)
        grads = torch.autograd.grad(loss, weights)
        for f, g in zip(fisher, grads):
            f.add_(g.float().square())
    for w in weights:
        w.requires_grad_(False)
    for n, p in model.named_parameters():
        p.requires_grad_(flags[n])
    result = {k: (f / len(idx)).cpu() for k, f in zip(leaves, fisher)}
    return result, dict(samples=len(idx), seconds=time.monotonic() - started)


def ewc_penalty(adapters, fisher, lam):
    terms = []
    with torch.autocast('cuda', enabled=False):
        for k, a in adapters.items():
            delta = (a.B @ a.A) * a.scaling
            terms.append((fisher[k] * delta.square()).sum())
    return .5 * lam * torch.stack(terms).sum()


def install_budget_inflora(model,designs):
    maps=attentions(model)
    if set(designs)!={name for name,_ in maps}:raise ValueError('Incomplete input designs')
    model.requires_grad_(False);model.zero_grad(set_to_none=True)
    adapters={}
    for name,module in maps:
        if designs[name].ndim!=2 or len(designs[name])!=19:
            raise ValueError('Budget-matched design must provide exactly19 input directions')
        for projection,rank in [('q_proj',19),('v_proj',18)]:
            adapter=InfLoRALinear(getattr(module,projection),designs[name][:rank])
            adapter.train(module.training);setattr(module,projection,adapter)
            adapters[name+'.'+projection]=adapter
    count=sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected=sum(a.B.shape[0]*a.B.shape[1] for a in adapters.values())
    if count!=expected:raise ValueError('Unexpected trainable parameters outside B factors')
    if len(maps)==28 and all(a.A.shape[1]==2048 for a in adapters.values()) and count!=1605632:
        raise ValueError('Qwen3 incremental budget must exactly match CSP')
    return adapters
