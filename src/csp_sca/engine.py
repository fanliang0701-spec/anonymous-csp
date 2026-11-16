"""Training primitives from the experiment code with portable checkpoint I/O."""
from pathlib import Path
import json
import math
import os
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

def loader(traces, labels, batch_size):
    return DataLoader(TensorDataset(torch.from_numpy(traces), torch.from_numpy(labels)),
                      batch_size=batch_size, shuffle=False)


def schedule_multiplier(step, total_steps):
    warmup_steps = max(1, math.ceil(.05 * total_steps))
    if step < warmup_steps:
        return step / warmup_steps
    progress = min(1., (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return .5 * (1 + math.cos(math.pi * progress))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_epoch(model, loader, optimizer, device, scheduler=None, accumulation_steps=8, progress=None):
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    samples = steps = window_samples = 0
    total_nll = window_nll = 0.
    for microbatch, (traces, labels) in enumerate(loader, 1):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(traces.to(device))
            loss = F.cross_entropy(logits.float(), labels.to(device), reduction='sum')
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError('nonfinite training logits/loss')
        loss.backward()
        count = len(labels)
        samples += count
        window_samples += count
        value = loss.detach().item()
        total_nll += value
        window_nll += value
        if microbatch % accumulation_steps == 0 or microbatch == len(loader):
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(window_samples)
            torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            if progress is not None:
                progress({'step': steps, 'samples': samples, 'window_samples': window_samples,
                          'nll': total_nll / samples, 'window_nll': window_nll / window_samples})
            window_samples = 0
            window_nll = 0.
    synchronize(device)
    if not samples:
        raise ValueError('training loader is empty')
    return {'nll': total_nll / samples, 'samples': samples, 'steps': steps}


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    logits = []
    nll = correct = samples = 0
    for traces, labels in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            output = model(traces.to(device))
        output = output.float().cpu()
        loss = F.cross_entropy(output, labels, reduction='sum')
        if not torch.isfinite(output).all() or not torch.isfinite(loss):
            raise FloatingPointError('nonfinite validation logits/loss')
        nll += loss.item()
        correct += int((output.argmax(dim=1) == labels).sum())
        samples += len(labels)
        logits.append(output.numpy())
    if not samples:
        raise ValueError('validation loader is empty')
    return {'nll': nll / samples, 'accuracy': correct / samples, 'samples': samples,
            'logits': np.concatenate(logits)}


def baseline_epoch(model, batches, optimizer, device, scheduler=None, accumulation_steps=1,
                progress=None, *, method, controller=None, coefficient=.5):
    from .baselines import olora_regularization
    if accumulation_steps != 1:
        raise ValueError('Locked baseline protocol uses one batch per update')
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    samples = steps = 0
    total_nll = total_regularizer = 0.
    for traces, labels in batches:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(traces.to(device))
            ce = F.cross_entropy(logits.float(), labels.to(device), reduction='mean')
        penalty = olora_regularization(model, coefficient) if method == 'olora' else ce.new_zeros(())
        loss = ce + penalty
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite baseline logits/loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        if method == 'gpm':
            if controller is None:
                raise ValueError('GPM needs its preceding model activation bank')
            controller.project_gradients()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        count = len(labels)
        samples += count
        steps += 1
        total_nll += float(ce.detach()) * count
        total_regularizer += float(penalty.detach()) * count
        if progress:
            progress(dict(step=steps, samples=samples, window_samples=count,
                nll=total_nll / samples, window_nll=float(ce.detach()),
                regularization=total_regularizer / samples))
    synchronize(device)
    if not samples:
        raise ValueError('Empty training data')
    return dict(nll=total_nll / samples, regularization=total_regularizer / samples,
                samples=samples, steps=steps)


def penalized_epoch(model, batches, optimizer, device, scheduler, progress, penalty_fn):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    samples = steps = 0
    total_nll = total_pen = 0.
    for traces, labels in batches:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            logits = model(traces.to(device))
            ce = F.cross_entropy(logits.float(), labels.to(device), reduction='mean')
        pen = penalty_fn()
        loss = ce + pen
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite penalized loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        samples += len(labels)
        steps += 1
        total_nll += float(ce.detach()) * len(labels)
        total_pen += float(pen.detach()) * len(labels)
        if progress:
            progress(dict(step=steps, samples=samples, nll=total_nll / samples, penalty=total_pen / samples))
    synchronize(device)
    return dict(nll=total_nll / samples, penalty=total_pen / samples, samples=samples, steps=steps)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def restore_parameters(model, state):
    parameters = dict(model.named_parameters())
    expected = {n for n, p in parameters.items() if p.requires_grad}
    if expected != set(state):
        raise ValueError('Checkpoint trainable parameter scope differs')
    for name, value in state.items():
        if parameters[name].shape != value.shape or not torch.isfinite(value).all():
            raise ValueError('Invalid checkpoint tensor: ' + name)
        parameters[name].copy_(value.to(parameters[name]))


def fit(model, current, output, device, *, seed, epochs=10, groups=None,
        learning_rate=1e-4, method='csp', controller=None, coefficient=.5,
        penalty_fn=None, make_batches=None, validation_batch_size=64):
    """Select the earliest minimum of current-task validation NLL."""
    output = Path(output)
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError('Training requires FP32 trainable parameters')
    groups = groups or [dict(params=parameters, lr=learning_rate)]
    ids = [id(p) for group in groups for p in group['params']]
    if len(ids) != len(set(ids)) or set(ids) != {id(p) for p in parameters}:
        raise ValueError('Optimizer does not match the trainable scope')
    frozen = [(p, p._version) for p in model.parameters() if not p.requires_grad]
    optimizer = (torch.optim.SGD(groups, momentum=0, weight_decay=0)
                 if method == 'gpm' else torch.optim.AdamW(groups, weight_decay=0))
    steps = (len(current['train_labels']) + 63) // 64
    if epochs < 1 or steps < 1:
        raise ValueError('Training requires positive epochs and a nonempty training split')
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: schedule_multiplier(step, epochs * steps))
    validation = loader(current['val_traces'], current['val_labels'], validation_batch_size)
    rng = torch.Generator(device='cpu').manual_seed(seed)
    history, best, selected = [], None, None
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(current['train_labels']), generator=rng).numpy()
        batches = (make_batches(order, epoch) if make_batches is not None else
                   loader(current['train_traces'][order], current['train_labels'][order], 64))
        if penalty_fn is not None:
            trained = penalized_epoch(model, batches, optimizer, device, scheduler, None, penalty_fn)
        elif method in ('gpm', 'olora'):
            trained = baseline_epoch(model, batches, optimizer, device, scheduler,
                                     method=method, controller=controller, coefficient=coefficient)
        else:
            trained = train_epoch(model, batches, optimizer, device, scheduler, accumulation_steps=1)
        metrics = evaluate(model, validation, device)
        metrics.pop('logits')
        row = dict(epoch=epoch, training=trained, validation=metrics)
        history.append(row)
        if best is None or metrics['nll'] < best['validation']['nll']:
            best, selected = row, trainable_state(model)
        write_json(output / 'history.json', history)
    restore_parameters(model, selected)
    if any(p._version != version for p, version in frozen):
        raise RuntimeError('A frozen parameter changed during training')
    report = dict(selected_epoch=best['epoch'], validation=best['validation'], history=history,
                  criterion='earliest minimum current-task validation NLL',
                  trainable_parameters=sum(p.numel() for p in parameters),
                  training_seconds=time.monotonic() - started)
    write_json(output / 'training.json', report)
    return report, selected
