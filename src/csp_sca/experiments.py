"""Portable task orchestration around the original CSP and baseline implementations."""
from pathlib import Path
import copy
import json
import time
import numpy as np
import torch

from .models import build_model, WideMLP
from .protected_lowrank import ProtectedLowRankLinear, install_qv_adapters
from .memory import (anchors, layer_names, class_means, memory_from_means,
                     wide_maps, wide_class_means, task_directions, budget_union)
from .baselines import install_olora, setup_gpm, extract_gpm_memory
from .inflora import install_inflora, extract_covariances, design_down, memory_from_covariances
from .controls import (install_budget_inflora, refresh_buffer, ReplayBatches,
                       estimate_fisher, ewc_penalty)
from .engine import fit, loader, evaluate, restore_parameters, save_checkpoint, write_json
from .metrics import rank_attack, persistent_recovery
from .data import TaskStore

METHODS = ('csp', 'unprotected', 'gpm', 'olora', 'inflora', 'inflora_m',
           'replay', 'ewc', 'mean_fixed_a', 'cov_ab', 'csp_cap', 'joint')


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        if value.dtype.kind in 'iu':
            value = value.astype(np.int64)
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {key: cpu_tree(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(val) for val in value]
    return value


def empty_bank(model):
    if isinstance(model, WideMLP):
        return {name: torch.empty(model.width, 0) for name, _, _ in wide_maps(model)}
    return {name: torch.empty(model.backbone.config.hidden_size, 0) for name in layer_names(model)}


def update_memory(model, current, previous, method, device, config, index):
    cap = anchors(current, config.get('anchors_per_class', 16))
    bank = previous['bases']
    if method in ('csp', 'mean_fixed_a'):
        if isinstance(model, WideMLP):
            means = wide_class_means(model, current, device, cap)
        else:
            means, _ = class_means(model, current, device, per_class=cap)
        return memory_from_means(means, bank)
    if method == 'csp_cap':
        return budget_union(previous, task_directions(model, current, device), config['memory_cap'])
    if method == 'gpm':
        return extract_gpm_memory(model, current, bank, device, per_class=cap,
                                  energy=.9, initial=index == 0)
    if method in ('inflora', 'inflora_m', 'cov_ab'):
        payload = extract_covariances(model, current, device, per_class=cap)
        threshold = .95 + .05 * index / len(config['tasks'])
        return memory_from_covariances(payload, bank, device, threshold, initial=index == 0)
    return previous


def designs_for(model, current, bank, device, method, rank):
    payload = extract_covariances(model, current, device, per_class=anchors(current))
    count = 19 if method == 'inflora_m' else rank
    return {name: design_down(cov.to(device), bank[name].to(device), count)[0].cpu()
            for name, cov in payload['covariances'].items()}


def install_stage(model, method, bank, designs, seed, config):
    rank = config.get('rank', 8)
    if isinstance(model, WideMLP):
        if method not in ('csp', 'unprotected'):
            raise ValueError('The wide-MLP comparison supports csp and unprotected updates')
        model.requires_grad_(False)
        result = {}
        for name, container, index in wide_maps(model):
            basis = bank[name] if method == 'csp' else bank[name][:, :0]
            adapter = ProtectedLowRankLinear(container[index], basis, rank, rank, seed)
            container[index] = adapter
            result[name] = adapter
        return result
    if method == 'gpm':
        return setup_gpm(model, bank)
    if method == 'olora':
        return install_olora(model, rank=rank, alpha=rank, seed=seed)
    if method == 'inflora_m':
        return install_budget_inflora(model, designs)
    if method in ('inflora', 'mean_fixed_a'):
        return install_inflora(model, designs)
    protected = method in ('csp', 'cov_ab', 'csp_cap')
    return install_qv_adapters(model, bank, 'protected' if protected else 'unconstrained',
                               rank=rank, alpha=rank, seed=seed)


def checkpoint_path(output, index):
    return Path(output) / 'checkpoints' / f'stage{index:02d}.pt'


def _load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def train(config, data_root, output, seed, method='csp', model_path=None,
          device='cuda', resume=False, model_factory=None):
    """Train a complete chain; resume only from completed stage boundaries."""
    if method not in METHODS:
        raise ValueError('Unknown method: ' + method)
    if config.get('batch_size', 64) != 64:
        raise ValueError('The released experiment loop uses batches of 64')
    output = Path(output)
    public = dict(config=copy.deepcopy(config), seed=int(seed), method=method)
    marker = output / 'run.json'
    if marker.exists():
        if not resume:
            raise FileExistsError('Output already exists; use --resume for the same configuration')
        if json.loads(marker.read_text(encoding='utf-8')) != public:
            raise ValueError('Resume configuration differs')
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty run directory')
    write_json(marker, public)
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; --device cpu is available for small checks')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    builder = model_factory or (lambda: build_model(config, seed, model_path))
    model = builder().to(device)
    store = TaskStore(data_root, config)
    tasks = [task['name'] for task in config['tasks']]
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError('Task names must be nonempty and unique')
    epochs = config.get('epochs', 10)
    empty = empty_bank(model)
    state = dict(bases=empty, weights={name: torch.empty(0) for name in empty})
    buffer, fisher = None, None
    source_path = checkpoint_path(output, 0)
    if source_path.exists() and resume:
        ckpt = _load_checkpoint(source_path)
        restore_parameters(model, ckpt['parameters'])
        state, fisher = ckpt['memory'], ckpt.get('fisher')
        if ckpt.get('buffer') is not None:
            buffer = {key: val.numpy() for key, val in ckpt['buffer'].items()}
    else:
        current = store.profile(tasks[0])
        source_lr = config.get('source_backbone_lr', 1e-4 if config.get('backbone') == 'transformer' else 1e-5)
        groups = model.optimizer_groups(config.get('source_io_lr', 1e-3), source_lr)
        report, params = fit(model, current, output / 'stages/00', device, seed=seed,
                             epochs=epochs, groups=groups, validation_batch_size=128)
        model.requires_grad_(False).eval()
        state = update_memory(model, current, state, method, device, config, 0)
        if method == 'replay':
            buffer = refresh_buffer(None, 0, current, config.get('replay_capacity', 4096), seed)
        elif method == 'ewc':
            fisher, _ = estimate_fisher(model, current, device)
        save_checkpoint(source_path, cpu_tree(dict(stage=0, task=tasks[0], parameters=params,
                        memory=state, report=report, buffer=buffer, fisher=fisher)))
    model.requires_grad_(False).eval()
    stages = [len(tasks) - 1] if method == 'joint' else list(range(1, len(tasks)))
    if method == 'joint' and len(tasks) < 2:
        raise ValueError('Joint reference requires at least two tasks')
    for index in stages:
        path = checkpoint_path(output, index)
        stage_seed = seed + 1000 * (index - 1)
        if path.exists() and resume:
            ckpt = _load_checkpoint(path)
            install_stage(model, method, ckpt['input_bank'], ckpt['designs'], stage_seed, config)
            restore_parameters(model, ckpt['parameters'])
            state, fisher = ckpt['memory'], ckpt.get('fisher')
            if ckpt.get('buffer') is not None:
                buffer = {key: val.numpy() for key, val in ckpt['buffer'].items()}
            model.requires_grad_(False).eval()
            continue
        if method == 'joint':
            pooled = [store.profile(name) for name in tasks]
            current = {key: np.concatenate([part[key] for part in pooled]) for key in pooled[0]}
        else:
            current = store.profile(tasks[index])
        bank = state['bases']
        designs = (designs_for(model, current, bank, device, method, config.get('rank', 8))
                   if method in ('inflora', 'inflora_m', 'mean_fixed_a') else None)
        probe = loader(current['val_traces'][:32], current['val_labels'][:32], 64)
        before = evaluate(model, probe, device)['logits']
        adapters = install_stage(model, method, bank, designs, stage_seed, config)
        after = evaluate(model, probe, device)['logits']
        np.testing.assert_array_equal(before, after, err_msg='New branch changed inherited predictions')
        penalty = batches = None
        if method == 'replay':
            batches = lambda order, epoch: ReplayBatches(current, order, buffer, np.random.default_rng([10000000 + stage_seed, epoch]))
        elif method == 'ewc':
            gpu_fisher = {name: value.to(device) for name, value in fisher.items()}
            penalty = lambda: ewc_penalty(adapters, gpu_fisher, config.get('ewc_lambda', 100.))
        lr = config.get('gpm_lr', .01) if method == 'gpm' else config.get('increment_lr', 1e-4)
        report, parameters = fit(model, current, output / 'stages' / f'{index:02d}', device,
                                 seed=stage_seed, epochs=epochs, learning_rate=lr, method=method,
                                 controller=adapters if method == 'gpm' else None,
                                 coefficient=config.get('olora_coefficient', .5),
                                 penalty_fn=penalty, make_batches=batches)
        model.requires_grad_(False).eval()
        next_state = update_memory(model, current, state, method, device, config, index)
        if method == 'replay':
            buffer = refresh_buffer(buffer, index, current, config.get('replay_capacity', 4096), seed)
        elif method == 'ewc':
            new_fisher, _ = estimate_fisher(model, current, device)
            fisher = {name: fisher[name] + value for name, value in new_fisher.items()}
        save_checkpoint(path, cpu_tree(dict(stage=index, task=tasks[index], parameters=parameters,
                         input_bank=bank, designs=designs, memory=next_state, report=report,
                         buffer=buffer, fisher=fisher)))
        state = next_state
        print(f'{method} seed={seed} stage={index} task={tasks[index]} selected={report["selected_epoch"]}', flush=True)
    write_json(output / 'training_complete.json', dict(status='complete', tasks=tasks, stages=[0] + stages))
    return output


def restore_model(output, stage, model_path=None, device='cpu', model_factory=None):
    output = Path(output)
    run = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    config, seed, method = run['config'], run['seed'], run['method']
    model = (model_factory or (lambda: build_model(config, seed, model_path)))().to(device)
    source = _load_checkpoint(checkpoint_path(output, 0))
    restore_parameters(model, source['parameters'])
    model.requires_grad_(False).eval()
    indices = ([stage] if method == 'joint' and stage else range(1, stage + 1))
    for index in indices:
        ckpt = _load_checkpoint(checkpoint_path(output, index))
        install_stage(model, method, ckpt['input_bank'], ckpt['designs'],
                      seed + 1000 * (index - 1), config)
        restore_parameters(model, ckpt['parameters'])
        model.requires_grad_(False).eval()
    return model


def evaluate_run(output, data_root, model_path=None, device='cuda', model_factory=None):
    output = Path(output)
    completion = json.loads((output / 'training_complete.json').read_text(encoding='utf-8'))
    if completion['status'] != 'complete':
        raise ValueError('Complete training before attack evaluation')
    run = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    config, method = run['config'], run['method']
    tasks = [task['name'] for task in config['tasks']]
    store = TaskStore(data_root, config)
    device = torch.device(device)
    rows = []
    for stage in completion['stages']:
        model = restore_model(output, stage, model_path, device, model_factory)
        seen = tasks if method == 'joint' and stage else tasks[:stage + 1]
        for name in seen:
            data = store.attack(name)
            result = evaluate(model, loader(data['traces'], data['labels'], 64), device)
            logits = result.pop('logits')
            group_count = min(np.unique(data['group_id'], return_counts=True)[1]) if 'group_id' in data else len(logits)
            horizon = min(config.get('attack_horizon', 5000), int(group_count))
            metrics = rank_attack(logits, data['hypothesis_labels'], data, list(range(1, horizon + 1)),
                                  config.get('attack_orders', 200), config.get('attack_order_seed', 13))
            row = dict(stage=stage, task=name, seed=run['seed'], method=method,
                       backbone=config.get('backbone', 'qwen3'), accuracy=result['accuracy'],
                       nll=result['nll'], n_attack=persistent_recovery(metrics['guessing_entropy_zero_based']),
                       metrics=metrics)
            rows.append(row)
            destination = output / 'predictions'
            destination.mkdir(exist_ok=True)
            np.savez_compressed(destination / f'stage{stage:02d}_{name}.npz',
                                logits=logits, labels=data['labels'])
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    source = _load_checkpoint(checkpoint_path(output, 0))
    counts = dict(tokenizer_parameters=sum(value.numel() for name, value in source['parameters'].items()
                                           if name.startswith('frontend.')),
                  source_trainable_parameters=source['report']['trainable_parameters'])
    write_json(output / 'evaluation.json', dict(run=run, cells=rows, parameter_counts=counts))
    return rows
