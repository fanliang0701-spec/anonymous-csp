"""NICV and the recovery/retention summaries used by the paper tables."""
from pathlib import Path
import csv
import json
import numpy as np
from .engine import write_json


def nicv(traces, labels):
    x, y = np.asarray(traces, dtype=np.float64), np.asarray(labels)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x) or not np.isfinite(x).all():
        raise ValueError('NICV requires finite aligned traces and labels')
    mean, variance = x.mean(0), x.var(0)
    between = np.zeros(x.shape[1], dtype=np.float64)
    for label, count in zip(*np.unique(y, return_counts=True)):
        between += count / len(y) * (x[y == label].mean(0) - mean) ** 2
    values = np.divide(between, variance, out=np.zeros_like(between), where=variance > 0)
    if (values < -1e-10).any() or (values > 1 + 1e-10).any():
        raise FloatingPointError('NICV exceeded its numerical range')
    return values


def load_evaluations(root):
    root = Path(root)
    paths = [root] if root.is_file() else sorted(root.rglob('evaluation.json'))
    if not paths:
        raise FileNotFoundError('No evaluation.json files were found')
    return [json.loads(path.read_text(encoding='utf-8')) for path in paths]


def summarize(records, require_complete=True):
    groups = {}
    for record in records:
        run = record['run']
        identity = json.dumps(dict(config=run['config'], method=run['method']), sort_keys=True)
        groups.setdefault(identity, []).append(record)
    output = []
    for group in groups.values():
        first = group[0]['run']; config = first['config']
        seeds = [r['run']['seed'] for r in group]
        if len(set(seeds)) != len(seeds):
            raise ValueError('Duplicate seed in an experiment cohort')
        if require_complete and set(seeds) != set(config.get('seeds', [13, 42, 2026])):
            raise ValueError('The requested seed cohort is incomplete')
        tasks = [row['name'] for row in config['tasks']]
        subset = config.get('accuracy_tasks', tasks)
        costs = {task: [] for task in tasks}
        acc, ratios, losses, maxima = [], [], [], []
        successes, cases = 0, 0
        for record in group:
            cells = record['cells']
            lookup = {(row['stage'], row['task']): row for row in cells}
            if len(lookup) != len(cells):
                raise ValueError('Duplicate task/stage cell')
            final_stage = len(tasks) - 1
            final = [lookup[(final_stage, task)] for task in tasks]
            cases += len(final)
            successes += sum(row['n_attack'] is not None for row in final)
            if all(row['n_attack'] is not None for row in final):
                maxima.append(max(row['n_attack'] for row in final))
            for index, task in enumerate(tasks):
                row = final[index]
                if row['n_attack'] is not None:
                    costs[task].append(float(row['n_attack']))
                if task in subset:
                    acc.append(100. * row['accuracy'])
                acquisition = lookup.get((index, task))
                if first['method'] != 'joint' and acquisition is not None and index < final_stage:
                    losses.append(100. * (acquisition['accuracy'] - row['accuracy']))
                if (first['method'] != 'joint' and task in subset and acquisition is not None and
                        acquisition['n_attack'] is not None and row['n_attack'] is not None):
                    ratios.append(row['n_attack'] / acquisition['n_attack'])
        output.append(dict(
            label=config.get('label', config.get('backbone', 'qwen3') + '/' + first['method']),
            method=first['method'], backbone=config.get('backbone', 'qwen3'),
            configuration=config,
            tokenizer_parameters_m=(float(np.mean([r['parameter_counts']['tokenizer_parameters'] for r in group])) / 1e6
                                    if all('parameter_counts' in r for r in group) else None),
            seeds=sorted(seeds), tasks=tasks, tokens=config.get('tokens', 8),
            recovered_cases=successes, evaluated_cases=cases, recovery_percent=100. * successes / cases,
            final_accuracy_percent=float(np.mean(acc)),
            attack_cost_inflation=float(np.exp(np.log(ratios).mean())) if ratios else None,
            inflation_cases=len(ratios),
            mean_accuracy_loss_pp=float(np.mean(losses)) if losses else None,
            maximum_final_n_attack_mean=float(np.mean(maxima)) if len(maxima) == len(group) else None,
            task_costs={task: dict(mean=float(np.mean(values)) if values else None,
                                    recovered=len(values), cases=len(group)) for task, values in costs.items()}))
    return output


def write_summary(records, output, require_complete=True):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    rows = summarize(records, require_complete)
    write_json(output / 'summary.json', rows)
    fields = ['label', 'backbone', 'method', 'tokens', 'tokenizer_parameters_m', 'recovered_cases', 'evaluated_cases',
              'recovery_percent', 'final_accuracy_percent', 'attack_cost_inflation',
              'inflation_cases', 'mean_accuracy_loss_pp', 'maximum_final_n_attack_mean']
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    with (output / 'task_costs.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['label', 'task', 'mean', 'recovered', 'cases'])
        writer.writeheader()
        for row in rows:
            for task, value in row['task_costs'].items():
                writer.writerow(dict(label=row['label'], task=task, **value))
    return rows


def feature_drift(before, after):
    if set(before) != set(after) or 'trace_ids' not in before:
        raise ValueError('Feature archives must have identical maps and trace IDs')
    if not np.array_equal(before['trace_ids'], after['trace_ids']):
        raise ValueError('Feature comparisons require the same traces in the same order')
    if '_task' in before and not np.array_equal(before['_task'], after['_task']):
        raise ValueError('Feature archives refer to different tasks')
    result = {}
    for name in before:
        if name == 'trace_ids' or name.startswith('_'):
            continue
        x, y = np.asarray(before[name], dtype=np.float64), np.asarray(after[name], dtype=np.float64)
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Feature shapes or values are invalid')
        denominator = np.linalg.norm(x)
        result[name] = float(np.linalg.norm(y - x) / denominator) if denominator else None
    return result
