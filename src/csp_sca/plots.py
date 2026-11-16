"""Paper-style figures generated from evaluation files, without embedded results."""
from pathlib import Path
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .analysis import nicv, load_evaluations, summarize


def _style():
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})


def plot_nicv(data_root, config, output, samples=4500, seed=13):
    _style(); values = []
    for task in config['tasks']:
        with np.load(Path(data_root) / task['name'] / 'train.npz', allow_pickle=False) as data:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(data['labels']), min(samples, len(data['labels'])), replace=False)
            values.append(100. * float(nicv(data['traces'][indices], data['labels'][indices]).max()))
    fig, ax = plt.subplots(figsize=(6.8, 2.3), layout='constrained')
    ax.bar([row['name'] for row in config['tasks']], values, color='#007f85')
    ax.set_ylabel('Peak NICV (%)'); ax.tick_params(axis='x', labelrotation=35)
    _save(fig, output)


def plot_ge(root, output):
    _style(); records = load_evaluations(root)
    cohorts = summarize(records)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(max(3.2, 3.1 * len(cohorts)), 2.7), squeeze=False, layout='constrained')
    for ax, cohort in zip(axes[0], cohorts):
        selected = [r for r in records if r['run']['config'] == cohort['configuration'] and r['run']['method'] == cohort['method']]
        for task in cohort['tasks']:
            curves = []
            for record in selected:
                stage = len(record['run']['config']['tasks']) - 1
                row = next(c for c in record['cells'] if c['stage'] == stage and c['task'] == task)
                mapping = row['metrics']['guessing_entropy_zero_based']
                curves.append(np.array([mapping[str(i)] for i in range(1, len(mapping) + 1)]))
            length = min(map(len, curves))
            ax.plot(np.arange(1, length + 1), np.mean([c[:length] for c in curves], 0), label=task, linewidth=1)
        ax.set(title=cohort['label'], xlabel='Attack traces', ylabel='Mean key-byte rank')
        ax.set_xscale('log'); ax.set_ylim(bottom=-1); ax.grid(alpha=.2)
    axes[0, -1].legend(fontsize=6, ncol=2)
    _save(fig, output)


def plot_retention(root, output, tasks=('F0', 'XMEGA', 'ASCAD-0'), drift=None):
    _style(); records = load_evaluations(root); cohorts = summarize(records)
    panels = 3 if drift is not None else 2
    fig, axes = plt.subplots(1, panels, figsize=(3.2 * panels, 2.8), layout='constrained')
    styles = ['-', '--', ':']
    for cohort in cohorts:
        selected = [r for r in records if r['run']['config'] == cohort['configuration'] and r['run']['method'] == cohort['method']]
        for task, style in zip(tasks, styles):
            if task not in cohort['tasks']:
                continue
            first = cohort['tasks'].index(task); stages = list(range(first, len(cohort['tasks'])))
            curves = []
            for record in selected:
                lookup = {(c['stage'], c['task']): c['accuracy'] for c in record['cells']}
                if all((stage, task) in lookup for stage in stages):
                    curves.append([100 * lookup[(stage, task)] for stage in stages])
            if curves:
                axes[0].plot(np.array(stages) + 1, np.mean(curves, 0), style, label=cohort['label'] + '/' + task)
        means = []
        for record in selected:
            curves = []
            for stage in range(len(cohort['tasks'])):
                values = [c['accuracy'] for c in record['cells'] if c['stage'] == stage]
                if not values:
                    break
                curves.append(100. * np.mean(values))
            if len(curves) == len(cohort['tasks']):
                means.append(curves)
        if means:
            axes[1].plot(np.arange(len(cohort['tasks'])) + 1, np.mean(means, 0), label=cohort['label'])
    axes[0].set(xlabel='Training stage', ylabel='Test ACC (%)')
    axes[1].set(xlabel='Number of learned tasks', ylabel='Mean test ACC (%)')
    axes[0].legend(fontsize=5); axes[1].legend(fontsize=6)
    if drift is not None:
        groups = json.loads(Path(drift).read_text(encoding='utf-8'))
        for method, maps in groups.items():
            by_layer = {}
            for name, values in maps.items():
                match = re.search(r'layers\.(\d+)\.', name)
                if match:
                    finite = [v for v in (values if isinstance(values, list) else [values]) if v is not None]
                    by_layer.setdefault(int(match.group(1)) + 1, []).extend(finite)
            layers = sorted(k for k, v in by_layer.items() if v)
            axes[2].plot(layers, [100 * np.median(by_layer[k]) for k in layers], label=method)
        axes[2].set(xlabel='Decoder block', ylabel='Input drift (%)'); axes[2].legend(fontsize=6)
    _save(fig, output)


def _save(fig, output):
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', metadata={'Creator': 'CSP code artifact'})
    plt.close(fig)
