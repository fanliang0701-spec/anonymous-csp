"""Command-line entry points for the anonymous paper artifact."""
import argparse
import copy
import json
from pathlib import Path
import re
import numpy as np


def read_config(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def collect_features(run_path, data_root, model_path, stage, task, device, samples, output):
    import torch
    from .data import _split
    from .engine import loader
    from .experiments import restore_model
    from .memory import layer_names
    run = read_config(Path(run_path) / 'run.json')
    config = run['config']
    names = [row['name'] for row in config['tasks']]
    if task not in names[:stage + 1]:
        raise ValueError('Choose a task learned by the selected stage')
    path = Path(data_root) / task / 'validation.npz'
    data = _split(path, config.get('trace_length', 700))
    with np.load(path, allow_pickle=False) as archive:
        ids = archive['trace_ids'][:samples].copy()
    if not len(ids):
        raise ValueError('No validation samples selected')
    model = restore_model(run_path, stage, model_path, device)
    chunks, handles = {}, []
    def collect(name, values):
        chunks.setdefault(name, []).append(values.detach().float().cpu().numpy())
    for name in layer_names(model):
        for projection in ('q_proj', 'v_proj'):
            key = name + '.' + projection
            handles.append(model.get_submodule(key).register_forward_pre_hook(
                lambda module, values, key=key: collect(key, values[0])))
    try:
        with torch.inference_mode(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16,
                                                   enabled=torch.device(device).type == 'cuda'):
            for x, _ in loader(data['traces'][:len(ids)], data['labels'][:len(ids)], 64):
                model(x.to(device))
    finally:
        for handle in handles:
            handle.remove()
    values = {name: np.concatenate(parts) for name, parts in chunks.items()}
    values.update(trace_ids=ids, _task=np.array(task), _stage=np.array(stage))
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **values)


def main(argv=None):
    parser = argparse.ArgumentParser(description='CSP continual profiled SCA')
    sub = parser.add_subparsers(dest='command', required=True)
    pack = sub.add_parser('pack', help='Validate and separate train/validation/attack bundles')
    pack.add_argument('--input', required=True); pack.add_argument('--output', required=True)
    pack.add_argument('--label-rule', required=True, choices=['first_sbox', 'last_transition'])
    check = sub.add_parser('validate-data')
    check.add_argument('--config', required=True); check.add_argument('--data', required=True)
    train = sub.add_parser('train')
    train.add_argument('--config', required=True); train.add_argument('--data', required=True)
    train.add_argument('--output', required=True); train.add_argument('--model')
    train.add_argument('--method', default='csp'); train.add_argument('--seed', type=int, default=13)
    train.add_argument('--device', default='cuda'); train.add_argument('--resume', action='store_true')
    train.add_argument('--tokens', type=int)
    attack = sub.add_parser('evaluate')
    attack.add_argument('--run', required=True); attack.add_argument('--data', required=True)
    attack.add_argument('--model'); attack.add_argument('--device', default='cuda')
    report = sub.add_parser('report')
    report.add_argument('--runs', required=True); report.add_argument('--output', required=True)
    report.add_argument('--allow-incomplete', action='store_true')
    plot = sub.add_parser('plot')
    plot.add_argument('kind', choices=['nicv', 'ge', 'retention'])
    plot.add_argument('--output', required=True); plot.add_argument('--runs')
    plot.add_argument('--config'); plot.add_argument('--data'); plot.add_argument('--drift')
    features = sub.add_parser('features')
    for arg in ['run', 'data', 'task', 'output']:
        features.add_argument('--' + arg, required=True)
    features.add_argument('--model'); features.add_argument('--stage', type=int, required=True)
    features.add_argument('--device', default='cuda'); features.add_argument('--samples', type=int, default=512)
    drift = sub.add_parser('drift')
    for arg in ['before', 'after', 'output', 'label']:
        drift.add_argument('--' + arg, required=True)
    drift.add_argument('--append', action='store_true')
    suite = sub.add_parser('suite')
    suite.add_argument('--recipe', default='configs/suites.json'); suite.add_argument('--name', required=True)
    suite.add_argument('--data'); suite.add_argument('--model'); suite.add_argument('--output', default='runs')
    suite.add_argument('--device', default='cuda'); suite.add_argument('--dry-run', action='store_true')
    suite.add_argument('--evaluate', action='store_true'); suite.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'pack':
        from .data import pack_task
        print(pack_task(args.input, args.output, args.label_rule))
    elif args.command == 'validate-data':
        from .data import TaskStore
        cfg = read_config(args.config); store = TaskStore(args.data, cfg)
        for task in cfg['tasks']:
            p, a = store.profile(task['name']), store.attack(task['name'])
            print(task['name'], len(p['train_labels']), len(p['val_labels']), len(a['labels']))
    elif args.command == 'train':
        from .experiments import train
        cfg = read_config(args.config)
        if args.tokens is not None:
            cfg['tokens'] = args.tokens
        train(cfg, args.data, args.output, args.seed, args.method, args.model, args.device, args.resume)
    elif args.command == 'evaluate':
        from .experiments import evaluate_run
        evaluate_run(args.run, args.data, args.model, args.device)
    elif args.command == 'report':
        from .analysis import load_evaluations, write_summary
        write_summary(load_evaluations(args.runs), args.output, not args.allow_incomplete)
    elif args.command == 'plot':
        from .plots import plot_nicv, plot_ge, plot_retention
        if args.kind == 'nicv':
            if not args.config or not args.data:
                parser.error('plot nicv requires --config and --data')
            plot_nicv(args.data, read_config(args.config), args.output)
        else:
            if not args.runs:
                parser.error('plot ge/retention requires --runs')
            if args.kind == 'ge':
                plot_ge(args.runs, args.output)
            else:
                plot_retention(args.runs, args.output, drift=args.drift)
    elif args.command == 'features':
        collect_features(args.run, args.data, args.model, args.stage, args.task,
                         args.device, args.samples, args.output)
    elif args.command == 'drift':
        from .analysis import feature_drift
        from .engine import write_json
        with np.load(args.before, allow_pickle=False) as before, np.load(args.after, allow_pickle=False) as after:
            values = feature_drift(dict(before), dict(after))
        output = Path(args.output)
        result = read_config(output) if args.append and output.exists() else {}
        group = result.setdefault(args.label, {})
        for key, value in values.items():
            group.setdefault(key, []).append(value)
        write_json(output, result)
    elif args.command == 'suite':
        recipe_path = Path(args.recipe)
        recipe = read_config(recipe_path)
        if args.name not in recipe:
            parser.error('Unknown suite; choose ' + ', '.join(sorted(recipe)))
        if not args.dry_run and not args.data:
            parser.error('Running a suite requires --data')
        for job in recipe[args.name]:
            if not re.fullmatch(r'[A-Za-z0-9_-]+', job['id']):
                raise ValueError('Job IDs must be portable directory names')
            cfg = read_config(recipe_path.parent / job['config'])
            cfg.update(copy.deepcopy(job.get('overrides', {})))
            for seed in cfg['seeds']:
                name = job['id'] + '-seed' + str(seed)
                if args.dry_run:
                    print(name, cfg['backbone'], job['method'], len(cfg['tasks']), 'tasks', cfg['tokens'], 'tokens')
                    continue
                from .experiments import train, evaluate_run
                output = Path(args.output) / args.name / name
                train(cfg, args.data, output, seed, job['method'], args.model, args.device, args.resume)
                if args.evaluate:
                    evaluate_run(output, args.data, args.model, args.device)


if __name__ == '__main__':
    main()
