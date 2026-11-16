"""CPU integration checks use reduced dimensions, not paper benchmark results."""
import copy
import json
import numpy as np
import pytest
import torch
from csp_sca.aes import SBOX
from csp_sca.models import OrdinaryTransformer, WideMLP
from csp_sca.experiments import train, evaluate_run, restore_model
from csp_sca.analysis import summarize


def small_config():
    return dict(backbone='transformer', trace_length=8, tokens=2, rank=2, epochs=1,
                seeds=[13], anchors_per_class=1, attack_horizon=8, attack_orders=3,
                encoder=dict(hidden_size=32, layers=1, heads=4, feedforward_size=48),
                memory_cap=4, replay_capacity=256, ewc_lambda=1.,
                tasks=[dict(name=name, label_rule='first_sbox') for name in ['F0', 'XMEGA']])


def prepare_tasks(root):
    for task_index, name in enumerate(['F0', 'XMEGA']):
        folder = root / name; folder.mkdir(parents=True)
        rng = np.random.default_rng(task_index)
        labels = np.arange(256, dtype=np.int64)
        for role in ['train', 'validation']:
            np.savez(folder / (role + '.npz'), traces=rng.normal(size=(256, 8)).astype(np.float32),
                     labels=labels, trace_ids=np.arange(256) + (0 if role == 'train' else 256))
        plaintext = np.arange(16, dtype=np.uint8)
        np.savez(folder / 'attack.npz', traces=rng.normal(size=(16, 8)).astype(np.float32),
                 labels=SBOX[plaintext ^ 42], plaintext=plaintext, key_byte=np.array(42, np.uint8),
                 trace_ids=np.arange(512, 528))


@pytest.mark.parametrize('method', ['csp', 'unprotected', 'gpm', 'olora', 'replay', 'ewc',
                                    'inflora', 'inflora_m', 'mean_fixed_a', 'cov_ab', 'csp_cap', 'joint'])
def test_two_stage_training_resume_and_attack(method, tmp_path):
    torch.set_num_threads(2)
    cfg = small_config(); prepare_tasks(tmp_path / 'data')
    if method == 'inflora_m':
        cfg['encoder'].update(hidden_size=64, feedforward_size=96)
    output = tmp_path / 'run'
    train(cfg, tmp_path / 'data', output, 13, method, device='cpu')
    first = restore_model(output, 1, device='cpu')
    probe = torch.randn(2, 8)
    with torch.no_grad(): expected = first(probe)
    train(cfg, tmp_path / 'data', output, 13, method, device='cpu', resume=True)
    second = restore_model(output, 1, device='cpu')
    with torch.no_grad(): torch.testing.assert_close(second(probe), expected, rtol=0, atol=0)
    rows = evaluate_run(output, tmp_path / 'data', device='cpu')
    assert {(row['stage'], row['task']) for row in rows} == {(0, 'F0'), (1, 'F0'), (1, 'XMEGA')}
    record = json.loads((output / 'evaluation.json').read_text())
    assert summarize([record])[0]['evaluated_cases'] == 2
    assert summarize([record])[0]['tokenizer_parameters_m'] > 0
    wrong = copy.deepcopy(cfg); wrong['tokens'] = 3
    with pytest.raises(ValueError, match='configuration differs'):
        train(wrong, tmp_path / 'data', output, 13, method, device='cpu', resume=True)


def test_wide_mlp_uses_both_linear_maps_and_restores_adapters(tmp_path):
    torch.set_num_threads(2)
    cfg = small_config(); cfg.update(backbone='mlp', mlp=dict(width=12, blocks=2))
    prepare_tasks(tmp_path / 'data')
    train(cfg, tmp_path / 'data', tmp_path / 'run', 13, 'csp', device='cpu')
    model = restore_model(tmp_path / 'run', 1, device='cpu')
    assert all(hasattr(layer, 'basis') for layer in list(model.fc1) + list(model.fc2))
    assert model(torch.randn(3, 8)).shape == (3, 256)


def test_parameter_matched_inflora_trains_only_b(tmp_path):
    from csp_sca.controls import install_budget_inflora
    from csp_sca.memory import layer_names
    model = OrdinaryTransformer(13, trace_length=8, tokens=2, hidden_size=32, layers=2, heads=4, feedforward_size=48)
    designs = {name: torch.eye(32)[:19] / 3**.5 for name in layer_names(model)}
    adapters = install_budget_inflora(model, designs)
    assert {name.rsplit('.', 1)[-1] for name, p in model.named_parameters() if p.requires_grad} == {'B'}
    assert all(a.A.shape[0] == (19 if name.endswith('q_proj') else 18) for name, a in adapters.items())
