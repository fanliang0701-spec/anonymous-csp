import copy
import numpy as np
import pytest
from csp_sca.aes import SBOX, INV_SBOX
from csp_sca.data import hypotheses, pack_task, TaskStore
from csp_sca.analysis import nicv, feature_drift, summarize


def test_last_round_hypotheses_include_the_transition_byte():
    ciphertext = np.arange(64, dtype=np.uint8).reshape(4, 16)
    actual = hypotheses({'ciphertext': ciphertext}, 'last_transition')
    for index in range(4):
        for key in [0, 42, 255]:
            assert actual[index, key] == INV_SBOX[ciphertext[index, 11] ^ key] ^ ciphertext[index, 7]


def raw_bundle():
    values = {}
    for prefix, offset in [('train', 0), ('val', 10), ('attack', 20)]:
        values[prefix + '_traces'] = np.arange(32, dtype=np.float32).reshape(4, 8)
        values[prefix + '_labels'] = SBOX[np.arange(4, dtype=np.uint8) ^ 42]
        values[prefix + '_ids'] = np.arange(offset, offset + 4)
    values.update(attack_plaintext=np.arange(4, dtype=np.uint8), attack_key_byte=np.array(42, np.uint8))
    return values


def test_pack_rejects_overlap_and_validates_attack_labels(tmp_path):
    data = raw_bundle(); data['attack_ids'][0] = data['train_ids'][0]
    np.savez(tmp_path / 'overlap.npz', **data)
    with pytest.raises(ValueError, match='overlap'):
        pack_task(tmp_path / 'overlap.npz', tmp_path / 'out', 'first_sbox')
    data = raw_bundle(); data['attack_labels'][0] ^= 1
    np.savez(tmp_path / 'wrong.npz', **data)
    with pytest.raises(ValueError, match='labels disagree'):
        pack_task(tmp_path / 'wrong.npz', tmp_path / 'out', 'first_sbox')


def test_profile_access_does_not_require_or_open_attack_file(tmp_path):
    np.savez(tmp_path / 'raw.npz', **raw_bundle())
    pack_task(tmp_path / 'raw.npz', tmp_path / 'data/F0', 'first_sbox')
    store = TaskStore(tmp_path / 'data', {'trace_length': 8, 'tasks': [{'name': 'F0', 'label_rule': 'first_sbox'}]})
    assert store.attack('F0')['hypothesis_labels'].shape == (4, 256)
    (tmp_path / 'data/F0/attack.npz').unlink()
    assert store.profile('F0')['train_traces'].shape == (4, 8)


def test_nicv_detects_label_information_and_constant_columns():
    labels = np.repeat([0, 1], 4)
    traces = np.column_stack([labels, np.ones(8), np.tile([0, 1, 0, 1], 2)])
    np.testing.assert_allclose(nicv(traces, labels), [1, 0, 0])


def test_feature_drift_requires_matching_samples_and_tasks():
    before = {'trace_ids': np.array([1, 2]), '_task': np.array('F0'), 'map': np.ones((2, 3))}
    after = copy.deepcopy(before); after['map'] *= 2
    assert feature_drift(before, after)['map'] == pytest.approx(1.)
    after['trace_ids'] = np.array([2, 1])
    with pytest.raises(ValueError, match='same traces'):
        feature_drift(before, after)


def test_summary_uses_conditional_costs_and_does_not_fill_failures():
    cfg = {'tasks': [{'name': 'F0'}, {'name': 'XMEGA'}], 'seeds': [13, 42], 'accuracy_tasks': ['F0']}
    records = []
    for seed, cost in [(13, 20), (42, None)]:
        records.append({'run': {'config': cfg, 'method': 'csp', 'seed': seed}, 'cells': [
            {'stage': 0, 'task': 'F0', 'n_attack': 10, 'accuracy': .8},
            {'stage': 1, 'task': 'F0', 'n_attack': cost, 'accuracy': .6},
            {'stage': 1, 'task': 'XMEGA', 'n_attack': 5, 'accuracy': .5}]})
    row = summarize(records)[0]
    assert row['recovery_percent'] == 75
    assert row['task_costs']['F0'] == {'mean': 20., 'recovered': 1, 'cases': 2}
    assert row['attack_cost_inflation'] == pytest.approx(2.)
    assert row['mean_accuracy_loss_pp'] == pytest.approx(20.)
    assert row['maximum_final_n_attack_mean'] is None
    with pytest.raises(ValueError, match='cohort'):
        summarize(records[:1])
