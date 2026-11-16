import numpy as np
import pytest

from csp_sca.sca_suite_metrics import key_rank_metrics


def test_perfect_predictions_recover_key_for_arbitrary_label_mapping():
    rng = np.random.default_rng(13)
    hypotheses = np.array([rng.permutation(256) for _ in range(100)], dtype=np.uint8)
    logits = np.zeros((100, 256))
    logits[np.arange(100), hypotheses[:, 71]] = 8
    result = key_rank_metrics(logits, hypotheses, np.full(100, 71), counts=(1, 10, 100))
    assert result['success_rate'] == {'1': 1., '10': 1., '100': 1.}
    assert result['guessing_entropy_zero_based'] == {'1': 0., '10': 0., '100': 0.}
    assert result['n90'] == 1


def test_uniform_predictions_do_not_count_ties_as_success():
    hypotheses = np.tile(np.arange(256, dtype=np.uint8), (50, 1))
    result = key_rank_metrics(np.zeros((50, 256)), hypotheses, np.zeros(50), counts=(1, 50))
    assert result['guessing_entropy_zero_based'] == {'1': 127.5, '50': 127.5}
    assert result['success_rate'] == {'1': 0., '50': 0.}
    assert result['n90'] is None


def test_repeated_attacks_are_reproducible_and_reject_mixed_keys():
    rng = np.random.default_rng(13)
    logits = rng.normal(size=(100, 256))
    hypotheses = np.tile(np.arange(256), (100, 1))
    keys = np.zeros(100, dtype=np.uint8)
    assert key_rank_metrics(logits, hypotheses, keys) == key_rank_metrics(logits, hypotheses, keys)
    keys[0] = 1
    with pytest.raises(ValueError, match='fixed key'):
        key_rank_metrics(logits, hypotheses, keys)


@pytest.mark.parametrize('bad', [np.full((50, 256), 256), np.full((50, 256), .5)])
def test_invalid_hypothesis_labels_rejected(bad):
    with pytest.raises(ValueError):
        key_rank_metrics(np.zeros((50, 256)), bad, np.zeros(50), counts=(1,))
