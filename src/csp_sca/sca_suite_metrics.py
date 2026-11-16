"""Repeated key-ranking attacks for explicit, dataset-specific label hypotheses."""
import numpy as np


def key_rank_metrics(logits, hypothesis_labels, key_byte, counts=None, repeats=200, seed=13):
    logits = np.asarray(logits, dtype=np.float64)
    labels, keys = np.asarray(hypothesis_labels), np.asarray(key_byte)
    if logits.ndim != 2 or logits.shape[1] != 256 or not len(logits) or not np.isfinite(logits).all():
        raise ValueError('logits must be finite [N,256]')
    for value in (labels, keys):
        if (value.dtype.kind not in 'iuf' or not np.isfinite(value).all()
                or (value < 0).any() or (value > 255).any() or (value != np.floor(value)).any()):
            raise ValueError('hypotheses and keys must contain byte integers')
    if labels.shape != logits.shape or keys.shape != (len(logits),):
        raise ValueError('hypotheses/key shapes differ from logits')
    if np.unique(keys).size != 1:
        raise ValueError('each attack group requires a fixed key')
    counts = tuple(counts) if counts is not None else tuple(
        n for n in [*range(1, 101), 200, 500, 1000] if n <= len(logits))
    if (not counts or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
                         or not 1 <= n <= len(logits) for n in counts)
            or tuple(sorted(set(counts))) != counts):
        raise ValueError('counts must be increasing valid trace counts')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be positive integer')
    # The log-softmax normalizer is identical across key candidates for each trace.
    scores = np.take_along_axis(logits - logits.max(axis=1, keepdims=True), labels.astype(np.int64), axis=1)
    rng = np.random.default_rng(seed)
    ranks, strict_ranks, success = np.zeros((3, len(counts)))
    positions = np.asarray(counts) - 1
    for _ in range(repeats):
        order = rng.permutation(len(scores))[:counts[-1]]
        totals = np.cumsum(scores[order], axis=0)[positions]
        if not np.isfinite(totals).all():
            raise ValueError('nonfinite cumulative scores')
        truth = totals[:, int(keys[0]), None]
        better, ties = (totals > truth).sum(axis=1), (totals == truth).sum(axis=1)
        ranks += better + (ties - 1) / 2
        strict_ranks += better
        success += (better == 0) & (ties == 1)
    ranks, strict_ranks, success = ranks / repeats, strict_ranks / repeats, success / repeats
    z = 1.959963984540054
    den = 1 + z*z/repeats
    center = (success + z*z/(2*repeats)) / den
    radius = z * np.sqrt(success*(1-success)/repeats + z*z/(4*repeats**2)) / den
    return {
        'guessing_entropy_zero_based': dict(zip(map(str, counts), ranks.tolist())),
        'rank_tie_policy': 'average tied ranks; success requires a unique correct winner',
        'strictly_better_key_count': dict(zip(map(str, counts), strict_ranks.tolist())),
        'success_rate': dict(zip(map(str, counts), success.tolist())),
        'success_rate_mc_wilson95': {str(n): [float(max(0, c-r)), float(min(1, c+r))]
                                    for n, c, r in zip(counts, center, radius)},
        'interval_scope': 'Monte Carlo attack-order uncertainty on this fixed pool, not population uncertainty',
        'n90': next((int(n) for i, n in enumerate(counts) if np.all(success[i:] >= .9)), None),
        'tge': next((int(n) for n, ge in zip(counts, ranks) if ge < 1), None),
        'trace_counts': list(map(int, counts)), 'repeats': repeats, 'seed': seed,
    }
