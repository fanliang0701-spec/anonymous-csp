"""Fixed-key and grouped key-ranking metrics from the experiment code."""
import numpy as np
from .sca_suite_metrics import key_rank_metrics

def persistent_recovery(curve):
    budgets=sorted(map(int,curve));last=max((int(n) for n,v in curve.items() if v!=0),default=0)
    return next((n for n in budgets if n>last),None)


def rank_attack(logits,hypotheses,data,counts,repeats=200,seed=13):
    if 'group_id' not in data:
        return key_rank_metrics(logits,hypotheses,data['key_byte'],counts,repeats,seed)
    ids=np.asarray(data['group_id'])
    if ids.shape!=(len(logits),) or ids.dtype.kind not in 'iu':raise ValueError('Require one integer group ID per trace')
    groups=[]
    for group in np.unique(ids):
        chosen=ids==group
        if max(counts)>int(chosen.sum()):raise ValueError('Group has fewer traces than evaluation budget')
        metric=key_rank_metrics(logits[chosen],hypotheses[chosen],data['key_byte'][chosen],counts,repeats,seed)
        groups.append(dict(group_id=int(group),key_byte=int(data['key_byte'][chosen][0]),samples=int(chosen.sum()),
            accuracy=float(np.mean(np.argmax(logits[chosen],axis=1)==data['labels'][chosen])),
            n_ge0=persistent_recovery(metric['guessing_entropy_zero_based']),metrics=metric))
    result={field:{str(n):float(np.mean([g['metrics'][field][str(n)] for g in groups])) for n in counts}
            for field in ['guessing_entropy_zero_based','strictly_better_key_count','success_rate']}
    result.update(groups=groups,groups_count=len(groups),groups_recovered=sum(g['n_ge0'] is not None for g in groups),
        aggregation='equal-weight mean of independently ranked full-master-key groups; no cross-key accumulation',
        rank_tie_policy=groups[0]['metrics']['rank_tie_policy'],trace_counts=list(counts),repeats=repeats,seed=seed,
        interval_scope='Wilson intervals apply within each group only; no pooled interval',
        tge=next((n for n in counts if result['guessing_entropy_zero_based'][str(n)]<1),None),
        n90=next((n for i,n in enumerate(counts) if all(result['success_rate'][str(m)]>=.9 for m in counts[i:])),None))
    budget=persistent_recovery(result['guessing_entropy_zero_based'])
    expected=max(g['n_ge0'] for g in groups) if result['groups_recovered']==len(groups) else None
    if budget!=expected:raise ValueError('Macro recovery differs from all-group recovery')
    return result
