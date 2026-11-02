"""Run-level metrics and honest inferential summaries; AP/mAP intentionally unavailable."""
from __future__ import annotations
import numpy as np
import pandas as pd

METRICS=('latency_mean_ms','latency_p95_ms','deadline_violation_pct','drop_pct',
         'offload_pct','system_energy_kj','vehicle_energy_kj','accuracy_proxy_pct',
         'xai_invocation_pct','mean_queue_ms')

def aggregate_run(df:pd.DataFrame) -> dict:
    if df.empty: raise ValueError('No task outcomes')
    return {'n_tasks':len(df),
        'latency_mean_ms':float(df.latency_ms.mean()),
        'latency_p95_ms':float(df.latency_ms.quantile(0.95)),
        'deadline_violation_pct':float(100*df.deadline_violation.mean()),
        'drop_pct':float(100*df.dropped.mean()),
        'offload_pct':float(100*df.offloaded.mean()),
        'system_energy_kj':float(df.system_j.sum()/1000),
        'vehicle_energy_kj':float(df.vehicle_j.sum()/1000),
        'system_energy_j_per_task':float(df.system_j.mean()),
        'accuracy_proxy_pct':float(df.accuracy_proxy_pct.mean()),
        'xai_invocation_pct':float(100*df.xai_invoked.mean()),
        'mean_queue_ms':float(df.queue_ms.mean()),
        'total_reward':float(df.reward.sum()),
        'actual_map_50':'NOT_COMPUTED_missing_ground_truth_and_real_detections'}

def summarize_runs(runs:pd.DataFrame):
    """mean±SD computed across independent run rows (not across tasks)."""
    if runs.empty: return pd.DataFrame()
    return runs.groupby('policy',sort=False)[list(METRICS)].agg(['mean','std']).round(4)

def paired_comparison(runs:pd.DataFrame,proposed='ppo',baseline='energy_fixed'):
    """Wilcoxon on paired seed values, signed effect magnitude; >=~10 seeds preferable."""
    from scipy.stats import wilcoxon
    a=runs[runs.policy==proposed].set_index('seed')
    b=runs[runs.policy==baseline].set_index('seed')
    seeds=sorted(set(a.index)&set(b.index))
    if len(seeds)<2: return {'error':'At least two paired seeds required'}
    out={'proposed':proposed,'baseline':baseline,'paired_n':len(seeds)}
    for k in METRICS:
        x=a.loc[seeds,k].to_numpy(dtype=float)
        y=b.loc[seeds,k].to_numpy(dtype=float)
        delta=x-y
        p=float(wilcoxon(delta,zero_method='wilcox').pvalue) if np.any(delta!=0) else 1.0
        out[k]={'mean_difference_proposed_minus_baseline':float(delta.mean()),
                'wilcoxon_p_two_sided':p,
                'paired_effect_dz':float(delta.mean()/delta.std(ddof=1)) if len(delta)>1 and delta.std(ddof=1)>0 else None}
    return out
