"""Repeatable paired trials, per-task CSV, per-run statistics, manifest with provenance."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .config import Config
from .controllers import controller_from_name
from .simulator import GreenCAVEnv
from .analysis import aggregate_run,paired_comparison,summarize_runs

DEFAULT_POLICIES=('local','edge','greedy','deadline','energy_fixed','proposed_heuristic')

def run_one(cfg:Config,policy:str,seed:int,checkpoint=None):
    env=GreenCAVEnv(cfg,seed)
    controller=controller_from_name(policy,seed,checkpoint)
    obs=env.observe()
    outcomes=[]
    for _ in range(cfg.n_tasks):
        action=controller.act(env,obs)
        obs,reward,done,out=env.step(action)
        outcomes.append(out.to_dict())
        if done: break
    return pd.DataFrame.from_records(outcomes)

def run_suite(cfg:Config,outdir:str,policies=DEFAULT_POLICIES,seeds=3,checkpoint=None,save_tasks=True):
    folder=Path(outdir);folder.mkdir(parents=True,exist_ok=True)
    runs=[]
    for seed_offset in range(seeds):
        seed=cfg.seed+seed_offset
        for pol in policies:
            df=run_one(cfg,pol,seed,checkpoint)
            metric=aggregate_run(df)
            metric.update({'seed':seed,'policy':pol})
            runs.append(metric)
            if save_tasks:
                df.to_csv(folder/f'tasks_{pol}_seed{seed}.csv',index=False)
            print(f'{pol:20s} seed={seed} latency={metric["latency_mean_ms"]:.2f}ms '
                  f'energy={metric["system_energy_j_per_task"]:.3f}J/task '
                  f'drops={metric["drop_pct"]:.2f}%')
    table=pd.DataFrame(runs)
    table.to_csv(folder/'run_metrics.csv',index=False)
    summary=summarize_runs(table)
    summary.to_csv(folder/'summary_mean_sd.csv')
    comparison={}
    if 'ppo' in policies and 'energy_fixed' in policies:
        comparison=paired_comparison(table,'ppo','energy_fixed')
    elif 'proposed_heuristic' in policies and 'energy_fixed' in policies:
        comparison=paired_comparison(table,'proposed_heuristic','energy_fixed')
    (folder/'paired_tests.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    (folder/'manifest.json').write_text(json.dumps(cfg.manifest()|{'policies':list(policies),'seeds':seeds,
        'checkpoint':str(checkpoint) if checkpoint else None,
        'result_provenance':'NEW MODELED SYNTHETIC OUTPUT; not the manuscript reported data'},indent=2),encoding='utf-8')
    return table

def run_study(cfg:Config,kind:str,outdir:str,seeds=3,policy='proposed_heuristic',checkpoint=None):
    """Controlled sensitivity/ablation sweeps with independent scenario provenance.

    Each variant is rerun under matching seed IDs; modifications are logged.
    A separate CSV compares run means and SD; data are synthetic, not paper tables.
    """
    from dataclasses import replace
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    variants=[]
    if kind=='weather':
        variants=[(w,{'scenario':w}) for w in ('clear','light_rain','heavy_rain','dense_smog')]
    elif kind=='uplink':
        variants=[(f'{r}Mbps',{'link_rates_mbps':(r,r,r)}) for r in (5.,15.,30.)]
    elif kind=='density':
        variants=[(f'{n}vehicles',{'n_vehicles':n}) for n in (2,5,8,10)]
    elif kind=='preference':
        variants=[(f'alpha{a:.1f}',{'alpha':a}) for a in (0.2,0.5,0.8)]
    elif kind=='ablation':
        variants=[('full',{}),('without_xai',{'enable_xai':False,'feedback_enabled':False}),
                  ('without_feedback',{'feedback_enabled':False}),
                  ('without_energy_objective',{'alpha':0.0}),
                  ('fixed_YOLOv8s',{'fixed_detector':1})]
    else: raise ValueError('Unknown study '+kind)
    rows=[]
    for variant,changes in variants:
        subcfg=replace(cfg,**changes)
        subcfg.validate()
        subcfg.save(out/f'config_{variant}.yaml')
        for k in range(seeds):
            seed=cfg.seed+k
            df=run_one(subcfg,policy,seed,checkpoint)
            m=aggregate_run(df)
            rows.append(m|{'seed':seed,'variant':variant,'policy':policy,'study':kind})
        print(f'Study {kind}: {variant} complete ({seeds} seeds)')
    results=pd.DataFrame(rows)
    results.to_csv(out/f'{kind}_runs.csv',index=False)
    summary=results.groupby('variant',sort=False)[['latency_mean_ms','system_energy_j_per_task',
        'deadline_violation_pct','offload_pct','accuracy_proxy_pct','xai_invocation_pct']].agg(['mean','std'])
    summary.to_csv(out/f'{kind}_summary.csv')
    (out/'manifest.json').write_text(json.dumps(cfg.manifest()|{'study':kind,'policy':policy,
        'variant_changes':{label:change for label,change in variants},
        'warning':'synthetic assumptions and surrogate perception; NOT actual CARLA-SUMO or real mAP'},indent=2),encoding='utf-8')
    return results
