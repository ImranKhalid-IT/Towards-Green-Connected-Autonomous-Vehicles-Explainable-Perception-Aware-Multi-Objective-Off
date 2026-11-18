"""End-to-end experiment pipeline: train every learned variant, evaluate all
policies with paired seeds and write the paper's tables and figures.

    python scripts/run_all.py --quick          # minutes, small budget
    python scripts/run_all.py                  # paper budget (2e6 steps, 30 x 30,000 tasks)
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import SimConfig, load_config
from .evaluate import (algorithmS2_evaluate, table_policies, table_stats, table_variants, table_weather,
                       write_table)
from .policies import (VARIANT_ALPHA, VARIANT_NAMES, DeadlineAware, FixedSite, Greedy, PPOPolicy,
                       variant_config)
from .ppo import PPOAgent, algorithm2_train

VARIANTS = ["proposed", "drl", "no_xai", "no_energy", "no_dyn"]


def train_variants(cfg: SimConfig, out_dir: str, steps: int, variants=VARIANTS, seed: int = 0, n_envs: int = 4,
                   log=print) -> Dict[str, PPOAgent]:
    os.makedirs(out_dir, exist_ok=True)
    agents = {}
    for v in variants:
        path = os.path.join(out_dir, f"{v}.npz")
        vcfg = variant_config(cfg, v)
        if os.path.exists(path):
            log(f"[train] loading {path}")
            agents[v] = PPOAgent.load(path, vcfg)
            continue
        log(f"[train] variant={v} steps={steps:,d}")
        agent, tl = algorithm2_train(vcfg, steps=steps, n_envs=n_envs, seed=seed, fixed_alpha=VARIANT_ALPHA.get(v), log_fn=log)
        agent.save(path, meta={"variant": v, "steps": steps})
        pd.DataFrame({"steps": tl.steps, "mean_cost": tl.mean_cost, "entropy": tl.entropy}).to_csv(
            os.path.join(out_dir, f"{v}_trainlog.csv"), index=False)
        agents[v] = agent
    return agents


def baseline_policies(cfg: SimConfig, agents: Dict[str, PPOAgent]):
    return [FixedSite(cfg, 0, "YOLOv8s"), FixedSite(cfg, 1, "YOLOv8s"), Greedy(cfg), DeadlineAware(cfg),
            PPOPolicy(agents["drl"], variant_config(cfg, "drl"), "Energy-aware DRL"),
            PPOPolicy(agents["proposed"], cfg, "Proposed")]


def run_all(cfg: Optional[SimConfig] = None, out: str = "results", steps: Optional[int] = None,
            runs: Optional[int] = None, tasks: Optional[int] = None, workers: int = 1, seed0: int = 0,
            sweeps: bool = True, log=print):
    cfg = cfg or load_config()
    steps = steps or cfg.train_steps
    runs = runs or cfg.n_runs
    tasks = tasks or cfg.tasks_per_run
    seeds = list(range(seed0, seed0 + runs))
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    # ---- Algorithm 2 for every learned variant ---------------------------------------
    agents = train_variants(cfg, os.path.join(out, "models"), steps, log=log)

    # ---- Table 4 / S5 / S7: policy comparison ------------------------------------------
    pols = baseline_policies(cfg, agents)
    runs_df, weather_df = algorithmS2_evaluate(pols, seeds, tasks, workers=workers)
    runs_df.to_csv(os.path.join(out, "runs_policies.csv"), index=False)
    weather_df.to_csv(os.path.join(out, "runs_policies_weather.csv"), index=False)
    write_table(table_policies(runs_df), os.path.join(out, "table4_policy_comparison"))
    write_table(table_weather(weather_df, "Proposed"), os.path.join(out, "table5_weather"))
    write_table(table_stats(runs_df, "Proposed", ["Energy-aware DRL", "Deadline-aware"]), os.path.join(out, "tableS7_statistics"))
    veh = runs_df.groupby("policy", sort=False)[["Wh_h", "veh_Wh_h", "veh_share_pct", "range_loss_km", "co2_g_km"]].mean().round(3)
    write_table(veh, os.path.join(out, "tableS5_vehicle_level"))
    log(f"[eval] policies done ({time.time() - t0:.0f}s)")

    # ---- Table 6: XAI overhead measured in the simulator ---------------------------------
    prop = runs_df[runs_df.policy == "Proposed"]
    xai = pd.DataFrame({"Cost/call (ms)": [cfg.xai_call_s * 1e3],
                        "Realised invocation (%)": [prop.xai_share_pct.mean()],
                        "Amortised overhead (ms/task)": [prop.xai_ms_task.mean()],
                        "Share of mean latency (%)": [100 * prop.xai_ms_task.mean() / prop.latency_ms.mean()]},
                       index=["Dual-head XAI (simulated)"]).round(2)
    write_table(xai, os.path.join(out, "table6_xai_overhead"))

    # ---- Table 7: preference modes and ablations ----------------------------------------
    var_pols = [PPOPolicy(agents["proposed"], cfg, "Full, balanced (alpha=0.5)", alpha=0.5),
                PPOPolicy(agents["proposed"], cfg, "Energy-saving (alpha=0.8)", alpha=0.8),
                PPOPolicy(agents["proposed"], cfg, "Latency-priority (alpha=0.2)", alpha=0.2)]
    for v in ["no_xai", "no_energy", "no_dyn"]:
        var_pols.append(PPOPolicy(agents[v], variant_config(cfg, v), VARIANT_NAMES[v], alpha=VARIANT_ALPHA.get(v)))
    var_df, _ = algorithmS2_evaluate(var_pols, seeds, tasks, workers=workers)
    var_df.to_csv(os.path.join(out, "runs_variants.csv"), index=False)
    write_table(table_variants(var_df, "Full, balanced (alpha=0.5)"), os.path.join(out, "table7_modes_ablation"))
    log(f"[eval] variants done ({time.time() - t0:.0f}s)")

    if sweeps:
        # ---- Supplementary Table S3: uplink-rate sweep --------------------------------------
        rows = []
        p = PPOPolicy(agents["proposed"], cfg, "Proposed")
        for s, rate in enumerate(cfg.uplink_rates_mbps):
            df, _ = algorithmS2_evaluate([p], seeds, tasks, overrides={"forced_link_state": s}, workers=workers)
            r = df.mean(numeric_only=True)
            rows.append({"Rate (Mbit/s)": rate, "Offload (%)": r.offload_pct, "Mean latency (ms)": r.latency_ms,
                         "Viol. (%)": r.viol_pct, "Drop (%)": r.drop_pct, "J/task": r.J_task})
        write_table(pd.DataFrame(rows).set_index("Rate (Mbit/s)").round(2), os.path.join(out, "tableS3_uplink"))
        # ---- Supplementary Table S4: traffic-density sweep ----------------------------------
        rows = []
        for regime, load in cfg.traffic_loads.items():
            df, _ = algorithmS2_evaluate([p], seeds, tasks, overrides={"traffic_regime": regime}, workers=workers)
            r = df.mean(numeric_only=True)
            succ = 1000 * (1 - (r.drop_pct + r.viol_pct) / 100)
            rows.append({"Regime": regime, "Edge load (%)": 100 * load, "Mean latency (ms)": r.latency_ms,
                         "Viol. (%)": r.viol_pct, "Drop (%)": r.drop_pct, "Successes / 1,000": succ,
                         "Energy / 1,000 (J)": 1000 * r.J_task, "Goodput (tasks/J)": succ / (1000 * r.J_task)})
        write_table(pd.DataFrame(rows).set_index("Regime").round(2), os.path.join(out, "tableS4_traffic"))
        log(f"[eval] sweeps done ({time.time() - t0:.0f}s)")

    try:
        from .figures import make_figures
        make_figures(out)
    except Exception as e:  # figures are optional
        log(f"[figures] skipped: {e}")
    log(f"[done] results in {out} ({time.time() - t0:.0f}s)")
    return runs_df
