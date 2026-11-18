"""Supplementary Algorithm S2: paired evaluation, metric aggregation and tables."""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import WEATHERS, SimConfig
from .detectors import DETECTOR_NAMES
from .env import GreenCAVEnv
from .policies import Policy
from .stats import paired_test

FIELDS = ["T", "E_veh", "E_sys", "drop", "viol", "q", "o", "m", "x", "w", "s", "cost", "u"]


def run_episode(policy: Policy, seed: int, n_tasks: int, overrides: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """One run of one policy (Algorithm S2, line 5); returns the per-task log."""
    cfg = policy.cfg.copy(**(overrides or {}))
    alpha = policy.alpha if policy.alpha is not None else cfg.alpha
    env = GreenCAVEnv(cfg, seed=seed, n_tasks=n_tasks, alpha=alpha)
    if policy.tau is not None:
        env.tau = policy.tau
    obs = env.reset(seed)
    rows, done = [], False
    while not done:
        obs, _, done, out = env.step(policy.act(env, obs))
        rows.append((out.T, out.E_veh, out.E_sys, out.drop, out.viol, out.q, out.o, out.m, out.x, out.w, out.s, out.cost, out.u_pred))
    a = np.asarray(rows, dtype=float)
    return {k: a[:, j] for j, k in enumerate(FIELDS)}


def summarise(log: Dict[str, np.ndarray], cfg: SimConfig) -> Dict[str, float]:
    """Run-level metrics (Algorithm S2, line 6) and vehicle-level quantities of Eq. (32)."""
    ok = log["drop"] == 0
    T = log["T"][ok]
    J = float(log["E_sys"].mean())
    Jveh = float(log["E_veh"].mean())
    wh_h = J * cfg.f_s                      # J/task * tasks/s = W = Wh per driving hour
    veh_wh_h = Jveh * cfg.f_s
    trac_wh_h = cfg.traction_wh_per_km * cfg.speed_kmh
    d = {"latency_ms": 1e3 * T.mean() if len(T) else np.nan,
         "p95_ms": 1e3 * np.percentile(T, 95) if len(T) else np.nan,
         "viol_pct": 100 * log["viol"].mean(), "drop_pct": 100 * log["drop"].mean(),
         "offload_pct": 100 * log["o"].mean(), "energy_kJ": log["E_sys"].sum() / 1e3,
         "J_task": J, "Wh_h": wh_h, "veh_Wh_h": veh_wh_h,
         "veh_share_pct": 100 * veh_wh_h / trac_wh_h, "range_loss_km": cfg.range_km * veh_wh_h / trac_wh_h,
         "co2_g_km": wh_h / cfg.speed_kmh * cfg.grid_g_per_wh,
         "quality_pct": 100 * log["q"].mean(), "xai_share_pct": 100 * log["x"].mean(),
         "xai_ms_task": 1e3 * log["x"].mean() * cfg.xai_call_s, "cost": log["cost"].mean()}
    for k, name in enumerate(DETECTOR_NAMES):
        d[f"mix_{name}"] = 100 * float((log["m"] == k).mean())
    return d


def summarise_weather(log: Dict[str, np.ndarray], cfg: SimConfig) -> List[Dict[str, float]]:
    out = []
    for w, name in enumerate(WEATHERS):
        sel = log["w"] == w
        if not sel.any():
            continue
        sub = {k: v[sel] for k, v in log.items()}
        d = summarise(sub, cfg)
        d.update(weather=name, tasks=int(sel.sum()), uncertainty=float(sub["u"].mean()))
        out.append(d)
    return out


def _job(args):
    policy, seed, n_tasks, overrides = args
    log = run_episode(policy, seed, n_tasks, overrides)
    cfg = policy.cfg.copy(**(overrides or {}))
    s = summarise(log, cfg)
    s.update(policy=policy.name, seed=seed)
    ws = summarise_weather(log, cfg)
    for r in ws:
        r.update(policy=policy.name, seed=seed)
    return s, ws


def algorithmS2_evaluate(policies: Sequence[Policy], seeds: Sequence[int], n_tasks: int,
                         overrides: Optional[dict] = None, workers: int = 1):
    """Paired evaluation: every policy sees the same seeds (Algorithm S2, lines 1-9).

    Returns (run-level DataFrame, per-weather DataFrame)."""
    jobs = [(p, sd, n_tasks, overrides) for sd in seeds for p in policies]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_job, jobs))
    else:
        res = [_job(j) for j in jobs]
    runs = pd.DataFrame([r[0] for r in res])
    weather = pd.DataFrame([w for r in res for w in r[1]])
    return runs, weather


# ---------------------------------------------------------------------------
# Tables (same layout as the paper)
# ---------------------------------------------------------------------------

def _ms(df: pd.DataFrame, col: str, nd: int = 1) -> pd.Series:
    g = df.groupby("policy", sort=False)[col]
    return g.mean().round(nd).astype(str) + " ± " + g.std().round(nd).astype(str)


def table_policies(runs: pd.DataFrame) -> pd.DataFrame:
    """Analogue of Table 4 (policy comparison, mean ± SD over runs)."""
    g = runs.groupby("policy", sort=False)
    t = pd.DataFrame({
        "Latency (ms)": _ms(runs, "latency_ms"), "P95 (ms)": g["p95_ms"].mean().round(1),
        "Viol. (%)": g["viol_pct"].mean().round(1), "Drop (%)": g["drop_pct"].mean().round(1),
        "Offl. (%)": g["offload_pct"].mean().round(1), "Energy (kJ)": _ms(runs, "energy_kJ"),
        "J/task": g["J_task"].mean().round(2), "Wh/h": g["Wh_h"].mean().round(2),
        "CO2e (g/km)": g["co2_g_km"].mean().round(2), "Quality proxy (%)": _ms(runs, "quality_pct")})
    return t


def table_weather(weather: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Analogue of Table 5 (weather-conditioned behaviour of one policy)."""
    w = weather[weather.policy == policy]
    g = w.groupby("weather", sort=False)
    mix = g[[f"mix_{n}" for n in DETECTOR_NAMES]].mean().round(1)
    t = pd.DataFrame({
        "Tasks": g["tasks"].mean().round(0).astype(int),
        "Mix v5s/v8s/v8m/FRCNN (%)": mix.astype(str).agg(" / ".join, axis=1),
        "Expected quality (%)": g["quality_pct"].mean().round(1),
        "Drop (%)": g["drop_pct"].mean().round(1), "Offload (%)": g["offload_pct"].mean().round(1),
        "Latency (ms)": g["latency_ms"].mean().round(1), "Viol. (%)": g["viol_pct"].mean().round(1),
        "Uncertainty": g["uncertainty"].mean().round(2), "J/task": g["J_task"].mean().round(3)})
    t["Loss (pp)"] = (t["Expected quality (%)"].iloc[0] - t["Expected quality (%)"]).round(1)
    return t


def table_variants(runs: pd.DataFrame, full: str) -> pd.DataFrame:
    """Analogue of Table 7 (preference modes and ablations relative to the full model)."""
    g = runs.groupby("policy", sort=False)
    t = pd.DataFrame({"Latency (ms)": g["latency_ms"].mean().round(1), "Energy (kJ)": g["energy_kJ"].mean().round(1),
                      "Drop (%)": g["drop_pct"].mean().round(1), "Viol. (%)": g["viol_pct"].mean().round(1),
                      "Expected quality (%)": g["quality_pct"].mean().round(1)})
    e0, q0 = t.loc[full, "Energy (kJ)"], t.loc[full, "Expected quality (%)"]
    t["Energy change vs full (%)"] = ((t["Energy (kJ)"] - e0) / e0 * 100).round(1)
    t["Quality change vs full (pp)"] = (t["Expected quality (%)"] - q0).round(1)
    return t


def table_stats(runs: pd.DataFrame, proposed: str, baselines: Sequence[str],
                metrics=(("latency_ms", "Latency (ms)"), ("energy_kJ", "Energy (kJ)"), ("quality_pct", "Quality (pp)"))) -> pd.DataFrame:
    """Analogue of Supplementary Table S7 (paired Wilcoxon tests)."""
    rows = []
    for b in baselines:
        for col, label in metrics:
            pa = runs[runs.policy == proposed].sort_values("seed")[col].to_numpy()
            pb = runs[runs.policy == b].sort_values("seed")[col].to_numpy()
            r = paired_test(pa, pb)
            r.update(comparison=f"{proposed} vs {b}", metric=label)
            rows.append(r)
    return pd.DataFrame(rows).set_index(["comparison", "metric"]).round(4)


def write_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path + ".csv")
    try:
        text = df.to_markdown()
    except ImportError:            # 'tabulate' not installed
        text = df.to_string()
    with open(path + ".md", "w") as f:
        f.write(text + "\n")
