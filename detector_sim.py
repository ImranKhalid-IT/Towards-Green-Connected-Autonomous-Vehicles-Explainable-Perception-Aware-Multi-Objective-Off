"""Detector-specific simulation shared by the four detector modules.

For one detector m (fixed for every task) it compares where to run it:
Local-only, Edge-only, the Greedy heuristic and the Deadline-aware heuristic,
over paired seeds of the mixed-weather workload, and writes
  <out_dir>/<detector>_policies.{csv,md}  (latency, energy, quality, ...)
  <out_dir>/<detector>_weather.{csv,md}   (per-weather breakdown, Deadline-aware)
"""
from __future__ import annotations

import os
from typing import Optional

from .config import SimConfig
from .detectors import DETECTORS
from .evaluate import algorithmS2_evaluate, table_policies, table_weather, write_table
from .policies import DeadlineAware, FixedSite, Greedy


def simulate_detector(name: str, cfg: Optional[SimConfig] = None, runs: int = 5, n_tasks: int = 30_000,
                      seed0: int = 0, out_dir: str = "results/detectors", workers: int = 1, verbose: bool = True):
    cfg = cfg or SimConfig()
    if name not in DETECTORS:
        raise KeyError(name)
    pols = [FixedSite(cfg, 0, name, name=f"Local-only [{name}]"),
            FixedSite(cfg, 1, name, name=f"Edge-only [{name}]"),
            Greedy(cfg, detector=name), DeadlineAware(cfg, detector=name)]
    pols[2].name, pols[3].name = f"Greedy [{name}]", f"Deadline-aware [{name}]"
    runs_df, weather_df = algorithmS2_evaluate(pols, range(seed0, seed0 + runs), n_tasks, workers=workers)
    tag = name.replace(" ", "_").replace("-", "").lower()
    os.makedirs(out_dir, exist_ok=True)
    t1 = table_policies(runs_df)
    t2 = table_weather(weather_df, pols[3].name)
    write_table(t1, os.path.join(out_dir, f"{tag}_policies"))
    write_table(t2, os.path.join(out_dir, f"{tag}_weather"))
    runs_df.to_csv(os.path.join(out_dir, f"{tag}_runs.csv"), index=False)
    if verbose:
        print(f"\n=== Detector-specific simulation: {name} ({runs} runs x {n_tasks} tasks) ===")
        print(DETECTORS[name].summary())
        print(t1.to_string())
        print(t2.to_string())
    return runs_df, weather_df
