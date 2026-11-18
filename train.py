#!/usr/bin/env python3
"""Algorithm 2: train one learned variant.

    python scripts/train.py --variant proposed --steps 2000000
variants: proposed | drl | no_xai | no_energy | no_dyn
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from greencav.config import load_config
from greencav.policies import VARIANT_ALPHA, variant_config
from greencav.ppo import algorithm2_train

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--variant", default="proposed")
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--envs", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="results/models")
a = ap.parse_args()
cfg = variant_config(load_config(a.config), a.variant)
agent, log = algorithm2_train(cfg, steps=a.steps, n_envs=a.envs, seed=a.seed, fixed_alpha=VARIANT_ALPHA.get(a.variant))
os.makedirs(a.out, exist_ok=True)
agent.save(os.path.join(a.out, f"{a.variant}.npz"), meta={"variant": a.variant})
pd.DataFrame({"steps": log.steps, "mean_cost": log.mean_cost, "entropy": log.entropy}).to_csv(
    os.path.join(a.out, f"{a.variant}_trainlog.csv"), index=False)
print("saved", os.path.join(a.out, f"{a.variant}.npz"))
