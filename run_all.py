#!/usr/bin/env python3
"""Full reproduction pipeline: train all variants, evaluate, write tables and figures.

    python scripts/run_all.py --quick     # ~15-25 min on 2 CPU cores
    python scripts/run_all.py             # paper budget: 2e6 steps per variant, 30 runs x 30,000 tasks
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from greencav.config import load_config
from greencav.pipeline import run_all

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--out", default="results")
ap.add_argument("--quick", action="store_true", help="2e5 steps, 5 runs x 5,000 tasks")
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--runs", type=int, default=None)
ap.add_argument("--tasks", type=int, default=None)
ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
ap.add_argument("--no-sweeps", action="store_true")
a = ap.parse_args()
steps, runs, tasks = a.steps, a.runs, a.tasks
if a.quick:
    steps, runs, tasks = steps or 200_000, runs or 5, tasks or 5_000
run_all(load_config(a.config), out=a.out, steps=steps, runs=runs, tasks=tasks, workers=a.workers, sweeps=not a.no_sweeps)
