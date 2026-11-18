#!/usr/bin/env python3
"""Runs the four detector-specific simulations one after another."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from greencav.config import load_config
from greencav.detector_sim import simulate_detector
from greencav.detectors import DETECTOR_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--runs", type=int, default=30)
ap.add_argument("--tasks", type=int, default=30_000)
ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
ap.add_argument("--out", default="results/detectors")
a = ap.parse_args()
for name in DETECTOR_NAMES:
    simulate_detector(name, cfg=load_config(a.config), runs=a.runs, n_tasks=a.tasks, out_dir=a.out, workers=a.workers)
