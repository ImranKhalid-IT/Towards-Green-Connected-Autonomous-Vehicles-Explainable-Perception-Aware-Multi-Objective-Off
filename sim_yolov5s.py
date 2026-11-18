#!/usr/bin/env python3
"""Detector-specific simulation for YOLOv5s.

    python scripts/sim_yolov5s.py --runs 30 --tasks 30000            # simulation
    python scripts/sim_yolov5s.py --profile-real --device cuda       # re-measure Table 3 on real hardware
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from greencav.config import load_config
from greencav.detectors import yolov5s as det

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--runs", type=int, default=30)
ap.add_argument("--tasks", type=int, default=30_000)
ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
ap.add_argument("--out", default="results/detectors")
ap.add_argument("--profile-real", action="store_true")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
if a.profile_real:
    print(det.profile_real(device=a.device))
else:
    from greencav.detector_sim import simulate_detector
    simulate_detector(det.PROFILE.name, cfg=load_config(a.config), runs=a.runs, n_tasks=a.tasks,
                      out_dir=a.out, workers=a.workers)
