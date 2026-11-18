"""YOLOv8m detector: profile, detector-specific simulation and optional real profiling.

Profile values are Table 3 of the paper (computation-only, single inference pass;
accuracy = mAP@0.5 for clear, light rain, heavy rain, dense smog).

Run the detector-specific simulation:
    python -m greencav.detectors.yolov8m --runs 5 --tasks 30000
"""
from __future__ import annotations

import argparse

from .base import DetectorProfile, time_inference, gpu_energy_per_call

PROFILE = DetectorProfile(
    name="YOLOv8m", params_m=25.9, input_px=640,
    local_s=0.0885, edge_s=0.0223, local_j=1.62, edge_j=0.58,
    acc=(84.6, 79.6, 68.1, 61.8), uncertainty_scale=0.85,
)


def simulate(cfg=None, runs: int = 5, n_tasks: int = 30_000, seed0: int = 0, out_dir: str = "results/detectors"):
    """Simulation with YOLOv8m fixed as the only detector.

    Compares Local-only, Edge-only and a deadline-aware site selector, all using
    YOLOv8m, under the mixed-weather workload; writes per-policy and per-weather
    tables to ``out_dir``.
    """
    from ..detector_sim import simulate_detector
    return simulate_detector(PROFILE.name, cfg=cfg, runs=runs, n_tasks=n_tasks, seed0=seed0, out_dir=out_dir)


def load_model(weights: str = "yolov8m.pt", device: str = "cuda"):
    """Load the real detector (requires torch + ultralytics)."""
    import torch
    from ultralytics import YOLO
    model = YOLO(weights)
    x = torch.zeros(1, 3, 640, 640, device=device)

    def run_once():
        model.predict(x, imgsz=640, device=device, verbose=False)
    return model, run_once


def profile_real(weights: str = "yolov8m.pt", device: str = "cuda", iters: int = 100):
    """Re-measure the computation-only profile of Table 3 on the current machine.

    Returns mean/P50/P95 latency of a single inference pass on a 640x640 input
    and, if NVML is available, the GPU energy per call. Not run by the simulator.
    """
    import torch  # noqa: F401
    model, run_once = load_model(weights, device)
    sync = (lambda: __import__("torch").cuda.synchronize()) if device.startswith("cuda") else None
    res = time_inference(run_once, iters=iters, sync=sync)
    res["energy_J"] = gpu_energy_per_call(run_once, iters=iters)
    res["model"] = PROFILE.name
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tasks", type=int, default=30_000)
    ap.add_argument("--out", default="results/detectors")
    ap.add_argument("--profile-real", action="store_true", help="measure real latency instead of simulating")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.profile_real:
        print(profile_real(device=a.device))
    else:
        simulate(runs=a.runs, n_tasks=a.tasks, out_dir=a.out)
