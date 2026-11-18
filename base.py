"""Detector profiles (Table 3 of the paper) and the accuracy model of Eq. (13).

A :class:`DetectorProfile` holds the computation-only profile of one detector
and its weather-conditioned accuracy A(m, w).  The per-detector modules
(``yolov5s.py``, ``yolov8s.py``, ``yolov8m.py``, ``faster_rcnn.py``) each
define one profile, a detector-specific simulation and an optional real
profiling routine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..config import WEATHERS


@dataclass(frozen=True)
class DetectorProfile:
    name: str
    params_m: float
    input_px: int
    local_s: float          # T^loc(m), computation only
    edge_s: float           # T^edge(m), computation only
    local_j: float          # E^loc(m)
    edge_j: float           # E^edge(m)
    acc: Sequence[float]    # A(m, w) for clear, light rain, heavy rain, dense smog [mAP@0.5 %]
    uncertainty_scale: float  # model choice: larger models are less uncertain

    # --- Eq. (9) -------------------------------------------------------------
    @property
    def speedup(self) -> float:
        """kappa_m = T^loc / T^edge."""
        return self.local_s / self.edge_s

    def cycles(self, f_loc_hz: float) -> float:
        """C_m such that T^loc(m) = C_m / f_loc (Eq. 9)."""
        return self.local_s * f_loc_hz

    # --- Eq. (10) ------------------------------------------------------------
    def switched_capacitance(self, f_loc_hz: float) -> float:
        """k_c such that E^loc(m) = k_c f_loc^2 C_m (Eq. 10)."""
        return self.local_j / (f_loc_hz ** 2 * self.cycles(f_loc_hz))

    @property
    def edge_power_w(self) -> float:
        """P_edge such that E^edge(m) = P_edge T^edge(m) (Eq. 10)."""
        return self.edge_j / self.edge_s

    # --- Eq. (13) ------------------------------------------------------------
    def accuracy(self, weather: int) -> float:
        """A(m, w) = A_clr(m) - Delta_w(m)."""
        return float(self.acc[weather])

    def weather_loss(self, weather: int) -> float:
        """Delta_w(m) in percentage points (0 for clear weather)."""
        return float(self.acc[0] - self.acc[weather])

    def summary(self) -> Dict[str, float]:
        d = {"model": self.name, "params_M": self.params_m, "local_ms": self.local_s * 1e3,
             "edge_ms": self.edge_s * 1e3, "local_J": self.local_j, "edge_J": self.edge_j,
             "speedup": round(self.speedup, 2)}
        for w, name in enumerate(WEATHERS):
            d[f"acc_{name}"] = self.acc[w]
        return d


# ---------------------------------------------------------------------------
# Optional real-hardware profiling (needs torch + the detector library).
# Not executed by the simulator; provided so that Table 3 can be re-measured.
# ---------------------------------------------------------------------------

def time_inference(run_once: Callable[[], None], warmup: int = 10, iters: int = 100,
                   sync: Optional[Callable[[], None]] = None) -> Dict[str, float]:
    """Wall-clock latency of ``run_once`` (single inference pass, no queueing)."""
    for _ in range(warmup):
        run_once()
    if sync:
        sync()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run_once()
        if sync:
            sync()
        samples.append(time.perf_counter() - t0)
    a = np.asarray(samples)
    return {"mean_ms": 1e3 * a.mean(), "p50_ms": 1e3 * np.median(a), "p95_ms": 1e3 * np.percentile(a, 95)}


def gpu_energy_per_call(run_once: Callable[[], None], iters: int = 100) -> Optional[float]:
    """Energy per call [J] from NVML power readings, or ``None`` if NVML is unavailable.

    On Jetson boards use ``jtop``/tegrastats instead; the integration is left to
    the user because it depends on the board.
    """
    try:
        import pynvml  # type: ignore
    except Exception:
        return None
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    try:
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)  # mJ
        for _ in range(iters):
            run_once()
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        return (e1 - e0) / 1e3 / iters
    finally:
        pynvml.nvmlShutdown()
