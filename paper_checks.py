"""Re-derives every computed number of the paper's tables (Supplementary Section S2).

These checks use only the published table values, so they reproduce exactly:
    python -m greencav.paper_checks
"""
from __future__ import annotations

import math

from .channel import min_transfer_s, total_noise_dbm
from .detectors import DETECTORS, DETECTOR_NAMES

W = [0.87, 0.08, 0.03, 0.02]
TABLE4 = {  # policy: latency, energy kJ, J/task, Wh/h, CO2, quality
    "Local-only": (112.4, 85.6, 2.85, 28.53, 0.19, 79.8), "Edge-only": (78.5, 62.3, 2.08, 20.77, 0.14, 76.2),
    "Greedy heuristic": (65.2, 58.1, 1.94, 19.37, 0.13, 75.3), "Deadline-aware": (61.4, 50.7, 1.69, 16.90, 0.11, 77.0),
    "Energy-aware DRL": (58.7, 45.4, 1.51, 15.13, 0.10, 78.7), "Proposed": (49.3, 38.9, 1.30, 12.97, 0.09, 79.2)}
TABLE5 = {  # mix (%), drop, uncertainty, offload, latency, viol, J/task, quality
    "clear": ([10.7, 89.2, 0.1, 0], 0.4, 0.12, 77.0, 47.3, 0.3, 1.25, 80.6),
    "light_rain": ([6.2, 90.2, 3.6, 0], 0.8, 0.28, 81.0, 55.6, 1.0, 1.48, 75.3),
    "heavy_rain": ([5, 63, 32, 0], 1.9, 0.45, 91.0, 71.5, 4.4, 1.75, 63.6),
    "dense_smog": ([1, 67, 32, 0], 3.0, 0.62, 95.0, 79.2, 7.0, 1.92, 57.1)}


def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"{'OK ' if ok else 'BAD'} {name:<58s} computed={got:10.4f} reported={want}")
    return ok


def main() -> bool:
    ok = True
    ok &= check("Eq.(3) total noise [dBm]", total_noise_dbm(10e6, 9), -95, 0.01)
    for r, want in zip([5e6, 15e6, 30e6], [40.0, 13.3, 6.7]):
        ok &= check(f"Eq.(6) min transfer at {r/1e6:.0f} Mbit/s [ms]", 1e3 * min_transfer_s(25_000, r), want, 0.05)
    ok &= check("mix-weighted V2I latency [ms]", 0.05 * 45.2 + 0.35 * 18.5 + 0.60 * 9.0, 14.1, 0.05)
    sp = [DETECTORS[n].speedup for n in DETECTOR_NAMES]
    ok &= check("min edge speed-up", min(sp), 3.18, 0.005) & check("max edge speed-up", max(sp), 3.97, 0.005)
    # Table 5 reproduction (Supplementary Table S3)
    qs = []
    for k, (wname, (mix, drop, u, off, lat, viol, jt, q)) in enumerate(TABLE5.items()):
        a = sum(mix[j] / 100 * DETECTORS[n].acc[k] for j, n in enumerate(DETECTOR_NAMES))
        qs.append(q)
        ok &= check(f"expected quality {wname} [%]", a * (1 - drop / 100), q, 0.06)
    agg = lambda idx: sum(W[k] * list(TABLE5.values())[k][idx] for k in range(4))
    ok &= check("weighted expected quality [%]", sum(W[k] * qs[k] for k in range(4)), 79.2, 0.05)
    ok &= check("weighted latency [ms]", agg(4), 49.3, 0.05)
    ok &= check("weighted violations [%]", agg(5), 0.6, 0.05)
    ok &= check("weighted offload [%]", agg(3), 78.1, 0.05)
    ok &= check("weighted J/task", agg(6), 1.297, 0.0005)
    ok &= check("weighted uncertainty", agg(2), 0.15, 0.005)
    ok &= check("energy 1.2968 J x 30,000 [kJ]", agg(6) * 30, 38.9, 0.05)
    v8 = sum(W[k] * DETECTORS["YOLOv8s"].acc[k] for k in range(4))
    ok &= check("DRL quality (fixed YOLOv8s, 1.4% drops)", v8 * (1 - 0.014), 78.7, 0.05)
    ok &= check("ablation fixed YOLOv8s (0.9% drops)", v8 * (1 - 0.009), 79.1, 0.05)
    for p, (lat, kj, jt, whh, co2, q) in TABLE4.items():
        ok &= check(f"{p}: Wh/h = kJ/30,000 x 10", kj / 30 * 10, whh, 0.01)
        ok &= check(f"{p}: CO2e = Wh/h / 60 x 0.40", whh / 60 * 0.4, co2, 0.005)
    d, p = TABLE4["Energy-aware DRL"], TABLE4["Proposed"]
    ok &= check("latency gain vs DRL [%]", (d[0] - p[0]) / d[0] * 100, 16.0, 0.05)
    ok &= check("energy gain vs DRL [%]", (d[1] - p[1]) / d[1] * 100, 14.3, 0.05)
    ok &= check("energy reduction vs Local-only [%]", (85.6 - 38.9) / 85.6 * 100, 54.6, 0.05)
    ok &= check("energy reduction vs Edge-only [%]", (62.3 - 38.9) / 62.3 * 100, 37.6, 0.05)
    ok &= check("Cohen d latency vs DRL", 9.4 / math.sqrt((1.9 ** 2 + 1.4 ** 2) / 2), 5.6, 0.05)
    ok &= check("Cohen d energy vs DRL", 6.5 / math.sqrt((1.5 ** 2 + 1.1 ** 2) / 2), 4.9, 0.05)
    ok &= check("normal-approx CI half-width latency [ms]", 1.96 * math.sqrt(1.9 ** 2 / 30 + 1.4 ** 2 / 30), 0.85, 0.01)
    ok &= check("amortised XAI 0.25 x 15.8 [ms]", 0.25 * 15.8, 3.95, 1e-9)
    ok &= check("XAI share of latency [%]", 3.95 / 49.3 * 100, 8.0, 0.05)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
