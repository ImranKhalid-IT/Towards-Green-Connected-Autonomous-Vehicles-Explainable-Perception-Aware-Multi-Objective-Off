"""Perception uncertainty and uncertainty-gated dual-head XAI: Eqs. (14)-(21), Algorithm 1.

The first part implements the equations on real detector outputs (class
distributions, saliency maps, boxes) and can be used with any detector.  The
second part (:class:`SyntheticPerception`) generates the same signals inside
the simulator, where no images are rendered.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Equations on real detector outputs
# ---------------------------------------------------------------------------


def normalised_entropy(p: np.ndarray) -> float:
    """h_b = H(p_b) / log K."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def predictive_uncertainty(class_probs: Sequence[np.ndarray]) -> float:
    """Eq. (14): mean normalised entropy over the detections; 1 if there are none."""
    if len(class_probs) == 0:
        return 1.0
    return float(np.mean([normalised_entropy(p) for p in class_probs]))


def box_mask(shape: Tuple[int, int], boxes: Iterable[Sequence[float]]) -> np.ndarray:
    """Omega_i: pixels inside the predicted boxes (x1, y1, x2, y2 in pixels)."""
    m = np.zeros(shape, dtype=bool)
    for x1, y1, x2, y2 in boxes:
        m[int(max(0, y1)):int(min(shape[0], y2)), int(max(0, x1)):int(min(shape[1], x2))] = True
    return m


def alignment_reliability(saliency: np.ndarray, boxes: Iterable[Sequence[float]], eps: float = 1e-8) -> float:
    """Eq. (15): r_i = sum_{p in Omega} S(p) / (sum_p S(p) + eps), with S = ReLU(g(F))."""
    s = np.maximum(np.asarray(saliency, dtype=float), 0.0)
    m = box_mask(s.shape, boxes)
    return float(s[m].sum() / (s.sum() + eps))


def fuse(u_pred: float, r: float, omega: float, gate: int) -> float:
    """Eq. (16): u = x * [(1-omega) u_pred + omega (1-r)] + (1-x) u_pred."""
    u_tilde = (1.0 - omega) * u_pred + omega * (1.0 - r)
    return gate * u_tilde + (1 - gate) * u_pred


def gate(u_pred: float, tau: float) -> int:
    """Eq. (17): x_i = 1[u_pred >= tau]."""
    return int(u_pred >= tau)


def calibrate_threshold(u_val: np.ndarray, budget: float) -> float:
    """Eq. (18): tau = (1 - rho_x)-quantile of u_pred on a validation split."""
    return float(np.quantile(np.asarray(u_val, dtype=float), 1.0 - budget))


def amortised_cost(gates: np.ndarray, t_call: float) -> Tuple[float, float]:
    """Eq. (19) / (R5): mean XAI cost per task and realised invocation share."""
    rho = float(np.mean(gates)) if len(gates) else 0.0
    return rho * t_call, rho


def top_mask(saliency: np.ndarray, frac: float) -> np.ndarray:
    """M^(z): binary mask of the top-z fraction of salient pixels."""
    s = np.asarray(saliency, dtype=float).ravel()
    k = int(round(frac * s.size))
    m = np.zeros(s.size, dtype=bool)
    if k > 0:
        m[np.argpartition(-s, k - 1)[:k]] = True
    return m.reshape(np.shape(saliency))


def _apply(x: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    return x * (mask2d[..., None] if x.ndim == 3 else mask2d)


def fidelity(model: Callable[[np.ndarray], float], images: Sequence[np.ndarray],
             saliencies: Sequence[np.ndarray], frac: float = 0.2) -> float:
    """Eq. (20): mean normalised confidence drop after removing the top-z0 salient pixels.

    ``model(x)`` must return the confidence of the originally predicted class.
    """
    drops = []
    for x, s in zip(images, saliencies):
        p0 = model(x)
        p1 = model(_apply(x, ~top_mask(s, frac)))
        drops.append(1.0 - p1 / max(p0, 1e-12))
    return float(np.mean(drops))


def deletion_insertion_auc(model: Callable[[np.ndarray], float], x: np.ndarray, saliency: np.ndarray,
                           steps: int = 20, baseline: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Eq. (21): AUC of the confidence while deleting / inserting salient pixels."""
    base = np.zeros_like(x) if baseline is None else baseline
    zs = np.linspace(0.0, 1.0, steps + 1)
    dele, ins = [], []
    for z in zs:
        m = top_mask(saliency, z)
        dele.append(model(_apply(x, ~m) + _apply(base, m)))
        ins.append(model(_apply(x, m) + _apply(base, ~m)))
    return float(np.trapezoid(dele, zs)), float(np.trapezoid(ins, zs))


# ---------------------------------------------------------------------------
# Algorithm 1 for one task (works with real or synthetic signals)
# ---------------------------------------------------------------------------

@dataclass
class XAIOutput:
    u: float
    r: float
    x: int
    u_pred: float


def algorithm1_gated_xai(u_pred: float, tau: float, omega: float, r_prev: float,
                         reliability_fn: Callable[[], float], enabled: bool = True) -> XAIOutput:
    """Algorithm 1: uncertainty-gated dual-head XAI feedback for one task.

    ``reliability_fn`` runs the explanation head and returns r_i (Eq. 15); it is only
    called when the gate opens, which is what makes the overhead amortised.
    """
    x = gate(u_pred, tau) if enabled else 0
    if x:
        r = reliability_fn()
        u = fuse(u_pred, r, omega, 1)
    else:
        r, u = r_prev, u_pred
    return XAIOutput(u=u, r=r, x=x, u_pred=u_pred)


# ---------------------------------------------------------------------------
# Synthetic perception signals for the simulator
# ---------------------------------------------------------------------------

class SyntheticPerception:
    """Generates u_pred and r for a detector/weather pair without rendering images.

    u_pred ~ Beta with mean base_w * scale_m (base_w = 0.12/0.28/0.45/0.62, Table 5);
    r ~ Beta with a mean that falls with the detector's accuracy A(m, w), so that low
    reliability signals frames the detector is likely to get wrong.  These shapes are
    model choices of the simulator.
    """

    def __init__(self, base: Sequence[float], concentration: float, rng: np.random.Generator):
        self.base = np.asarray(base, dtype=float)
        self.k = concentration
        self.rng = rng

    def _beta(self, mean: float) -> float:
        mean = float(np.clip(mean, 0.02, 0.98))
        return float(self.rng.beta(self.k * mean, self.k * (1 - mean)))

    def uncertainty(self, weather: int, scale: float) -> float:
        return self._beta(self.base[weather] * scale)

    def reliability(self, accuracy_pct: float) -> float:
        return self._beta(0.95 - 0.9 * (1.0 - accuracy_pct / 100.0) * 1.6)
