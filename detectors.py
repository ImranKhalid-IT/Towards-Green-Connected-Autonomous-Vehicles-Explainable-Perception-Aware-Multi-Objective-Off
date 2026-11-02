"""Detector profiling and explicit synthetic perception surrogate.

No images are inferred here. 'accuracy_proxy_pct' is an expected synthetic score,
NOT COCO/AP/mAP: actual mAP requires predicted boxes and ground-truth labels.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .models import DETECTORS

@dataclass(frozen=True)
class DetectorProfile:
    name: str
    local_ms: float
    local_j: float
    edge_ms: float
    edge_j: float
    clear_map_pct: float

PROFILES = (
    DetectorProfile('YOLOv5s',45.2,0.85,12.4,0.25,78.4),
    DetectorProfile('YOLOv8s',52.8,1.05,14.1,0.32,81.2),
    DetectorProfile('YOLOv8m',88.5,1.62,22.3,0.58,84.6),
    DetectorProfile('FasterRCNN',145.0,2.90,45.6,1.15,86.1),
)
# Paper overall loss is weather-dependent; detector-specific losses absent in manuscript.
# These are explicitly assumed profile-dependent proxy losses, NOT measured weather mAP.
LOSS_PCT = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [6.0, 5.7, 5.1, 4.8],
    [18.4, 17.3, 15.2, 13.6],
    [25.4, 24.3, 21.3, 19.6],
],dtype=float)

class SyntheticDetectorBank:
    _dom_grid=np.linspace(1/5,1-1e-9,2001)
    _tail=(1-_dom_grid)/4
    _h_grid=-(_dom_grid*np.log(_dom_grid)+4*_tail*np.log(_tail))/np.log(5)
    def __init__(self,seed):
        self.rng=np.random.default_rng(seed)
    @staticmethod
    def expected_accuracy(detector:int,weather:int) -> float:
        return float(max(0,PROFILES[detector].clear_map_pct-LOSS_PCT[weather,detector]))
    @staticmethod
    def entropy_of_probs(p):
        p=np.asarray(p,dtype=float)
        p=np.clip(p,1e-12,1)
        return float(-np.sum(p*np.log(p))/np.log(len(p)))
    @staticmethod
    def _probs_for_entropy(target,k=5):
        # Invert normalized entropy with a monotonic precomputed probability grid.
        # Identical mathematical construction to bisection, much faster at 30k-task scale.
        if k!=5: raise ValueError('Surrogate class count fixed to 5')
        dominant=float(np.interp(target,SyntheticDetectorBank._h_grid[::-1],SyntheticDetectorBank._dom_grid[::-1]))
        out=np.full(5,(1-dominant)/4)
        out[0]=dominant
        return out
    def predict(self,detector:int,weather:int):
        # Synthetic class probabilities: not predictions from a neural detector.
        baselines=(0.12,0.26,0.43,0.62)
        target=np.clip(baselines[weather]+(2-detector)*0.026 +self.rng.normal(0,0.065),0.015,0.97)
        p=self._probs_for_entropy(target)
        uncertainty=self.entropy_of_probs(p)
        accuracy=self.expected_accuracy(detector,weather)
        return accuracy,uncertainty
    def cost_profile(self,detector,site):
        p=PROFILES[detector]
        return (p.local_ms,p.local_j) if site==0 else (p.edge_ms,p.edge_j)
