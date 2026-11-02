"""Shared task/action/telemetry records; all times in milliseconds except global timestamps in seconds."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

WEATHERS = ('clear', 'light_rain', 'heavy_rain', 'dense_smog')
LINKS = ('poor', 'medium', 'good')
DETECTORS = ('YOLOv5s', 'YOLOv8s', 'YOLOv8m', 'FasterRCNN')

@dataclass(frozen=True)
class Action:
    site: int
    detector: int
    @classmethod
    def from_id(cls, value: int) -> 'Action':
        if not 0 <= int(value) < 8:
            raise ValueError('Action must be in [0,7]')
        return cls(int(value)//4, int(value)%4)
    @property
    def index(self) -> int:
        return self.site*4+self.detector

@dataclass(frozen=True)
class Task:
    id: int
    timestamp_s: float
    vehicle_id: int
    weather: int
    link: int
    density: float
    x: float
    y: float

@dataclass
class Outcome:
    task: int
    seed: int
    vehicle_id: int
    timestamp_s: float
    weather: str
    link: str
    detector: str
    offloaded: int
    x: float
    y: float
    density: float
    link_rate_mbps: float
    queue_ms: float
    queue_depth: int
    latency_ms: float
    vehicle_j: float
    system_j: float
    accuracy_proxy_pct: float
    uncertainty: float
    reliability: float
    xai_invoked: int
    dropped: int
    deadline_violation: int
    cost: float
    reward: float
    def to_dict(self) -> dict:
        return asdict(self)

# State: four soft weather estimates, previous (uncertainty,reliability),
# normalized link, estimated edge queue, estimated edge utilization, local backlog, alpha,beta
STATE_DIM = 12

def validate_state(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (STATE_DIM,) or not np.all(np.isfinite(state)):
        raise ValueError('Expected 12 finite state features')
    return state
