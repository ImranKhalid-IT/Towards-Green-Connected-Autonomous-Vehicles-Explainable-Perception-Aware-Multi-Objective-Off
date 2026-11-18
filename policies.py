"""Policies compared in the paper (Section III-H): five baselines and the proposed policy."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import SimConfig
from .detectors import DETECTOR_NAMES
from .env import GreenCAVEnv, N_ACTIONS, decode_action, encode_action


class Policy:
    name = "policy"
    cfg: SimConfig
    alpha: Optional[float] = None
    tau: Optional[float] = None

    def act(self, env: GreenCAVEnv, obs: np.ndarray) -> int:
        raise NotImplementedError


class FixedSite(Policy):
    """Local-only (o_i = 0) or Edge-only (o_i = 1) with a fixed detector."""

    def __init__(self, cfg: SimConfig, offload: int, detector: str = "YOLOv8s", name: Optional[str] = None):
        self.cfg = cfg.copy(fixed_detector=detector)
        self.a = encode_action(offload, DETECTOR_NAMES.index(detector))
        self.name = name or ("Edge-only" if offload else "Local-only")

    def act(self, env, obs):
        return self.a


class Greedy(Policy):
    """Minimises the predicted latency T_i of Eq. (22)."""
    name = "Greedy heuristic"

    def __init__(self, cfg: SimConfig, detector: Optional[str] = None):
        self.cfg = cfg.copy(fixed_detector=detector)

    def act(self, env, obs):
        best, best_T = None, np.inf
        for a in np.flatnonzero(env.action_mask):
            T, _ = env.predict(*decode_action(int(a)))
            if T < best_T:
                best, best_T = int(a), T
        return best


class DeadlineAware(Policy):
    """Minimises predicted energy subject to predicted T_i <= D (fallback: minimum latency)."""
    name = "Deadline-aware"

    def __init__(self, cfg: SimConfig, detector: Optional[str] = None, margin: float = 0.9):
        self.cfg = cfg.copy(fixed_detector=detector)
        self.margin = margin

    def act(self, env, obs):
        D = env.cfg.deadline * self.margin
        feas, fall = [], []
        for a in np.flatnonzero(env.action_mask):
            T, E = env.predict(*decode_action(int(a)))
            fall.append((T, int(a)))
            if T <= D:
                feas.append((E, int(a)))
        return min(feas)[1] if feas else min(fall)[1]


class PPOPolicy(Policy):
    """A trained PPO agent (proposed policy, Energy-aware DRL baseline or an ablation)."""

    def __init__(self, agent, cfg: SimConfig, name: str, alpha: Optional[float] = None):
        self.agent = agent
        self.cfg = cfg
        self.name = name
        self.alpha = alpha
        self.tau = agent.tau

    def act(self, env, obs):
        a, _, _ = self.agent.act(obs, greedy=True)       # arg max at test time
        return a


# ---------------------------------------------------------------------------
# Configurations of the learned variants (Section III-H and Table 7)
# ---------------------------------------------------------------------------

def variant_config(cfg: SimConfig, variant: str) -> SimConfig:
    if variant == "proposed":
        return cfg.copy()
    if variant == "drl":            # Energy-aware DRL: fixed YOLOv8s, no weather/uncertainty/XAI
        return cfg.copy(fixed_detector="YOLOv8s", use_weather_input=False, use_xai_input=False, xai_enabled=False)
    if variant == "no_xai":         # without XAI feedback: no (u, r) inputs and no gate
        return cfg.copy(use_xai_input=False, xai_enabled=False)
    if variant == "no_energy":      # without energy objective: alpha = 0
        return cfg.copy()
    if variant == "no_dyn":         # without dynamic perception: fixed YOLOv8s
        return cfg.copy(fixed_detector="YOLOv8s")
    raise KeyError(variant)


VARIANT_ALPHA = {"no_energy": 0.0}
VARIANT_NAMES = {"proposed": "Proposed", "drl": "Energy-aware DRL", "no_xai": "Without XAI feedback",
                 "no_energy": "Without energy objective", "no_dyn": "Without dynamic perception (fixed YOLOv8s)"}
