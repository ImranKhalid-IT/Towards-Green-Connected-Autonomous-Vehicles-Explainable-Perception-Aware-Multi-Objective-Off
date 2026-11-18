"""GreenCAV environment: one CAV, one RSU-assisted MEC server, Eqs. (1)-(27).

``GreenCAVEnv.step`` is Supplementary Algorithm S1 (link-aware split-inference
execution of one task): it builds the state, applies the action (o_i, m_i),
executes the task locally or on the edge, runs Algorithm 1 on the executing
node and returns the cost of Eq. (25) as the negative reward of Eq. (27).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .channel import Channel
from .config import SimConfig
from .detectors import DETECTORS, DETECTOR_NAMES
from .queueing import EdgeServer, LocalProcessor
from .weather import release_times, weather_sequence, weather_estimate
from .xai import SyntheticPerception, algorithm1_gated_xai, calibrate_threshold

N_DET = len(DETECTOR_NAMES)
N_ACTIONS = 2 * N_DET          # a = o * 4 + m
OBS_DIM = 13


def encode_action(o: int, m: int) -> int:
    return o * N_DET + m


def decode_action(a: int):
    return a // N_DET, a % N_DET


@dataclass
class TaskOutcome:
    """Everything logged for one task (Supplementary Algorithm S2, line 5)."""
    T: float
    E_veh: float
    E_sys: float
    drop: int
    viol: int
    q: float
    o: int
    m: int
    x: int
    w: int
    s: int
    cost: float
    u_pred: float


class GreenCAVEnv:
    def __init__(self, cfg: SimConfig, seed: int = 0, n_tasks: Optional[int] = None, alpha: Optional[float] = None):
        self.cfg = cfg
        self.base_seed = seed
        self.n_tasks = n_tasks or cfg.tasks_per_run
        self.alpha = cfg.alpha if alpha is None else alpha
        self.dets = [DETECTORS[n] for n in DETECTOR_NAMES]
        mask = np.ones(N_ACTIONS, dtype=bool)
        if cfg.fixed_detector is not None:
            k = DETECTOR_NAMES.index(cfg.fixed_detector)
            mask[:] = False
            mask[encode_action(0, k)] = mask[encode_action(1, k)] = True
        self.action_mask = mask
        frcnn = DETECTORS["Faster R-CNN"]
        self.E_ref = frcnn.local_j + cfg.e_pre + cfg.p_static * cfg.deadline   # Eq. (25) normaliser
        self.tau = None
        self.reset(seed)

    # ------------------------------------------------------------------ setup
    def calibrate_tau(self, u_values: Optional[np.ndarray] = None, n: int = 5000) -> float:
        """Eq. (18) on a validation split (default: YOLOv8s on the weather mixture)."""
        if u_values is None:
            rng = np.random.default_rng(10_000 + self.base_seed)
            per = SyntheticPerception(self.cfg.uncertainty_base, self.cfg.uncertainty_concentration, rng)
            ws = rng.choice(4, size=n, p=np.asarray(self.cfg.weather_shares) / sum(self.cfg.weather_shares))
            scale = DETECTORS["YOLOv8s"].uncertainty_scale
            u_values = np.array([per.uncertainty(int(w), scale) for w in ws])
        self.tau = calibrate_threshold(u_values, self.cfg.xai_budget)
        return self.tau

    def reset(self, seed: Optional[int] = None, alpha: Optional[float] = None) -> np.ndarray:
        cfg = self.cfg
        seed = self.base_seed if seed is None else seed
        if alpha is not None:
            self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        # independent sub-streams so that different policies see identical exogenous processes
        ss = np.random.SeedSequence(seed).spawn(6)
        (self.rng_w, self.rng_link, self.rng_edge, self.rng_per,
         self.rng_misc, self.rng_jit) = [np.random.default_rng(s) for s in ss]
        self.t, self.t_dl = release_times(self.n_tasks, cfg.f_s, cfg.deadline)
        self.w = weather_sequence(cfg, self.n_tasks, self.rng_w)
        self.channel = Channel(cfg, self.rng_link)
        self.edge = EdgeServer(cfg, self.rng_edge, horizon_s=self.n_tasks / cfg.f_s + 1.0)
        self.local = LocalProcessor(cfg)
        self.per = SyntheticPerception(cfg.uncertainty_base, cfg.uncertainty_concentration, self.rng_per)
        if self.tau is None:
            self.calibrate_tau()
        self.i = 0
        self.u_prev, self.r_prev = 0.0, 1.0
        self.prev_rate = cfg.uplink_rates_mbps[self.channel.state]
        self.offload_ema = 0.5
        self.u_log: List[float] = []
        self._prepare_task()
        return self.obs

    def _prepare_task(self) -> None:
        cfg = self.cfg
        self.s_now = self.channel.step()
        self.pw = weather_estimate(int(self.w[self.i]), cfg.weather_classifier_acc, self.rng_misc)
        t = float(self.t[self.i])
        self.edge.advance(t)
        b, self.tq_hat, self.rho_hat = self.edge.estimates(t, self.offload_ema * cfg.f_s)
        self.local_backlog = self.local.backlog(t)
        self.obs = self._observe()

    def _observe(self) -> np.ndarray:
        cfg = self.cfg
        pw = self.pw if cfg.use_weather_input else np.zeros(4)
        u, r = (self.u_prev, self.r_prev) if cfg.use_xai_input else (0.0, 0.0)
        return np.array([*pw, u, r,
                         self.prev_rate / max(cfg.uplink_rates_mbps),
                         self.tq_hat / cfg.deadline,
                         min(self.rho_hat, 1.5),
                         self.local_backlog / max(cfg.local_buffer, 1),
                         self.local.slowdown() - 1.0,
                         self.alpha, 1.0 - self.alpha], dtype=np.float64)

    # ------------------------------------------------------------ predictions
    def predict(self, o: int, m: int):
        """Deterministic latency/energy prediction used by the heuristic baselines."""
        cfg, d = self.cfg, self.dets[m]
        t = float(self.t[self.i])
        if o == 0:
            wait = max(0.0, self.local.free - (t + cfg.t_pre))
            T = cfg.t_pre + cfg.t_dec + wait + d.local_s * self.local.slowdown()
            E = cfg.e_pre + d.local_j + cfg.p_static * T
        else:
            s = int(np.argmin(np.abs(np.array(cfg.uplink_rates_mbps) - self.prev_rate)))
            T = cfg.t_pre + cfg.t_dec + self.channel.t_up[s] + self.tq_hat + d.edge_s + cfg.t_ret
            E = cfg.e_pre + cfg.radio_energy_j[s] + cfg.p_static * T + d.edge_j
        return T, E

    # ------------------------------------------------------------------- step
    def step(self, action: int):
        """Supplementary Algorithm S1 for task i."""
        cfg = self.cfg
        if not self.action_mask[action]:
            raise ValueError("action not allowed by the detector mask")
        o, m = decode_action(int(action))
        d = self.dets[m]
        i, w, s = self.i, int(self.w[self.i]), self.s_now
        t = float(self.t[i])
        jit = float(self.rng_jit.lognormal(-cfg.compute_jitter ** 2 / 2, cfg.compute_jitter))

        drop, T_exec, E_veh_extra, E_edge, E_tx = 0, 0.0, 0.0, 0.0, 0.0
        if o == 0:                                   # local execution, Eqs. (9)-(10)
            wait, eff = self.local.submit(t + cfg.t_pre + cfg.t_dec, d.local_s * jit)
            if wait is None:
                drop = 1                             # zeta^q: vehicle input buffer full
            else:
                T_exec = wait + eff
                E_veh_extra = d.local_j
        else:                                        # offloading, Eqs. (6), (8), (11)
            self.local.cool(0.0)
            t_up, E_tx, fail = self.channel.transmit(s)
            if fail:
                drop = 1                             # zeta^lnk
            else:
                qd, _ = self.edge.submit(t + cfg.t_pre + cfg.t_dec + t_up, d.edge_s * jit)
                if qd is None:
                    drop = 1                         # zeta^q: MEC buffer overflow
                else:
                    T_exec = t_up + qd + d.edge_s * jit + cfg.t_ret
                    E_edge = d.edge_j
            if drop and not fail:
                T_exec = t_up

        # Algorithm 1 on the executing node
        u_pred = self.per.uncertainty(w, d.uncertainty_scale)
        if not drop:
            acc = d.accuracy(w)
            xo = algorithm1_gated_xai(u_pred, self.tau, cfg.xai_fusion_weight, self.r_prev,
                                      lambda: self.per.reliability(acc), enabled=cfg.xai_enabled)
            self.u_log.append(u_pred)
        else:
            xo = algorithm1_gated_xai(u_pred, self.tau, cfg.xai_fusion_weight, self.r_prev,
                                      lambda: 0.0, enabled=False)
        x = xo.x

        # Eq. (22) latency
        T = cfg.t_pre + cfg.t_dec + x * cfg.xai_call_s + T_exec
        if drop:
            T = cfg.deadline                         # a dropped task is lost at its deadline
        # Eq. (23) energy
        E_veh = cfg.e_pre + (1 - o) * (E_veh_extra + x * cfg.e_xai_local) + o * E_tx + cfg.p_static * T
        E_sys = E_veh + o * (E_edge + x * cfg.e_xai_edge)
        # Eq. (24) drop / violation
        viol = int((1 - drop) * (T > cfg.deadline))
        # Eq. (25) cost
        q = (1 - drop) * d.accuracy(w) / 100.0
        a, b = self.alpha, 1.0 - self.alpha
        cost = a * E_sys / self.E_ref + b * T / cfg.deadline + cfg.psi * (1 - q) + cfg.lambda_d * viol + cfg.lambda_p * drop

        out = TaskOutcome(T=T, E_veh=E_veh, E_sys=E_sys, drop=drop, viol=viol, q=q, o=o, m=m, x=x,
                          w=w, s=s, cost=cost, u_pred=u_pred)
        # feedback for the next state
        self.u_prev, self.r_prev = xo.u, xo.r
        self.prev_rate = cfg.uplink_rates_mbps[s]       # link-state beacon of the last slot
        self.offload_ema = 0.95 * self.offload_ema + 0.05 * o
        self.i += 1
        done = self.i >= self.n_tasks
        if not done:
            self._prepare_task()
        return self.obs, -cost, done, out          # Eq. (27): R_t = -c_t
