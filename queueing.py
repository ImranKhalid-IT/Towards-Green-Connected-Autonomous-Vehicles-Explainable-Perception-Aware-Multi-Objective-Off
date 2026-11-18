"""Edge queue and on-vehicle processor: Eqs. (11)-(12)."""
from __future__ import annotations

from collections import deque

import numpy as np

from .config import SimConfig


def queue_estimate(backlog: int, mean_service_s: float, workers: int) -> float:
    """Eq. (11): T^q ~= n^q * mean_service / W_e (Little's law)."""
    return backlog * mean_service_s / workers


def edge_load(offload_rate_hz: float, mean_service_s: float, workers: int) -> float:
    """Eq. (12): rho_e = lambda_e * mean_service / W_e."""
    return offload_rate_hz * mean_service_s / workers


class EdgeServer:
    """FIFO multi-server queue (W_e workers, buffer n_max) shared with background vehicles.

    Background tasks of other vehicles arrive as a Poisson process whose rate gives the
    target edge load of the traffic regime (Supplementary Table S4).  The realised
    queueing delay T^q_i of Eq. (22) comes from this discrete-event model; the policy only
    observes the estimates of Eqs. (11)-(12).
    """

    def __init__(self, cfg: SimConfig, rng: np.random.Generator, horizon_s: float):
        self.cfg = cfg
        self.rng = rng
        self.W = cfg.edge_workers
        self.free = np.zeros(self.W)              # time at which each worker becomes free
        self.starts: deque = deque()              # start times of accepted jobs not yet started
        load = cfg.traffic_loads[cfg.traffic_regime]
        self.bg_rate = load * self.W / cfg.background_service_s
        self.t = 0.0
        self.mean_service = cfg.background_service_s
        self.arrivals_ema = self.bg_rate
        sig = np.sqrt(np.log(1 + cfg.background_service_cv ** 2))
        mu = np.log(cfg.background_service_s) - sig ** 2 / 2
        # The background process is drawn once for the whole run, so every policy
        # evaluated with the same seed faces exactly the same competing traffic.
        # Markov-modulated Poisson background: congestion bursts of load rho_b alternate with
        # normal periods of load rho_n; the time-average load equals the regime load.
        fb = cfg.burst_fraction
        rho_b = min(cfg.burst_load, 2.0 * load)
        rho_n = max((load - fb * rho_b) / (1.0 - fb), 0.0) if fb < 1 else rho_b
        t, burst, times = 0.0, False, []
        mean_normal = cfg.burst_mean_s * (1.0 - fb) / max(fb, 1e-9)
        while t < horizon_s:
            dur = rng.exponential(cfg.burst_mean_s if burst else mean_normal)
            end = min(t + dur, horizon_s)
            lam = (rho_b if burst else rho_n) * self.W / cfg.background_service_s
            n = rng.poisson(lam * (end - t))
            times.append(rng.uniform(t, end, n))
            t, burst = end, not burst
        self.bg_t = np.sort(np.concatenate(times)) if times else np.zeros(0)
        self.bg_s = rng.lognormal(mu, sig, len(self.bg_t))
        self.k = 0
        self._last_count_t, self._count = 0.0, 0

    def _assign(self, t_arr: float, service: float) -> float:
        k = int(np.argmin(self.free))
        start = max(t_arr, self.free[k])
        self.free[k] = start + service
        if start > t_arr:
            self.starts.append(start)
        return start - t_arr

    def backlog(self, t: float) -> int:
        while self.starts and self.starts[0] <= t:
            self.starts.popleft()
        # FIFO multi-server: start times are non-decreasing, so the deque is sorted
        return len(self.starts)

    def advance(self, t: float) -> None:
        """Inject the background arrivals in (self.t, t]."""
        if t <= self.t:
            return
        k0 = self.k
        while self.k < len(self.bg_t) and self.bg_t[self.k] <= t:
            ta = float(self.bg_t[self.k])
            if self.backlog(ta) < self.cfg.edge_buffer:
                self._assign(ta, float(self.bg_s[self.k]))
            self.k += 1
        dt = t - self.t
        self.arrivals_ema = 0.9 * self.arrivals_ema + 0.1 * ((self.k - k0) / dt)
        self.t = t

    def submit(self, t_arr: float, service: float):
        """Own task arriving at ``t_arr``. Returns (queue delay or None if overflow, backlog)."""
        self.advance(t_arr)
        b = self.backlog(t_arr)
        if b >= self.cfg.edge_buffer:
            return None, b
        return self._assign(t_arr, service), b

    def estimates(self, t: float, own_offload_hz: float):
        b = self.backlog(t)
        tq = queue_estimate(b, self.mean_service, self.W)
        rho = edge_load(self.arrivals_ema + own_offload_hz, self.mean_service, self.W)
        return b, tq, rho


class LocalProcessor:
    """Single on-vehicle accelerator with an input buffer and a thermal-throttling state."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.free = 0.0
        self.heat = 0.0
        self.pending: deque = deque()

    def backlog(self, t: float) -> int:
        while self.pending and self.pending[0] <= t:
            self.pending.popleft()
        return len(self.pending)

    def slowdown(self) -> float:
        c = self.cfg
        return 1.0 + c.throttle_gain * max(0.0, self.heat - c.throttle_onset)

    def submit(self, t_arr: float, service: float):
        """Returns (waiting time or None if the input buffer is full, effective service time)."""
        if self.backlog(t_arr) >= self.cfg.local_buffer:
            self.cool(0.0)
            return None, 0.0
        eff = service * self.slowdown()
        start = max(t_arr, self.free)
        self.free = start + eff
        if start > t_arr:
            self.pending.append(start)
        self.cool(min(1.0, eff * self.cfg.f_s))
        return start - t_arr, eff

    def cool(self, util: float) -> None:
        a = self.cfg.throttle_ema
        self.heat = (1 - a) * self.heat + a * util
