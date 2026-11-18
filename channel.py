"""Communication model: Eqs. (3)-(8) of the paper."""
from __future__ import annotations

import math
from typing import List

import numpy as np

from .config import SimConfig


def dbm_to_w(dbm: float) -> float:
    return 10 ** (dbm / 10) / 1000.0


def total_noise_dbm(bandwidth_hz: float, noise_figure_db: float) -> float:
    """Eq. (3): N_tot = -174 + 10 log10(B) + NF  [dBm]."""
    return -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db


def shannon_rate(bandwidth_hz: float, p_tx_dbm: float, gain: float, n_tot_dbm: float, eta_r: float) -> float:
    """Eq. (4): R_s = eta_r B log2(1 + P_tx Gamma_s / N_tot)  [bit/s]."""
    snr = dbm_to_w(p_tx_dbm) * gain / dbm_to_w(n_tot_dbm)
    return eta_r * bandwidth_hz * math.log2(1.0 + snr)


def calibrate_gain(target_rate_bps: float, bandwidth_hz: float, p_tx_dbm: float, n_tot_dbm: float,
                   eta_r: float = 0.6) -> float:
    """Effective channel gain Gamma_s that yields ``target_rate_bps`` in Eq. (4)."""
    snr = 2 ** (target_rate_bps / (eta_r * bandwidth_hz)) - 1.0
    return snr * dbm_to_w(n_tot_dbm) / dbm_to_w(p_tx_dbm)


def transition_matrix(nu: List[float], mixing: float) -> np.ndarray:
    """A Markov chain with stationary distribution ``nu`` (Eq. 5).

    P = (1 - e) I + e 1 nu^T satisfies nu P = nu for any 0 < e <= 1; small e gives
    strongly correlated link quality.
    """
    nu = np.asarray(nu, dtype=float)
    k = len(nu)
    return (1.0 - mixing) * np.eye(k) + mixing * np.tile(nu, (k, 1))


def stationary(P: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
    return v / v.sum()


def min_transfer_s(d_feat_bytes: int, rate_bps: float) -> float:
    return 8.0 * d_feat_bytes / rate_bps


def uplink_delay(d_feat_bytes: int, rate_bps: float, overhead_s: float) -> float:
    """Eq. (6): T^up = 8 D_feat / R_s + T^ovh_s."""
    return min_transfer_s(d_feat_bytes, rate_bps) + overhead_s


def modem_power(p_tx_dbm: float, eta_pa: float, p_circ_w: float) -> float:
    """Eq. (7): P^mod = P_tx / eta_pa + P^circ."""
    return dbm_to_w(p_tx_dbm) / eta_pa + p_circ_w


def radio_energy(p_mod_w: float, t_up_s: float) -> float:
    """Eq. (8): E^tx = P^mod T^up."""
    return p_mod_w * t_up_s


class Channel:
    """Markov-modulated effective V2I link calibrated to Table 2 / Supplementary Table S3."""

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.rates = [r * 1e6 for r in cfg.uplink_rates_mbps]
        self.t_up = [uplink_delay(cfg.d_feat_bytes, r, o) for r, o in zip(self.rates, cfg.link_overhead_s)]
        # P^mod_s implied by the radio energies of Eq. (8); P^circ_s then follows from Eq. (7).
        self.p_mod = [e / t for e, t in zip(cfg.radio_energy_j, self.t_up)]
        self.p_circ = [p - dbm_to_w(cfg.p_tx_dbm) / cfg.eta_pa for p in self.p_mod]
        self.P = transition_matrix(cfg.uplink_shares, cfg.link_mixing)
        self.n_tot_dbm = total_noise_dbm(cfg.bandwidth_hz, cfg.noise_figure_db)
        self.gains = [calibrate_gain(r, cfg.bandwidth_hz, cfg.p_tx_dbm, self.n_tot_dbm) for r in self.rates]
        if cfg.forced_link_state is not None:
            self.state = int(cfg.forced_link_state)
        else:
            self.state = int(rng.choice(len(self.rates), p=cfg.uplink_shares))

    def step(self) -> int:
        """Advance the chain by one task slot.  Two uniforms are drawn every slot so that
        the link process is identical for all policies (common random numbers)."""
        u_state, self._u_fail = self.rng.random(), self.rng.random()
        if self.cfg.forced_link_state is None:
            self.state = int(np.searchsorted(np.cumsum(self.P[self.state]), u_state, side="right"))
            self.state = min(self.state, len(self.rates) - 1)
        return self.state

    _u_fail: float = 1.0

    def transmit(self, s: int):
        """Returns (delay [s], energy [J], failed?) for one latent upload in state ``s``."""
        fail = self._u_fail > self.cfg.link_success[s]
        return self.t_up[s], radio_energy(self.p_mod[s], self.t_up[s]), fail
